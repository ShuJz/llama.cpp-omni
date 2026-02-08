import asyncio
import base64
import json
import logging
import os
import sys
import time
import wave
import contextlib
import tempfile
import aiohttp
import webrtcvad
import numpy as np
import uuid
from fractions import Fraction
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole
import av
from typing import Optional
try:
    from .output_utils import clear_output_dir
except ImportError:
    from output_utils import clear_output_dir

# Configuration
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
MODEL_DIR = os.environ.get("MODEL_DIR", "/mnt/e/Mochi/Workspace/minCPM/MiniCPM-o-4_5-gguf")
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", ".."))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(PROJECT_ROOT, "omni_output"))
REF_AUDIO_PATH = os.environ.get(
    "REF_AUDIO_PATH",
    os.path.join(PROJECT_ROOT, "llama.cpp-omni", "tools", "omni", "assets", "default_ref_audio.wav"),
)
OMNI_USE_TTS = os.environ.get("OMNI_USE_TTS", "1").lower() in ("1", "true", "yes", "y")
OMNI_AUTO_SYSTEM_PROMPT = os.environ.get("OMNI_AUTO_SYSTEM_PROMPT", "1").lower() in ("1", "true", "yes", "y")
OMNI_MEDIA_TYPE = int(os.environ.get("OMNI_MEDIA_TYPE", "2"))
OMNI_TTS_GPU_LAYERS = int(os.environ.get("OMNI_TTS_GPU_LAYERS", "99"))
OMNI_TOKEN2WAV_DEVICE = os.environ.get("OMNI_TOKEN2WAV_DEVICE", "gpu:0")
TTS_WAIT_SECONDS = float(os.environ.get("TTS_WAIT_SECONDS", "20"))
MAX_UTTERANCE_SECONDS = float(os.environ.get("MAX_UTTERANCE_SECONDS", "8"))
VIDEO_FPS = float(os.environ.get("VIDEO_FPS", "2"))
VIDEO_MAX_WIDTH = int(os.environ.get("VIDEO_MAX_WIDTH", "640"))
VIDEO_JPEG_COLOR_FIX = os.environ.get("VIDEO_JPEG_COLOR_FIX", "swap_rb").strip().lower()
EDGE_TTS_FALLBACK = os.environ.get("EDGE_TTS_FALLBACK", "1").lower() in ("1", "true", "yes", "y")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omni-server")

os.makedirs(OUTPUT_DIR, exist_ok=True)


class WebSocketBroadcaster:
    def __init__(self, queue_maxsize: int = 200):
        self._clients: dict[web.WebSocketResponse, tuple[asyncio.Queue, asyncio.Task]] = {}
        self._queue_maxsize = max(1, int(queue_maxsize))
        self._drop_counts: dict[str, int] = {}
        self._last_drop_log = time.monotonic()

    async def register(self, ws: web.WebSocketResponse) -> None:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_maxsize)
        task = asyncio.create_task(self._sender(ws, q))
        self._clients[ws] = (q, task)

    async def unregister(self, ws: web.WebSocketResponse) -> None:
        client = self._clients.pop(ws, None)
        if not client:
            return
        _q, task = client
        task.cancel()
        with contextlib.suppress(BaseException):
            await task

    def publish(self, message) -> None:
        if not self._clients:
            return

        dead: list[web.WebSocketResponse] = []
        msg_type = message.get("type") if isinstance(message, dict) else None

        for ws, (q, task) in list(self._clients.items()):
            if ws.closed or task.done():
                dead.append(ws)
                continue
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                if msg_type in ("video_frame", "audio_level", "llm_token"):
                    self._drop_counts[msg_type] = self._drop_counts.get(msg_type, 0) + 1
                    continue
                with contextlib.suppress(Exception):
                    _ = q.get_nowait()
                with contextlib.suppress(Exception):
                    q.put_nowait(message)
                if msg_type:
                    self._drop_counts[msg_type] = self._drop_counts.get(msg_type, 0) + 1

        now = time.monotonic()
        if self._drop_counts and (now - self._last_drop_log) >= 5.0:
            dropped = dict(sorted(self._drop_counts.items(), key=lambda kv: kv[0]))
            logger.warning(f"WebSocket drops (last 5s): {dropped}")
            self._drop_counts.clear()
            self._last_drop_log = now

        for ws in dead:
            client = self._clients.pop(ws, None)
            if not client:
                continue
            _q, task = client
            task.cancel()

    async def _sender(self, ws: web.WebSocketResponse, q: asyncio.Queue) -> None:
        while True:
            msg = await q.get()
            if ws.closed:
                return
            try:
                await ws.send_json(msg)
            except Exception:
                return


broadcaster = WebSocketBroadcaster(queue_maxsize=int(os.environ.get("WS_QUEUE_MAXSIZE", "200")))


async def broadcast(message):
    broadcaster.publish(message)


class OmniClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = None
        self.lock = asyncio.Lock()
        self.next_index = 1
        self.initialized = False
        self.effective_use_tts = OMNI_USE_TTS
        self._init_backoff_until = 0.0

    async def ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def init_omni(self, *, use_tts: bool | None = None):
        await self.ensure_session()
        now = time.monotonic()
        if now < self._init_backoff_until:
            return False
        clear_output_dir(OUTPUT_DIR)
        if use_tts is None:
            use_tts = self.effective_use_tts
        payload = {
            "msg_type": OMNI_MEDIA_TYPE,
            "use_tts": bool(use_tts),
            "duplex_mode": False,
            "model_dir": MODEL_DIR,
            "output_dir": OUTPUT_DIR,
            "n_predict": 1024,
            "auto_system_prompt": OMNI_AUTO_SYSTEM_PROMPT,
            "tts_gpu_layers": OMNI_TTS_GPU_LAYERS,
            "token2wav_device": OMNI_TOKEN2WAV_DEVICE,
        }
        if REF_AUDIO_PATH and os.path.exists(REF_AUDIO_PATH):
            payload["ref_audio_path"] = REF_AUDIO_PATH
        try:
            async with self.session.post(f"{self.base_url}/v1/stream/omni_init", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Omni init failed: {text}")
                    await broadcast({"type": "log", "level": "ERROR", "content": f"Omni init failed: {text}"})
                    self.initialized = False
                    self._init_backoff_until = time.monotonic() + 3.0
                    return False
                data = await resp.json()
                self.next_index = 1
                self.initialized = True
                self.effective_use_tts = bool(use_tts)
                logger.info(f"Omni initialized: {data}")
                await broadcast({"type": "log", "level": "INFO", "content": f"Omni initialized: {data}"})
                return True
        except Exception as e:
            logger.error(f"Omni init connection error: {e}")
            await broadcast({"type": "log", "level": "ERROR", "content": f"Omni init error: {e}"})
            self.initialized = False
            self._init_backoff_until = time.monotonic() + 3.0
            return False

    def _get_round_tts_wav_dir(self, turn_index: int) -> str:
        return os.path.join(OUTPUT_DIR, f"round_{turn_index:03d}", "tts_wav")

    def _get_round_tts_done_flag(self, turn_index: int) -> str:
        return os.path.join(self._get_round_tts_wav_dir(turn_index), "generation_done.flag")

    async def _edge_tts_fallback(self, text: str, turn_index: int) -> str | None:
        if not EDGE_TTS_FALLBACK:
            return None
        script_path = os.path.join(PROJECT_ROOT, "scripts", "generate_tts.py")
        if not os.path.exists(script_path):
            return None
        tts_dir = self._get_round_tts_wav_dir(turn_index)
        os.makedirs(tts_dir, exist_ok=True)
        out_path = os.path.join(tts_dir, f"edge_tts_{turn_index:03d}.mp3")
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
                f.write(text.strip())
                text_path = f.name
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                script_path,
                text_path,
                out_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            if proc.returncode != 0:
                return None
            if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
                return out_path
            return None
        finally:
            with contextlib.suppress(Exception):
                if "text_path" in locals() and os.path.exists(text_path):
                    os.remove(text_path)

    async def _collect_tts_wavs(self, turn_index: int) -> list:
        tts_dir = self._get_round_tts_wav_dir(turn_index)
        if not os.path.isdir(tts_dir):
            return []
        merged = os.path.join(tts_dir, "tts_output_merged.wav")
        if os.path.exists(merged):
            return [merged]
        files = [os.path.join(tts_dir, f) for f in os.listdir(tts_dir) if f.startswith("wav_") and f.endswith(".wav")]
        files.sort()
        return files

    async def process_audio(self, audio_path, image_path=None):
        async with self.lock:
            # Check file size (ignore short noise < 0.5s)
            # 16000 * 2 bytes = 32000 bytes/s
            file_size = os.path.getsize(audio_path)
            if file_size < 16000: # < 0.5s
                logger.info(f"Ignoring short audio: {file_size} bytes")
                return "", []

            await self.ensure_session()
            await broadcast({"type": "state", "content": "Processing"})

            if not self.initialized:
                ok = await self.init_omni(use_tts=self.effective_use_tts)
                if not ok and self.effective_use_tts:
                    await broadcast({"type": "log", "level": "WARNING", "content": "Omni init with TTS failed, retrying without TTS..."})
                    ok = await self.init_omni(use_tts=False)
                if not ok:
                    await broadcast({"type": "state", "content": "Idle"})
                    return "", []

            turn_index = self.next_index
            prefill_payload = {
                "audio_path_prefix": audio_path,
                "index": turn_index,
                "max_slice_nums": 1
            }
            if image_path:
                prefill_payload["img_path_prefix"] = image_path

            logger.info(f"Prefilling audio: {audio_path}")
            await broadcast({"type": "log", "level": "INFO", "content": f"Prefilling audio: {os.path.basename(audio_path)}"})

            prefill_ok = False
            tried_reinit = False
            try:
                # Add timeout for prefill (30 seconds)
                timeout = aiohttp.ClientTimeout(total=30)
                async with self.session.post(f"{self.base_url}/v1/stream/prefill", json=prefill_payload, timeout=timeout) as resp:
                    if resp.status != 200:
                        error_msg = await resp.text()
                        logger.error(f"Prefill failed: {error_msg}")
                        await broadcast({"type": "log", "level": "ERROR", "content": f"Prefill failed: {error_msg}"})
                        await broadcast({"type": "state", "content": "Idle"})
                        if (resp.status == 400 and "omni context not initialized" in error_msg) and not tried_reinit:
                            tried_reinit = True
                            await broadcast({"type": "log", "level": "WARNING", "content": "Omni context missing, reinitializing..."})
                            ok = await self.init_omni()
                            if ok:
                                turn_index = self.next_index
                                prefill_payload["index"] = turn_index
                                timeout = aiohttp.ClientTimeout(total=30)
                                async with self.session.post(f"{self.base_url}/v1/stream/prefill", json=prefill_payload, timeout=timeout) as retry_resp:
                                    if retry_resp.status == 200:
                                        prefill_ok = True
                                    else:
                                        retry_msg = await retry_resp.text()
                                        logger.error(f"Prefill retry failed: {retry_msg}")
                                        await broadcast({"type": "log", "level": "ERROR", "content": f"Prefill retry failed: {retry_msg}"})
                                        return "", []
                            else:
                                return "", []
                        if "Position encoding buffer overflow" in error_msg:
                            await broadcast({"type": "log", "level": "WARNING", "content": "Context overflow, reinitializing omni session..."})
                            ok = await self.init_omni()
                            if ok:
                                turn_index = self.next_index
                                prefill_payload["index"] = turn_index
                                timeout = aiohttp.ClientTimeout(total=30)
                                async with self.session.post(f"{self.base_url}/v1/stream/prefill", json=prefill_payload, timeout=timeout) as retry_resp:
                                    if retry_resp.status == 200:
                                        prefill_ok = True
                                    else:
                                        retry_msg = await retry_resp.text()
                                        logger.error(f"Prefill retry failed: {retry_msg}")
                                        await broadcast({"type": "log", "level": "ERROR", "content": f"Prefill retry failed: {retry_msg}"})
                                        return "", []
                            else:
                                return "", []
                        else:
                            return "", []
                    else:
                        prefill_ok = True
            except asyncio.TimeoutError:
                logger.error("Prefill timed out (server stuck?)")
                await broadcast({"type": "log", "level": "ERROR", "content": "Prefill timed out"})
                return "", []
            except Exception as e:
                logger.error(f"Prefill connection error: {e}")
                await broadcast({"type": "log", "level": "ERROR", "content": f"Prefill error: {e}"})
                await broadcast({"type": "state", "content": "Idle"})
                return "", []
            finally:
                if prefill_ok:
                    self.next_index = max(self.next_index, turn_index + 1)

            decode_payload = {
                "stream": True,
                "round_idx": turn_index
            }

            logger.info("Starting decode...")
            await broadcast({"type": "log", "level": "INFO", "content": "Starting decode (generating response)..."})
            await broadcast({"type": "state", "content": "Generating"})
            await broadcast({"type": "llm_start"})

            generated_text = ""
            token_buf: list[str] = []
            last_token_flush = time.monotonic()

            try:
                async with self.session.post(f"{self.base_url}/v1/stream/decode", json=decode_payload) as resp:
                    async for line in resp.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if "content" in data:
                                    content = data["content"]
                                    generated_text += content
                                    token_buf.append(content)
                                    now = time.monotonic()
                                    if now - last_token_flush >= 0.05:
                                        await broadcast({"type": "llm_token", "content": "".join(token_buf)})
                                        token_buf.clear()
                                        last_token_flush = now
                            except Exception:
                                pass
            except Exception as e:
                logger.error(f"Decode connection error: {e}")
                await broadcast({"type": "log", "level": "ERROR", "content": f"Decode error: {e}"})
                await broadcast({"type": "state", "content": "Idle"})
                return "", []
            if token_buf:
                await broadcast({"type": "llm_token", "content": "".join(token_buf)})
                token_buf.clear()

            await broadcast({"type": "llm_end"})
            logger.info(f"Generated text: {generated_text}")
            await broadcast({"type": "log", "level": "LLM", "content": f"Full response: {generated_text}"})

            audio_files = []
            if self.effective_use_tts:
                deadline = time.time() + TTS_WAIT_SECONDS
                last_listing = ()
                while time.time() < deadline:
                    listing = tuple(await self._collect_tts_wavs(turn_index))
                    done_flag = os.path.exists(self._get_round_tts_done_flag(turn_index))

                    if done_flag and listing:
                        await asyncio.sleep(0.5)
                        listing = tuple(await self._collect_tts_wavs(turn_index))
                        break

                    last_listing = listing
                    await asyncio.sleep(0.2)

                audio_files = list(last_listing)
                await broadcast({"type": "log", "level": "AUDIO", "content": f"Generated {len(audio_files)} audio response files"})
                if not audio_files:
                    await broadcast({"type": "log", "level": "WARNING", "content": f"No TTS wavs found under {self._get_round_tts_wav_dir(turn_index)}"})
            if not audio_files and generated_text.strip():
                edge_audio = await self._edge_tts_fallback(generated_text, turn_index)
                if edge_audio:
                    audio_files = [edge_audio]
                    await broadcast({"type": "log", "level": "AUDIO", "content": f"Edge-TTS fallback audio: {os.path.basename(edge_audio)}"})

            await broadcast({"type": "state", "content": "Idle"})
            return generated_text, audio_files


omni_client = OmniClient(LLAMA_SERVER_URL)
pcs = set()
latest_video_jpeg_bytes = None
latest_video_frame_ts = 0.0

# --- VAD Audio Track Wrapper ---


class VADAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, track, on_speech_end):
        super().__init__()
        self.track = track
        self.on_speech_end = on_speech_end
        self.vad = webrtcvad.Vad(2) # Mode 2: Aggressive
        self.frame_duration_ms = 20
        self.sample_rate = 16000
        self.buffer = bytearray()
        self.silence_frames = 0
        self.speech_frames = 0
        self.is_speaking = False
        self.silence_threshold = 40 # Reduced to 0.8s
        self.speech_start_threshold = 10 # Increased to 200ms
        self.energy_threshold = 300 # RMS threshold (0-32768)

        self.audio_data = bytearray()
        self.recording = True
        self.resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        self._last_audio_level_emit = 0.0
        self._audio_level_interval = float(os.environ.get("AUDIO_LEVEL_INTERVAL", "0.1"))

    async def recv(self):
        try:
            frame = await self.track.recv()
        except Exception:
            self.stop()
            raise

        # Calculate RMS for visualization
        try:
            # frame.to_ndarray() usually returns (channels, samples)
            # We assume mono or stereo, just take mean of squares
            # Convert to float to avoid overflow
            arr = frame.to_ndarray().astype(np.float32)
            rms = np.sqrt(np.mean(arr**2))
            # Normalize to 0-1 roughly (assuming 16bit input)
            # Max amplitude is 32768
            normalized_rms = min(rms / 10000.0, 1.0) # Adjust sensitivity
            now = time.monotonic()
            if now - self._last_audio_level_emit >= self._audio_level_interval:
                broadcaster.publish({"type": "audio_level", "content": float(normalized_rms)})
                self._last_audio_level_emit = now
        except Exception:
            pass

        # Process frame for VAD
        # Resample to 16k mono s16 for VAD
        resampled_frames = self.resampler.resample(frame)
        for f in resampled_frames:
            # Fix: AudioPlane object has no attribute 'to_bytes'
            # Use to_bytes() method on the plane if available, or just use the buffer directly
            # In PyAV < 10.0.0, planes[0] might be bytes-like or have to_bytes()
            # In newer PyAV, we should use f.to_ndarray() or bytes(f.planes[0])

            # Let's try safely converting to bytes
            raw_bytes = bytes(f.planes[0])
            self.audio_data.extend(raw_bytes)

            # VAD works on 10, 20, or 30ms frames.
            # 16000Hz * 0.02s * 2bytes = 640 bytes
            chunk_size = 640

            # Process chunks
            while len(self.buffer) + len(raw_bytes) >= chunk_size:
                needed = chunk_size - len(self.buffer)
                self.buffer.extend(raw_bytes[:needed])
                raw_bytes = raw_bytes[needed:]

                # Calculate RMS energy of this chunk
                chunk_arr = np.frombuffer(self.buffer, dtype=np.int16).astype(np.float32)
                rms = np.sqrt(np.mean(chunk_arr**2))

                # Combine VAD with Energy Threshold
                is_speech_vad = self.vad.is_speech(self.buffer, self.sample_rate)
                is_speech = is_speech_vad and (rms > self.energy_threshold)

                self.buffer = bytearray()

                if is_speech:
                    self.speech_frames += 1
                    self.silence_frames = 0
                    if self.speech_frames > self.speech_start_threshold:
                        if not self.is_speaking:
                            self.is_speaking = True
                            logger.info("Speech detected started")
                            asyncio.create_task(broadcast({"type": "vad", "status": "speaking"}))
                else:
                    self.silence_frames += 1
                    self.speech_frames = 0
                    if self.is_speaking and self.silence_frames > self.silence_threshold:
                        self.is_speaking = False
                        logger.info("Speech detected ended (Silence)")
                        asyncio.create_task(broadcast({"type": "vad", "status": "silence"}))
                        self.recording = False
                        asyncio.create_task(self.on_speech_end(bytes(self.audio_data)))
                        # Reset for next turn if needed, or stop track?
                        # Here we trigger processing.
                        self.audio_data = bytearray() # Clear buffer

        return frame


class FileAudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, files):
        super().__init__()
        self.files = files
        self.file_queue = list(files)
        self.container = None
        self.stream = None
        self.generator = None
        self.resampler = av.AudioResampler(format="s16", layout="mono", rate=48000)
        self._need_resampler_flush = False
        self._pcm_buffer = bytearray()
        self._time_base = Fraction(1, 48000)
        self._pts = 0
        self._last_audio_level_emit = 0.0
        self._audio_level_interval = float(os.environ.get("AUDIO_LEVEL_INTERVAL", "0.1"))
        self._next_file()

    def _next_file(self):
        if self.container:
            self.container.close()
            self.container = None
        self.generator = None
        if not self.file_queue:
            self.stop()
            return
        path = self.file_queue.pop(0)
        logger.info(f"Playing: {path}")
        asyncio.create_task(broadcast({"type": "log", "level": "AUDIO", "content": f"Streaming response: {os.path.basename(path)}"}))
        try:
            self.container = av.open(path)
            self.stream = self.container.streams.audio[0]
            self.generator = self.container.decode(self.stream)
            self.resampler = av.AudioResampler(format="s16", layout="mono", rate=48000)
            self._need_resampler_flush = True
        except Exception as e:
            logger.error(f"Error opening {path}: {e}")
            self._next_file()

    def _decode_one(self) -> tuple[Optional[bytes], Optional[float]]:
        try:
            in_frame = next(self.generator)
        except StopIteration:
            if self._need_resampler_flush:
                self._need_resampler_flush = False
                out_frames = self.resampler.resample(None)
                pcm_parts: list[bytes] = []
                if out_frames:
                    for of in out_frames:
                        pcm_parts.append(bytes(of.planes[0]))
                if pcm_parts:
                    return (b"".join(pcm_parts), None)
            return (None, None)

        out_frames = self.resampler.resample(in_frame)
        pcm_parts: list[bytes] = []
        if out_frames:
            for of in out_frames:
                pcm_parts.append(bytes(of.planes[0]))
        level = None
        try:
            arr = in_frame.to_ndarray().astype(np.float32)
            rms = np.sqrt(np.mean(arr**2))
            level = float(min(rms / 10000.0, 1.0))
        except Exception:
            level = None
        return (b"".join(pcm_parts), level)

    async def recv(self):
        while True:
            frame_size_samples = 960
            frame_size_bytes = frame_size_samples * 2
            if len(self._pcm_buffer) >= frame_size_bytes:
                pcm = bytes(self._pcm_buffer[:frame_size_bytes])
                del self._pcm_buffer[:frame_size_bytes]

                out = av.AudioFrame(format="s16", layout="mono", samples=frame_size_samples)
                out.sample_rate = 48000
                out.planes[0].update(pcm)
                out.pts = self._pts
                out.time_base = self._time_base
                self._pts += frame_size_samples
                return out

            if not self.container and not self.file_queue:
                if len(self._pcm_buffer) > 0:
                    pcm = bytes(self._pcm_buffer)
                    self._pcm_buffer.clear()
                    if len(pcm) < frame_size_bytes:
                        pcm += b"\x00" * (frame_size_bytes - len(pcm))
                    out = av.AudioFrame(format="s16", layout="mono", samples=frame_size_samples)
                    out.sample_rate = 48000
                    out.planes[0].update(pcm)
                    out.pts = self._pts
                    out.time_base = self._time_base
                    self._pts += frame_size_samples
                    return out
                self.stop()
                raise Exception("End of stream")

            try:
                if not self.generator:
                    self._next_file()
                    if not self.generator:
                        self.stop()
                        raise Exception("End of stream")
                pcm_chunk, level = self._decode_one()
                if pcm_chunk is None:
                    self._next_file()
                    continue
                if pcm_chunk:
                    self._pcm_buffer.extend(pcm_chunk)
                if level is not None:
                    now = time.monotonic()
                    if now - self._last_audio_level_emit >= self._audio_level_interval:
                        broadcaster.publish({"type": "audio_level", "content": level})
                        self._last_audio_level_emit = now
            except Exception:
                self.stop()
                raise

    def stop(self):
        try:
            if self.container:
                self.container.close()
        except Exception:
            pass
        self.container = None
        self.stream = None
        self.generator = None
        super().stop()


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    await broadcast({"type": "log", "level": "INFO", "content": "New WebRTC connection request"})

    response_queue = asyncio.Queue(maxsize=2)
    playback_busy = False
    pending_pcm_data = None

    last_video_emit = 0.0
    ws_enable_video = os.environ.get("WS_VIDEO_ENABLE", "1").lower() in ("1", "true", "yes", "y")

    jpeg_codec = None
    jpeg_w = 0
    jpeg_h = 0
    encode_lock = asyncio.Lock()

    def encode_jpeg(frame) -> bytes:
        nonlocal jpeg_codec, jpeg_w, jpeg_h
        if frame.width > VIDEO_MAX_WIDTH and VIDEO_MAX_WIDTH > 0:
            new_h = max(1, int(frame.height * VIDEO_MAX_WIDTH / frame.width))
            frame = frame.reformat(width=VIDEO_MAX_WIDTH, height=new_h)
        if VIDEO_JPEG_COLOR_FIX and VIDEO_JPEG_COLOR_FIX not in ("0", "false", "no", "off", "none", "disable", "disabled"):
            rgb = frame.to_ndarray(format="rgb24")
            if VIDEO_JPEG_COLOR_FIX in ("swap_rb", "rgb_swap_rb", "swap_r_b", "rb_swap", "bgr_fix"):
                rgb = rgb[..., ::-1]
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame = frame.reformat(format="yuvj420p")
        if jpeg_codec is None or jpeg_w != frame.width or jpeg_h != frame.height:
            jpeg_codec = av.CodecContext.create("mjpeg", "w")
            jpeg_codec.width = frame.width
            jpeg_codec.height = frame.height
            jpeg_codec.pix_fmt = "yuvj420p"
            jpeg_w = frame.width
            jpeg_h = frame.height
        packets = jpeg_codec.encode(frame)
        if not packets:
            packets = jpeg_codec.encode(None)
        return b"".join(bytes(p) for p in packets)

    async def consume_video(track):
        nonlocal last_video_emit
        global latest_video_jpeg_bytes, latest_video_frame_ts
        min_interval = 1.0 / max(VIDEO_FPS, 0.1)
        while True:
            try:
                frame = await track.recv()
            except Exception:
                break
            now = time.time()
            if now - last_video_emit < min_interval:
                continue
            last_video_emit = now
            try:
                async with encode_lock:
                    jpg = await asyncio.to_thread(encode_jpeg, frame)
                latest_video_jpeg_bytes = jpg
                latest_video_frame_ts = now
                if ws_enable_video:
                    b64 = await asyncio.to_thread(lambda: base64.b64encode(jpg).decode("ascii"))
                    broadcaster.publish({"type": "video_frame", "content": "data:image/jpeg;base64," + b64})
            except Exception:
                pass

    async def _do_process_speech_data(pcm_data):
        logger.info(f"Processing speech data: {len(pcm_data)} bytes")
        max_bytes = int(16000 * 2 * max(MAX_UTTERANCE_SECONDS, 0))
        if max_bytes > 0 and len(pcm_data) > max_bytes:
            pcm_data = pcm_data[-max_bytes:]
        # Save raw pcm to temp file
        raw_filename = f"raw_{uuid.uuid4()}.wav"
        clean_filename = f"clean_{uuid.uuid4()}.wav"
        image_filename = ""

        try:
            # 1. Write raw data
            with wave.open(raw_filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm_data)

            # 2. Use ffmpeg to sanitize/convert to standard 16k mono wav
            # This ensures headers are perfect for llama-server
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", raw_filename,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                clean_filename,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await proc.wait()

            if not os.path.exists(clean_filename):
                logger.error("FFmpeg failed to convert audio")
                return

            audio_path = os.path.abspath(clean_filename)
            await broadcast({"type": "log", "level": "INFO", "content": "VAD triggered processing..."})

            image_path = None
            if OMNI_MEDIA_TYPE == 2 and latest_video_jpeg_bytes:
                image_filename = f"frame_{uuid.uuid4()}.jpg"
                image_path = os.path.abspath(image_filename)
                with open(image_path, "wb") as f:
                    f.write(latest_video_jpeg_bytes)
            text, audio_files = await omni_client.process_audio(audio_path, image_path=image_path)
            try:
                response_queue.put_nowait(audio_files)
            except asyncio.QueueFull:
                with contextlib.suppress(Exception):
                    _ = response_queue.get_nowait()
                with contextlib.suppress(Exception):
                    response_queue.put_nowait(audio_files)

        except Exception as e:
            logger.error(f"Proc error: {e}")
            with contextlib.suppress(Exception):
                response_queue.put_nowait([])
        finally:
            if os.path.exists(raw_filename):
                os.remove(raw_filename)
            if os.path.exists(clean_filename):
                os.remove(clean_filename)
            if image_filename and os.path.exists(image_filename):
                os.remove(image_filename)

    async def process_speech_data(pcm_data):
        nonlocal playback_busy, pending_pcm_data
        if playback_busy:
            pending_pcm_data = pcm_data
            await broadcast({"type": "log", "level": "INFO", "content": "Playback in progress, buffered user speech"})
            return
        await _do_process_speech_data(pcm_data)

    class BlockingResponseTrack(MediaStreamTrack):
        kind = "audio"

        def __init__(self):
            super().__init__()
            self.current_track = None
            self._time_base = Fraction(1, 48000)
            self._pts = 0
            self._silence_frame_size_samples = 960
            self._started_at: float | None = None

        async def _pace(self, pts: int) -> None:
            if self._started_at is None:
                self._started_at = time.monotonic() - (pts / 48000.0)
            target = self._started_at + (pts / 48000.0)
            now = time.monotonic()
            delay = target - now
            if delay > 0:
                await asyncio.sleep(delay)

        async def recv(self):
            nonlocal playback_busy, pending_pcm_data
            if self.current_track:
                try:
                    frame = await self.current_track.recv()
                    pts = self._pts
                    frame.pts = pts
                    frame.time_base = self._time_base
                    await self._pace(pts)
                    self._pts += int(getattr(frame, "samples", self._silence_frame_size_samples))
                    return frame
                except Exception:
                    self.current_track = None
                    playback_busy = False
                    if pending_pcm_data:
                        pcm = pending_pcm_data
                        pending_pcm_data = None
                        asyncio.create_task(_do_process_speech_data(pcm))

            try:
                files = response_queue.get_nowait()
            except asyncio.QueueEmpty:
                out = av.AudioFrame(format="s16", layout="mono", samples=self._silence_frame_size_samples)
                out.sample_rate = 48000
                out.planes[0].update(b"\x00" * (self._silence_frame_size_samples * 2))
                pts = self._pts
                out.pts = pts
                out.time_base = self._time_base
                await self._pace(pts)
                self._pts += self._silence_frame_size_samples
                return out

            if not files:
                out = av.AudioFrame(format="s16", layout="mono", samples=self._silence_frame_size_samples)
                out.sample_rate = 48000
                out.planes[0].update(b"\x00" * (self._silence_frame_size_samples * 2))
                pts = self._pts
                out.pts = pts
                out.time_base = self._time_base
                await self._pace(pts)
                self._pts += self._silence_frame_size_samples
                return out

            logger.info(f"Playing response files: {files}")
            playback_busy = True
            self.current_track = FileAudioStreamTrack(files)
            frame = await self.current_track.recv()
            pts = self._pts
            frame.pts = pts
            frame.time_base = self._time_base
            await self._pace(pts)
            self._pts += int(getattr(frame, "samples", self._silence_frame_size_samples))
            return frame

    sender_track = BlockingResponseTrack()
    pc.addTrack(sender_track)

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track received: {track.kind}")
        if track.kind == "audio":
            # Wrap track with VAD
            vad_track = VADAudioTrack(track, process_speech_data)
            # We must consume the track for VAD to work
            # We can use a blackhole to pull frames
            recorder = MediaBlackhole()
            recorder.addTrack(vad_track)
            asyncio.create_task(recorder.start())
        elif track.kind == "video":
            asyncio.create_task(consume_video(track))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    )


async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    if omni_client.session:
        await omni_client.session.close()


async def safe_init_omni():
    try:
        async with omni_client.lock:
            ok = await omni_client.init_omni(use_tts=omni_client.effective_use_tts)
            if not ok and omni_client.effective_use_tts:
                await broadcast({"type": "log", "level": "WARNING", "content": "Background omni_init with TTS failed, retrying without TTS..."})
                await omni_client.init_omni(use_tts=False)
    except Exception as e:
        logger.error(f"Background init failed: {e}")


async def init_server(app):
    # Run init in background so we don't block server startup
    asyncio.create_task(safe_init_omni())


async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    await broadcaster.register(ws)
    try:
        try:
            async for msg in ws:
                pass
        except asyncio.CancelledError:
            pass
    finally:
        with contextlib.suppress(BaseException):
            await broadcaster.unregister(ws)
    return ws


async def index_handler(request):
    return web.FileResponse(os.path.join(os.path.dirname(__file__), 'static', 'index.html'))

if __name__ == "__main__":
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.on_startup.append(init_server)
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_post("/offer", offer)
    web.run_app(app, port=8081)
