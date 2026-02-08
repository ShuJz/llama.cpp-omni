import argparse
import asyncio
import base64
import contextlib
import io
import json
import logging
import os
import sys
import time
import urllib.parse
import wave
from fractions import Fraction
from typing import Optional

import aiohttp
import av
import numpy as np
import webrtcvad
from aiortc import MediaStreamTrack
from aiortc.contrib.media import MediaPlayer


logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger("rpi-client-e2e")


def log_event(event: str, **fields) -> None:
    payload = {"event": event, "ts": time.time(), **fields}
    try:
        logger.info(json.dumps(payload, ensure_ascii=False))
    except Exception:
        logger.info(f"{event} {fields}")


def sanitize_url(url: str) -> str:
    if not isinstance(url, str):
        return url
    s = url.strip()
    s = s.replace("\r", "").replace("\n", "")
    while True:
        prev = s
        s = s.strip()
        if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
            s = s[1:-1].strip()
        s = s.replace("`", "")
        s = s.strip()
        if s == prev:
            break

    if "http://" in s or "https://" in s:
        for marker in ("http://", "https://"):
            i = s.find(marker)
            if i >= 0:
                s = s[i:]
                break
        s = s.split()[0].strip()
        s = s.strip("`\"'").strip()
        if s.endswith(("`", '"', "'", ",")):
            s = s.rstrip("`\"',").strip()
    else:
        s = s.strip()
        s = s.strip("`\"'").strip()
        if s and "://" not in s and (":" in s or "." in s):
            s = "http://" + s
    return s


def v4l2_open_tip(video_device: str, resolution: str, framerate: str, err: Exception) -> str:
    errno = getattr(err, "errno", None)
    if errno == 22:
        return (
            "Tip: /dev/video* 存在但拒绝当前分辨率/帧率/格式。"
            f" 试试 --resolution 640x480 --framerate 15，或用 v4l2-ctl 查看支持格式："
            f" v4l2-ctl -d {video_device} --list-formats-ext"
        )
    if errno == 2:
        return f"Tip: 找不到 {video_device}。用 ls /dev/video* 检查设备；USB 摄像头通常是 /dev/video0。"
    return (
        "Tip: 如果你用的是树莓派 CSI 摄像头（Bookworm/libcamera），推荐安装 picamera2 并使用 "
        "--video-backend picamera2；如果要走 v4l2，需启用 legacy/v4l2 兼容栈。"
    )


def _tts_seq(name: str) -> Optional[int]:
    if not isinstance(name, str):
        return None
    if not name.startswith("wav_") or not name.endswith(".wav"):
        return None
    mid = name[len("wav_") : -len(".wav")]
    if not mid.isdigit():
        return None
    try:
        return int(mid)
    except Exception:
        return None


def _select_tts_names(tts_list_resp: dict, *, max_files: int) -> list[str]:
    files = (tts_list_resp or {}).get("files") or []
    merged = None
    wavs: list[tuple[int, str]] = []
    others: list[str] = []
    for f in files:
        if not isinstance(f, dict):
            continue
        name = f.get("name")
        if not name:
            continue
        if name == "tts_output_merged.wav":
            merged = name
            continue
        seq = _tts_seq(name)
        if seq is not None:
            wavs.append((seq, name))
        else:
            others.append(str(name))

    done = bool((tts_list_resp or {}).get("done"))
    if done and merged:
        return [merged]

    wavs.sort(key=lambda x: x[0])
    chosen = [n for _seq, n in wavs]
    if max_files > 0 and len(chosen) > max_files:
        chosen = chosen[:max_files]
    if not chosen and others:
        chosen = sorted(set(others))
        if max_files > 0 and len(chosen) > max_files:
            chosen = chosen[:max_files]
    return chosen


class PiCamera2VideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, resolution: str, framerate: str):
        super().__init__()
        self._picam2 = None
        self._frame_interval = 1.0 / max(float(framerate), 1.0)
        self._time_base = Fraction(1, 90000)
        self._start_ts = time.time()
        width, height = 640, 480
        try:
            w, h = resolution.lower().split("x")
            width, height = int(w), int(h)
        except Exception:
            pass

        try:
            from picamera2 import Picamera2
            from libcamera import Transform
        except Exception as e:
            raise RuntimeError(f"picamera2 not available: {e}")

        self._picam2 = Picamera2()
        config = self._picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            transform=Transform(),
        )
        self._picam2.configure(config)
        self._picam2.start()

    async def recv(self):
        await asyncio.sleep(self._frame_interval)
        arr = self._picam2.capture_array()
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        pts = int((time.time() - self._start_ts) / float(self._time_base))
        frame.pts = pts
        frame.time_base = self._time_base
        return frame

    def stop(self):
        try:
            if self._picam2 is not None:
                self._picam2.stop()
                self._picam2.close()
        except Exception:
            pass
        self._picam2 = None
        super().stop()


class VideoSampler:
    def __init__(self, track: MediaStreamTrack, *, fps: float, max_width: int, color_fix: str):
        self.track = track
        self.fps = max(0.1, float(fps))
        self.max_width = int(max_width)
        self.color_fix = (color_fix or "").strip().lower()
        self._stopped = asyncio.Event()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_ts = 0.0
        self._codec: Optional[av.CodecContext] = None
        self._codec_w = 0
        self._codec_h = 0

    def latest(self) -> tuple[Optional[bytes], float]:
        return self._latest_jpeg, self._latest_ts

    def stop(self) -> None:
        self._stopped.set()

    def _encode_jpeg(self, frame: av.VideoFrame) -> Optional[bytes]:
        if self.max_width > 0 and frame.width > self.max_width:
            new_h = max(1, int(frame.height * self.max_width / frame.width))
            frame = frame.reformat(width=self.max_width, height=new_h)

        if self.color_fix and self.color_fix not in ("0", "false", "no", "off", "none", "disable", "disabled"):
            rgb = frame.to_ndarray(format="rgb24")
            if self.color_fix in ("swap_rb", "rgb_swap_rb", "swap_r_b", "rb_swap", "bgr_fix"):
                rgb = rgb[..., ::-1]
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")

        frame = frame.reformat(format="yuvj420p")
        if self._codec is None or self._codec_w != frame.width or self._codec_h != frame.height:
            self._codec = av.CodecContext.create("mjpeg", "w")
            self._codec.width = frame.width
            self._codec.height = frame.height
            self._codec.pix_fmt = "yuvj420p"
            self._codec_w = frame.width
            self._codec_h = frame.height

        packets = self._codec.encode(frame)
        if not packets:
            packets = self._codec.encode(None)
        if not packets:
            return None
        return b"".join(bytes(p) for p in packets)

    async def run(self) -> None:
        last_emit = 0.0
        interval = 1.0 / self.fps
        while not self._stopped.is_set():
            try:
                frame = await self.track.recv()
            except asyncio.CancelledError:
                return
            except Exception as e:
                log_event("video_error", error=str(e))
                await asyncio.sleep(0.5)
                continue

            now = time.monotonic()
            if now - last_emit < interval:
                continue
            last_emit = now

            try:
                jpeg = self._encode_jpeg(frame)
                if jpeg:
                    self._latest_jpeg = jpeg
                    self._latest_ts = time.time()
            except Exception as e:
                log_event("video_encode_error", error=str(e))


def pcm16_to_wav_bytes(pcm_s16le: bytes, *, sample_rate: int, channels: int) -> bytes:
    bio = io.BytesIO()
    wf = wave.open(bio, "wb")
    try:
        wf.setnchannels(int(channels))
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm_s16le)
    finally:
        with contextlib.suppress(Exception):
            wf.close()
    return bio.getvalue()


class LlamaOmniClient:
    def __init__(
        self,
        base_url: str,
        *,
        use_tts: bool,
        auto_system_prompt: bool,
        media_type: int,
        tts_gpu_layers: int,
        token2wav_device: str,
        playback_device: str,
        tts_wait_seconds: float,
        request_timeout: float,
        tts_max_files: int,
    ):
        self.base_url = sanitize_url(base_url).rstrip("/")
        self.use_tts = bool(use_tts)
        self.auto_system_prompt = bool(auto_system_prompt)
        self.media_type = int(media_type)
        self.tts_gpu_layers = int(tts_gpu_layers)
        self.token2wav_device = token2wav_device
        self.playback_device = playback_device
        self.tts_wait_seconds = float(tts_wait_seconds)
        self.request_timeout = float(request_timeout)
        self.tts_max_files = int(tts_max_files)

        self.session: Optional[aiohttp.ClientSession] = None
        self.lock = asyncio.Lock()
        self.next_index = 1
        self.initialized = False
        self._init_backoff_until = 0.0

    async def _health_probe(self) -> tuple[Optional[int], str]:
        await self.ensure_session()
        timeout = aiohttp.ClientTimeout(total=3.0, connect=2.0, sock_connect=2.0, sock_read=2.0)
        try:
            async with self.session.get(f"{self.base_url}/v1/health", timeout=timeout) as resp:
                text = await resp.text()
                return (resp.status, text or "")
        except Exception as e:
            return (None, str(e))

    async def wait_for_ready(self, *, timeout_s: float = 180.0) -> bool:
        deadline = time.monotonic() + max(1.0, float(timeout_s))
        last_status: object = object()
        while time.monotonic() < deadline:
            status, body = await self._health_probe()
            if status == 200:
                log_event("health_ready", status=200)
                return True

            if status != last_status:
                if status is None:
                    log_event("health_waiting", status=None, detail=str(body)[:200])
                else:
                    log_event("health_waiting", status=int(status), body=str(body)[:200])
                last_status = status

            await asyncio.sleep(0.5 if status == 503 else 1.0)

        log_event("health_not_ready", timeout_s=int(timeout_s))
        return False

    async def ensure_session(self) -> None:
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(
                total=self.request_timeout,
                connect=min(5.0, self.request_timeout),
                sock_connect=min(5.0, self.request_timeout),
                sock_read=self.request_timeout,
            )
            self.session = aiohttp.ClientSession(timeout=timeout)

    def _parse_host_port(self) -> tuple[Optional[str], Optional[int]]:
        try:
            u = urllib.parse.urlparse(self.base_url)
        except Exception:
            return (None, None)
        host = u.hostname
        port = u.port
        if port is None:
            port = 443 if u.scheme == "https" else 80
        return (host, port)

    async def tcp_probe(self, *, timeout_s: float = 3.0) -> bool:
        host, port = self._parse_host_port()
        if not host or not port:
            log_event("tcp_probe_bad_url", base_url=self.base_url)
            return False
        t0 = time.monotonic()
        try:
            fut = asyncio.open_connection(host=host, port=port)
            reader, writer = await asyncio.wait_for(fut, timeout=timeout_s)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            log_event("tcp_probe_ok", host=host, port=port, elapsed_ms=int((time.monotonic() - t0) * 1000))
            return True
        except asyncio.TimeoutError:
            log_event("tcp_probe_timeout", host=host, port=port, elapsed_ms=int((time.monotonic() - t0) * 1000))
            return False
        except Exception as e:
            log_event("tcp_probe_error", host=host, port=port, error=str(e), error_type=type(e).__name__, error_repr=repr(e))
            return False

    async def close(self) -> None:
        if self.session:
            with contextlib.suppress(Exception):
                await self.session.close()
        self.session = None

    async def health(self) -> bool:
        await self.ensure_session()
        t0 = time.monotonic()
        try:
            async with self.session.get(f"{self.base_url}/v1/health") as resp:
                text = await resp.text()
                log_event("health", status=resp.status, elapsed_ms=int((time.monotonic() - t0) * 1000), body=text[:200])
                return resp.status == 200
        except asyncio.TimeoutError:
            log_event("health_timeout", elapsed_ms=int((time.monotonic() - t0) * 1000))
            return False
        except Exception as e:
            log_event("health_error", error=str(e), error_type=type(e).__name__, error_repr=repr(e))
            return False

    async def init_omni(self, *, use_tts: Optional[bool] = None) -> bool:
        await self.ensure_session()
        now = time.monotonic()
        if now < self._init_backoff_until:
            return False
        if use_tts is None:
            use_tts = self.use_tts
        payload = {
            "msg_type": self.media_type,
            "use_tts": bool(use_tts),
            "duplex_mode": False,
            "n_predict": 1024,
            "auto_system_prompt": self.auto_system_prompt,
            "tts_gpu_layers": self.tts_gpu_layers,
            "token2wav_device": self.token2wav_device,
        }

        t0 = time.monotonic()
        try:
            async with self.session.post(f"{self.base_url}/v1/stream/omni_init", json=payload) as resp:
                text = await resp.text()
                if resp.status != 200:
                    log_event("omni_init_failed", status=resp.status, body=text[:2000])
                    self.initialized = False
                    self._init_backoff_until = time.monotonic() + 3.0
                    return False
                try:
                    data = json.loads(text) if text else {}
                except Exception:
                    data = {"raw": text[:2000]}
                self.next_index = 1
                self.initialized = True
                log_event("omni_init_ok", elapsed_ms=int((time.monotonic() - t0) * 1000), resp=data)
                return True
        except asyncio.TimeoutError:
            log_event(
                "omni_init_timeout",
                elapsed_ms=int((time.monotonic() - t0) * 1000),
                url=f"{self.base_url}/v1/stream/omni_init",
            )
            self.initialized = False
            self._init_backoff_until = time.monotonic() + 3.0
            return False
        except Exception as e:
            log_event("omni_init_error", error=str(e), error_type=type(e).__name__, error_repr=repr(e))
            self.initialized = False
            self._init_backoff_until = time.monotonic() + 3.0
            return False

    async def _prefill_b64(self, *, wav_bytes: bytes, img_jpeg: Optional[bytes], turn_index: int) -> bool:
        await self.ensure_session()
        payload = {
            "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
            "index": int(turn_index),
            "max_slice_nums": 1,
        }
        if img_jpeg:
            payload["img_b64"] = base64.b64encode(img_jpeg).decode("ascii")
        t0 = time.monotonic()
        try:
            async with self.session.post(f"{self.base_url}/v1/stream/prefill", json=payload) as resp:
                if resp.status != 200:
                    msg = await resp.text()
                    log_event("prefill_failed", status=resp.status, body=msg[:2000])
                    if resp.status == 400 and "omni context not initialized" in msg:
                        self.initialized = False
                    if "Position encoding buffer overflow" in msg:
                        self.initialized = False
                    return False
                _ = await resp.text()
                log_event("prefill_ok", elapsed_ms=int((time.monotonic() - t0) * 1000), turn_index=turn_index)
                return True
        except asyncio.TimeoutError:
            log_event(
                "prefill_timeout",
                elapsed_ms=int((time.monotonic() - t0) * 1000),
                turn_index=turn_index,
                url=f"{self.base_url}/v1/stream/prefill",
            )
            return False
        except Exception as e:
            log_event("prefill_error", error=str(e), error_type=type(e).__name__, error_repr=repr(e))
            return False

    async def _prefill_text(self, *, text: str, img_jpeg: Optional[bytes], turn_index: int) -> bool:
        await self.ensure_session()
        payload = {
            "text": str(text),
            "index": int(turn_index),
            "max_slice_nums": 1,
        }
        if img_jpeg:
            payload["img_b64"] = base64.b64encode(img_jpeg).decode("ascii")
        t0 = time.monotonic()
        try:
            async with self.session.post(f"{self.base_url}/v1/stream/prefill", json=payload) as resp:
                if resp.status != 200:
                    msg = await resp.text()
                    log_event("prefill_failed", status=resp.status, body=msg[:2000])
                    if resp.status == 400 and "omni context not initialized" in msg:
                        self.initialized = False
                    if "Position encoding buffer overflow" in msg:
                        self.initialized = False
                    return False
                _ = await resp.text()
                log_event("prefill_ok", elapsed_ms=int((time.monotonic() - t0) * 1000), turn_index=turn_index, mode="text")
                return True
        except asyncio.TimeoutError:
            log_event(
                "prefill_timeout",
                elapsed_ms=int((time.monotonic() - t0) * 1000),
                turn_index=turn_index,
                url=f"{self.base_url}/v1/stream/prefill",
            )
            return False
        except Exception as e:
            log_event("prefill_error", error=str(e), error_type=type(e).__name__, error_repr=repr(e))
            return False

    async def _decode_stream(self, *, turn_index: int) -> str:
        await self.ensure_session()
        payload = {"stream": True, "round_idx": int(turn_index)}
        t0 = time.monotonic()
        log_event("decode_start", turn_index=turn_index)

        full = ""
        token_buf: list[str] = []
        last_flush = time.monotonic()

        try:
            async with self.session.post(f"{self.base_url}/v1/stream/decode", json=payload) as resp:
                if resp.status != 200:
                    msg = await resp.text()
                    raise RuntimeError(f"decode http {resp.status}: {msg[:400]}")
                async for line in resp.content:
                    s = line.decode("utf-8", errors="ignore").strip()
                    if not s.startswith("data: "):
                        continue
                    data_str = s[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except Exception:
                        continue
                    content = data.get("content")
                    if not content:
                        continue
                    full += content
                    token_buf.append(content)
                    now = time.monotonic()
                    if now - last_flush >= 0.2:
                        log_event("llm_token", turn_index=turn_index, content="".join(token_buf))
                        token_buf.clear()
                        last_flush = now
        except asyncio.TimeoutError:
            log_event(
                "decode_timeout",
                elapsed_ms=int((time.monotonic() - t0) * 1000),
                turn_index=turn_index,
                url=f"{self.base_url}/v1/stream/decode",
            )
            raise
        except Exception as e:
            log_event("decode_error", error=str(e), error_type=type(e).__name__, error_repr=repr(e), turn_index=turn_index)
            raise

        if token_buf:
            log_event("llm_token", turn_index=turn_index, content="".join(token_buf))
            token_buf.clear()

        log_event("decode_done", turn_index=turn_index, elapsed_ms=int((time.monotonic() - t0) * 1000), text_len=len(full))
        log_event("llm_full", turn_index=turn_index, content=full)
        return full

    async def _tts_list(self, *, turn_index: int) -> dict:
        await self.ensure_session()
        async with self.session.get(f"{self.base_url}/v1/stream/tts_list", params={"round_idx": str(turn_index)}) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"tts_list http {resp.status}: {text[:400]}")
            return json.loads(text) if text else {}

    async def _tts_file(self, *, turn_index: int, name: str) -> bytes:
        await self.ensure_session()
        async with self.session.get(
            f"{self.base_url}/v1/stream/tts_file",
            params={"round_idx": str(turn_index), "name": name},
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"tts_file http {resp.status}: {text[:400]}")
            return await resp.read()

    async def process_turn(self, *, pcm_s16le_16k: bytes, img_jpeg: Optional[bytes], dump_dir: str) -> None:
        async with self.lock:
            t_turn0 = time.monotonic()
            pcm_len = len(pcm_s16le_16k)
            dur_s = pcm_len / (16000 * 2) if pcm_len else 0.0
            log_event("turn_begin", pcm_bytes=pcm_len, dur_s=dur_s, has_img=bool(img_jpeg))
            if pcm_len < 16000:
                log_event("turn_skip_short", pcm_bytes=pcm_len)
                return

            wav_bytes = pcm16_to_wav_bytes(pcm_s16le_16k, sample_rate=16000, channels=1)

            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
                ts = int(time.time() * 1000)
                in_wav = os.path.join(dump_dir, f"utter_{ts}.wav")
                with open(in_wav, "wb") as f:
                    f.write(wav_bytes)
                if img_jpeg:
                    in_jpg = os.path.join(dump_dir, f"frame_{ts}.jpg")
                    with open(in_jpg, "wb") as f:
                        f.write(img_jpeg)
                log_event("turn_dumped", wav_path=in_wav, has_img=bool(img_jpeg))

            if not self.initialized:
                ok = await self.init_omni(use_tts=self.use_tts)
                if not ok and self.use_tts:
                    ok = await self.init_omni(use_tts=False)
                if not ok:
                    log_event("turn_abort", reason="omni_init_failed")
                    return

            turn_index = int(self.next_index)
            self.next_index = max(self.next_index, turn_index + 1)

            ok = await self._prefill_b64(wav_bytes=wav_bytes, img_jpeg=img_jpeg, turn_index=turn_index)
            if not ok:
                if not self.initialized:
                    ok2 = await self.init_omni(use_tts=self.use_tts)
                    if ok2:
                        turn_index = int(self.next_index)
                        self.next_index = max(self.next_index, turn_index + 1)
                        ok = await self._prefill_b64(wav_bytes=wav_bytes, img_jpeg=img_jpeg, turn_index=turn_index)
                if not ok:
                    log_event("turn_abort", reason="prefill_failed")
                    return

            text = await self._decode_stream(turn_index=turn_index)

            if not self.use_tts:
                log_event("turn_done", elapsed_ms=int((time.monotonic() - t_turn0) * 1000), text_len=len(text))
                return

            deadline = time.monotonic() + self.tts_wait_seconds
            last = None
            while time.monotonic() < deadline:
                try:
                    last = await self._tts_list(turn_index=turn_index)
                except Exception as e:
                    log_event("tts_list_error", error=str(e))
                    await asyncio.sleep(0.3)
                    continue
                done = bool(last.get("done"))
                files = last.get("files") or []
                if done and files:
                    await asyncio.sleep(0.5)
                    last = await self._tts_list(turn_index=turn_index)
                    break
                await asyncio.sleep(0.2)

            files = (last or {}).get("files") or []
            names = _select_tts_names(last or {}, max_files=int(self.tts_max_files))
            log_event("tts_ready", turn_index=turn_index, file_count=len(files), play_count=len(names), done=bool((last or {}).get("done")))
            if not names:
                log_event("turn_done", elapsed_ms=int((time.monotonic() - t_turn0) * 1000), text_len=len(text))
                return

            for name in names:
                t_dl0 = time.monotonic()
                data = await self._tts_file(turn_index=turn_index, name=name)
                log_event("tts_downloaded", name=name, bytes=len(data), elapsed_ms=int((time.monotonic() - t_dl0) * 1000))

                local_path = None
                if dump_dir:
                    local_path = os.path.join(dump_dir, f"tts_{turn_index:03d}_{name}")
                else:
                    local_path = os.path.join("/tmp", f"tts_{turn_index:03d}_{name}")
                with open(local_path, "wb") as f:
                    f.write(data)

                yield_path = local_path
                log_event("tts_play_start", path=yield_path, device=self.playback_device)
                await play_wav_file(yield_path, device=self.playback_device)
                log_event("tts_play_done", path=yield_path)

            log_event("turn_done", elapsed_ms=int((time.monotonic() - t_turn0) * 1000), text_len=len(text))

    async def greet(self, *, text: str, img_jpeg: Optional[bytes], dump_dir: str) -> None:
        text = (text or "").strip()
        if not text:
            return

        async with self.lock:
            t_turn0 = time.monotonic()
            log_event("greet_begin", text=text, has_img=bool(img_jpeg))

            if not self.initialized:
                ok = await self.init_omni(use_tts=self.use_tts)
                if not ok and self.use_tts:
                    ok = await self.init_omni(use_tts=False)
                if not ok:
                    log_event("greet_abort", reason="omni_init_failed")
                    return

            turn_index = int(self.next_index)
            self.next_index = max(self.next_index, turn_index + 1)

            ok = await self._prefill_text(text=text, img_jpeg=img_jpeg, turn_index=turn_index)
            if not ok:
                log_event("greet_abort", reason="prefill_failed")
                return

            _ = await self._decode_stream(turn_index=turn_index)

            if not self.use_tts:
                log_event("greet_done", elapsed_ms=int((time.monotonic() - t_turn0) * 1000))
                return

            deadline = time.monotonic() + self.tts_wait_seconds
            last = None
            while time.monotonic() < deadline:
                try:
                    last = await self._tts_list(turn_index=turn_index)
                except Exception as e:
                    log_event("tts_list_error", error=str(e))
                    await asyncio.sleep(0.3)
                    continue
                done = bool(last.get("done"))
                files = last.get("files") or []
                if done and files:
                    await asyncio.sleep(0.5)
                    last = await self._tts_list(turn_index=turn_index)
                    break
                await asyncio.sleep(0.2)

            files = (last or {}).get("files") or []
            names = _select_tts_names(last or {}, max_files=int(self.tts_max_files))
            log_event("tts_ready", turn_index=turn_index, file_count=len(files), play_count=len(names), done=bool((last or {}).get("done")))
            for name in names:
                data = await self._tts_file(turn_index=turn_index, name=name)
                local_path = os.path.join(dump_dir or "/tmp", f"tts_greet_{turn_index:03d}_{name}")
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(data)
                log_event("tts_play_start", path=local_path, device=self.playback_device)
                await play_wav_file(local_path, device=self.playback_device)
                log_event("tts_play_done", path=local_path)

            log_event("greet_done", elapsed_ms=int((time.monotonic() - t_turn0) * 1000))


async def play_wav_file(path: str, *, device: str) -> None:
    resolved = device
    if resolved.startswith("hw:"):
        resolved = "plughw:" + resolved[3:]
    args = ["aplay", "-D", resolved, path]
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        if proc.stderr:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                s = line.decode("utf-8", errors="ignore").strip()
                if s:
                    log_event("aplay", msg=s)
    finally:
        rc = await proc.wait()
        if rc != 0:
            raise RuntimeError(f"aplay exited {rc}")


class VADSegmenter:
    def __init__(
        self,
        track: MediaStreamTrack,
        *,
        on_segment,
        max_utterance_seconds: float,
        vad_mode: int,
        energy_threshold: float,
        speech_start_frames: int,
        silence_end_frames: int,
        busy_event: asyncio.Event,
        video_sampler: Optional[VideoSampler],
        dump_dir: str,
    ):
        self.track = track
        self.on_segment = on_segment
        self.max_utterance_seconds = float(max_utterance_seconds)
        self.vad = webrtcvad.Vad(int(vad_mode))
        self.energy_threshold = float(energy_threshold)
        self.speech_start_frames = int(speech_start_frames)
        self.silence_end_frames = int(silence_end_frames)
        self.busy_event = busy_event
        self.video_sampler = video_sampler
        self.dump_dir = dump_dir

        self.resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        self.buffer = bytearray()
        self.audio_data = bytearray()
        self.silence_frames = 0
        self.speech_frames = 0
        self.is_speaking = False
        self._last_level_log = 0.0
        self._utter_start_ts = 0.0

    async def run(self) -> None:
        chunk_size = 640
        while True:
            try:
                frame = await self.track.recv()
            except asyncio.CancelledError:
                return
            except Exception as e:
                log_event("audio_error", error=str(e))
                await asyncio.sleep(0.1)
                continue

            if self.busy_event.is_set():
                if self.is_speaking or self.audio_data or self.buffer:
                    self._reset()
                continue

            try:
                arr = frame.to_ndarray().astype(np.float32)
                rms = float(np.sqrt(np.mean(arr * arr))) if arr.size else 0.0
                now = time.monotonic()
                if now - self._last_level_log >= 1.0:
                    log_event("audio_level", rms=rms)
                    self._last_level_log = now
            except Exception:
                pass

            resampled_frames = self.resampler.resample(frame)
            for f in resampled_frames:
                raw = bytes(f.planes[0])
                self.audio_data.extend(raw)
                max_bytes = int((self.max_utterance_seconds + 1.0) * 16000 * 2)
                if max_bytes > 0 and len(self.audio_data) > max_bytes:
                    del self.audio_data[:-max_bytes]

                while len(self.buffer) + len(raw) >= chunk_size:
                    needed = chunk_size - len(self.buffer)
                    self.buffer.extend(raw[:needed])
                    raw = raw[needed:]

                    chunk_arr = np.frombuffer(self.buffer, dtype=np.int16).astype(np.float32)
                    rms = float(np.sqrt(np.mean(chunk_arr * chunk_arr))) if chunk_arr.size else 0.0
                    try:
                        is_speech_vad = self.vad.is_speech(self.buffer, 16000)
                    except Exception:
                        is_speech_vad = False
                    is_speech = bool(is_speech_vad and (rms > self.energy_threshold))

                    self.buffer = bytearray()

                    if is_speech:
                        self.speech_frames += 1
                        self.silence_frames = 0
                        if self.speech_frames >= self.speech_start_frames and not self.is_speaking:
                            self.is_speaking = True
                            self._utter_start_ts = time.monotonic()
                            log_event("vad_start")
                    else:
                        self.silence_frames += 1
                        self.speech_frames = 0

                    if self.is_speaking and (time.monotonic() - self._utter_start_ts) >= self.max_utterance_seconds:
                        log_event("vad_force_end", reason="max_utterance_seconds")
                        await self._finish_segment()
                        continue

                    if self.is_speaking and self.silence_frames >= self.silence_end_frames:
                        log_event("vad_end", silence_frames=self.silence_frames)
                        await self._finish_segment()
                        continue

    async def _finish_segment(self) -> None:
        if not self.audio_data:
            self._reset()
            return
        pcm = bytes(self.audio_data)
        max_bytes = int(16000 * 2 * max(self.max_utterance_seconds, 0))
        if max_bytes > 0 and len(pcm) > max_bytes:
            log_event("segment_trim", before_bytes=len(pcm), after_bytes=max_bytes)
            pcm = pcm[-max_bytes:]
        self._reset()

        img = None
        if self.video_sampler:
            img, ts = self.video_sampler.latest()
            if img:
                log_event("segment_attach_frame", age_s=max(0.0, time.time() - ts))

        await self.on_segment(pcm, img, self.dump_dir)

    def _reset(self) -> None:
        self.audio_data = bytearray()
        self.buffer = bytearray()
        self.silence_frames = 0
        self.speech_frames = 0
        self.is_speaking = False
        self._utter_start_ts = 0.0


async def main_async(args) -> int:
    raw_url = args.llama_server_url
    llama_url = sanitize_url(raw_url)
    log_event("start", llama_server_url_raw=raw_url, llama_server_url=llama_url, script=__file__)
    if not (llama_url.startswith("http://") or llama_url.startswith("https://")):
        log_event("bad_url", value=llama_url, tip="不要用反引号包 URL（shell 会当命令执行）；直接传 http://x.x.x.x:8080")
        return 2

    audio_player = None
    video_player = None
    video_track = None

    try:
        audio_player = MediaPlayer(args.audio_device, format="alsa", options={"channels": "1", "sample_rate": "16000"})
        if not audio_player.audio:
            raise RuntimeError("no audio track from MediaPlayer")
        log_event("audio_open_ok", device=args.audio_device)
    except Exception as e:
        log_event("audio_open_failed", device=args.audio_device, error=str(e))
        return 2

    if not args.no_video:
        if args.video_backend in ("auto", "v4l2"):
            try:
                options = {"video_size": args.resolution, "framerate": args.framerate}
                video_player = MediaPlayer(args.video_device, format="v4l2", options=options)
                video_track = video_player.video
                if video_track:
                    log_event("video_open_ok", backend="v4l2", device=args.video_device)
            except Exception as e:
                if args.video_backend == "v4l2":
                    log_event("video_open_failed", backend="v4l2", device=args.video_device, error=str(e), tip=v4l2_open_tip(args.video_device, args.resolution, args.framerate, e))
                    return 2
                log_event("video_open_failed", backend="v4l2", device=args.video_device, error=str(e), tip=v4l2_open_tip(args.video_device, args.resolution, args.framerate, e))

        if (args.video_backend in ("auto", "picamera2")) and (video_track is None):
            try:
                video_track = PiCamera2VideoTrack(args.resolution, args.framerate)
                log_event("video_open_ok", backend="picamera2")
            except Exception as e:
                if args.video_backend == "picamera2":
                    log_event("video_open_failed", backend="picamera2", error=str(e))
                    return 2
                log_event("video_open_failed", backend="picamera2", error=str(e))

    busy = asyncio.Event()

    client = LlamaOmniClient(
        llama_url,
        use_tts=not args.no_tts,
        auto_system_prompt=not args.no_auto_system_prompt,
        media_type=args.media_type,
        tts_gpu_layers=args.tts_gpu_layers,
        token2wav_device=args.token2wav_device,
        playback_device=args.playback_device,
        tts_wait_seconds=args.tts_wait_seconds,
        request_timeout=args.request_timeout,
        tts_max_files=args.tts_max_files,
    )

    async def on_segment(pcm: bytes, img: Optional[bytes], dump_dir: str) -> None:
        if busy.is_set():
            return
        busy.set()
        try:
            await client.process_turn(pcm_s16le_16k=pcm, img_jpeg=img, dump_dir=dump_dir)
        except Exception as e:
            log_event("turn_error", error=str(e))
        finally:
            busy.clear()

    video_sampler = None
    video_task = None
    if video_track is not None:
        video_sampler = VideoSampler(
            video_track,
            fps=args.video_fps,
            max_width=args.video_max_width,
            color_fix=args.video_jpeg_color_fix,
        )
        video_task = asyncio.create_task(video_sampler.run())

    try:
        _ = await client.tcp_probe(timeout_s=3.0)
        ready = await client.wait_for_ready(timeout_s=args.server_ready_timeout)
        if ready:
            _ = await client.health()

        ok = False
        if ready:
            for _ in range(3):
                ok = await client.init_omni()
                if ok:
                    break
                await asyncio.sleep(1.0)
        if not ok:
            log_event("init_failed", tip="检查 llama-server 是否可达（/v1/health），以及 server 端是否设置了 MODEL_DIR/OUTPUT_DIR/REF_AUDIO_PATH")
        if ok and (not args.no_auto_greet):
            busy.set()
            try:
                img = None
                if video_sampler:
                    img, _ts = video_sampler.latest()
                await client.greet(text=args.greet_text, img_jpeg=img, dump_dir=args.dump_dir)
            finally:
                busy.clear()

        vad = VADSegmenter(
            audio_player.audio,
            on_segment=on_segment,
            max_utterance_seconds=args.max_utterance_seconds,
            vad_mode=args.vad_mode,
            energy_threshold=args.energy_threshold,
            speech_start_frames=args.speech_start_frames,
            silence_end_frames=args.silence_end_frames,
            busy_event=busy,
            video_sampler=video_sampler,
            dump_dir=args.dump_dir,
        )
        audio_task = asyncio.create_task(vad.run())

        await asyncio.gather(*(t for t in [video_task, audio_task] if t is not None))
        return 0
    finally:
        if video_sampler:
            video_sampler.stop()
        if video_task:
            with contextlib.suppress(Exception):
                video_task.cancel()
        with contextlib.suppress(Exception):
            audio_task.cancel()
        with contextlib.suppress(Exception):
            await client.close()
        with contextlib.suppress(Exception):
            if video_track is not None and hasattr(video_track, "stop"):
                video_track.stop()
        with contextlib.suppress(Exception):
            if audio_player is not None:
                audio_player.stop()
        await asyncio.sleep(0.05)
        with contextlib.suppress(Exception):
            if audio_player is not None:
                audio_player.stop()
        with contextlib.suppress(Exception):
            if video_player is not None:
                video_player.stop()


def main() -> int:
    p = argparse.ArgumentParser(description="Raspberry Pi end-to-end client (mic/video -> llama-server -> TTS playback)")
    p.add_argument("--llama-server-url", default=os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080"))
    p.add_argument("--media-type", type=int, default=int(os.environ.get("OMNI_MEDIA_TYPE", "2")))
    p.add_argument("--no-tts", action="store_true")
    p.add_argument("--no-auto-system-prompt", action="store_true")
    p.add_argument("--tts-gpu-layers", type=int, default=int(os.environ.get("OMNI_TTS_GPU_LAYERS", "99")))
    p.add_argument("--token2wav-device", default=os.environ.get("OMNI_TOKEN2WAV_DEVICE", "gpu:0"))
    p.add_argument("--tts-wait-seconds", type=float, default=float(os.environ.get("TTS_WAIT_SECONDS", "20")))
    p.add_argument("--request-timeout", type=float, default=float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "60")))
    p.add_argument("--server-ready-timeout", type=float, default=float(os.environ.get("SERVER_READY_TIMEOUT_SECONDS", "180")))
    p.add_argument("--tts-max-files", type=int, default=int(os.environ.get("TTS_MAX_FILES", "8")))

    p.add_argument("--audio-device", default="default")
    p.add_argument("--playback-device", default=os.environ.get("PLAYBACK_DEVICE", "default"))
    p.add_argument("--video-device", default="/dev/video0")
    p.add_argument("--resolution", default="640x480")
    p.add_argument("--framerate", default="15")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--video-backend", choices=["auto", "v4l2", "picamera2"], default="auto")
    p.add_argument("--video-fps", type=float, default=float(os.environ.get("VIDEO_FPS", "5")))
    p.add_argument("--video-max-width", type=int, default=int(os.environ.get("VIDEO_MAX_WIDTH", "640")))
    p.add_argument("--video-jpeg-color-fix", default=os.environ.get("VIDEO_JPEG_COLOR_FIX", "swap_rb"))

    p.add_argument("--vad-mode", type=int, default=2)
    p.add_argument("--energy-threshold", type=float, default=float(os.environ.get("ENERGY_THRESHOLD", "300")))
    p.add_argument("--speech-start-frames", type=int, default=int(os.environ.get("SPEECH_START_FRAMES", "10")))
    p.add_argument("--silence-end-frames", type=int, default=int(os.environ.get("SILENCE_END_FRAMES", "40")))
    p.add_argument("--max-utterance-seconds", type=float, default=float(os.environ.get("MAX_UTTERANCE_SECONDS", "8")))

    p.add_argument("--dump-dir", default=os.environ.get("DUMP_DIR", "/tmp/omni_e2e"))
    p.add_argument("--no-auto-greet", action="store_true")
    p.add_argument("--greet-text", default=os.environ.get("OMNI_GREET_TEXT", "你好，你是谁？"))

    args = p.parse_args()
    args.llama_server_url = sanitize_url(args.llama_server_url)

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        log_event("stopped", reason="KeyboardInterrupt")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
