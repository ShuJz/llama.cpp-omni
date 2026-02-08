import argparse
import asyncio
import logging
import subprocess
import threading
import time
import aiohttp
import wave
import queue
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaBlackhole
import av
import numpy as np
from fractions import Fraction

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rpi-client")


def set_sdp_video_bandwidth(sdp: str, max_bitrate_bps: int) -> str:
    if not sdp or max_bitrate_bps <= 0:
        return sdp
    sep = "\r\n" if "\r\n" in sdp else "\n"
    raw_lines = [ln for ln in sdp.split(sep) if ln != ""]
    if not raw_lines:
        return sdp

    session_lines: list[str] = []
    media_sections: list[list[str]] = []
    cur: list[str] = []

    for ln in raw_lines:
        if ln.startswith("m="):
            if not media_sections and not cur:
                cur = [ln]
                media_sections.append(cur)
                continue
            if cur and (cur[0].startswith("m=")):
                cur = [ln]
                media_sections.append(cur)
            else:
                cur = [ln]
                media_sections.append(cur)
        else:
            if not media_sections:
                session_lines.append(ln)
            else:
                media_sections[-1].append(ln)

    kbps = max(1, int(max_bitrate_bps // 1000))

    for sec in media_sections:
        if not sec or not sec[0].startswith("m=video"):
            continue
        sec[:] = [ln for ln in sec if not (ln.startswith("b=AS:") or ln.startswith("b=TIAS:"))]
        insert_at = 1
        for i, ln in enumerate(sec):
            if ln.startswith("c="):
                insert_at = i + 1
                break
        sec[insert_at:insert_at] = [f"b=AS:{kbps}", f"b=TIAS:{int(max_bitrate_bps)}"]

    merged: list[str] = []
    merged.extend(session_lines)
    for sec in media_sections:
        merged.extend(sec)
    return "\r\n".join(merged) + "\r\n"


def sanitize_server_url(url: str) -> str:
    if not isinstance(url, str):
        return url
    s = url.strip()
    while True:
        prev = s
        s = s.strip()
        if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
            s = s[1:-1].strip()
        s = s.strip("`").strip()
        if s == prev:
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
        return (
            f"Tip: 找不到 {video_device}。用 ls /dev/video* 检查设备；USB 摄像头通常是 /dev/video0。"
        )
    return (
        "Tip: 如果你用的是树莓派 CSI 摄像头（Bookworm/libcamera），推荐安装 picamera2 并使用 "
        "--video-backend picamera2；如果要走 v4l2，需启用 legacy/v4l2 兼容栈。"
    )

# Audio Playback Handling


class AudioPlayerTrack(MediaStreamTrack):
    """
    A MediaStreamTrack that consumes received audio frames and plays them
    via a subprocess (aplay for Linux/Pi).
    """
    kind = "audio"

    def __init__(
        self,
        track,
        playback_device: str = "default",
        playback_gain: float = 1.0,
        playback_channels: int = 2,
        dump_wav_path: str = "",
    ):
        super().__init__()
        self.track = track
        self.playback_device = playback_device
        self.playback_device_resolved = playback_device
        self.playback_gain = float(playback_gain)
        self.playback_channels = 1 if int(playback_channels) == 1 else 2
        self.force_mono_mix = True
        self.process = None
        self._last_level_log = 0.0
        self.dump_wav_path = dump_wav_path
        self._wav = None
        self._stop_event = threading.Event()
        self._pcm_queue: "queue.Queue[bytes | None]" = queue.Queue(maxsize=2000)
        self._writer_thread: threading.Thread | None = None
        self._start_player()

    def _start_player(self):
        # Start aplay subprocess to play raw audio from stdin
        # Assuming 48kHz stereo s16le which is standard for WebRTC usually
        # But we need to check what PyAV gives us.
        # Typically aiortc resamples to 48kHz stereo or mono.
        # We will inspect the first frame to configure aplay.
        self.process = None

    async def recv(self):
        try:
            frame = await self.track.recv()
        except Exception as e:
            self.stop()
            raise e

        if self.process is None:
            channels = len(frame.layout.channels)
            rate = frame.sample_rate
            format_str = frame.format.name

            logger.info(f"Received Audio Format: {format_str}, Rate: {rate}, Channels: {channels}")

            self.resampler = av.AudioResampler(format="s16", layout="mono" if self.playback_channels == 1 else "stereo", rate=48000)

            if self.playback_device.startswith("hw:"):
                self.playback_device_resolved = "plughw:" + self.playback_device[3:]
            else:
                self.playback_device_resolved = self.playback_device

            self.process = subprocess.Popen(
                [
                    "aplay",
                    "-D", self.playback_device_resolved,
                    "-t", "raw",
                    "-r", "48000",
                    "-c", str(self.playback_channels),
                    "-f", "S16_LE",
                    "--buffer-time", "500000",
                    "--period-time", "20000",
                    "-",
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            threading.Thread(target=self._drain_stderr, daemon=True).start()

            if self.dump_wav_path:
                try:
                    self._wav = wave.open(self.dump_wav_path, "wb")
                    self._wav.setnchannels(self.playback_channels)
                    self._wav.setsampwidth(2)
                    self._wav.setframerate(48000)
                    logger.info(f"Dumping received audio to wav: {self.dump_wav_path}")
                except Exception as e:
                    logger.error(f"Failed to open dump wav: {e}")
                    self._wav = None

            self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._writer_thread.start()

        frames = self.resampler.resample(frame)
        for f in frames:
            try:
                arr = f.to_ndarray()
                if arr.ndim == 2:
                    if arr.shape[0] in (1, 2) and arr.shape[1] > 2:
                        arr = arr.T
                arr = np.ascontiguousarray(arr)
                if arr.dtype != np.int16:
                    arr = arr.astype(np.int16, copy=False)

                if self.force_mono_mix and self.playback_channels == 2 and arr.ndim == 2 and arr.shape[1] == 2:
                    m = ((arr[:, 0].astype(np.int32) + arr[:, 1].astype(np.int32)) // 2).astype(np.int16)
                    arr[:, 0] = m
                    arr[:, 1] = m

                if self.playback_gain != 1.0:
                    a = arr.astype(np.float32, copy=False) * self.playback_gain
                    a = np.clip(a, -32768, 32767)
                    arr = a.astype(np.int16)

                pcm = arr.tobytes()

                try:
                    try:
                        self._pcm_queue.put_nowait(pcm)
                    except queue.Full:
                        try:
                            _ = self._pcm_queue.get_nowait()
                        except Exception:
                            pass
                        try:
                            self._pcm_queue.put_nowait(pcm)
                        except Exception:
                            pass
                except queue.Full:
                    pass

                now = time.time()
                if now - self._last_level_log > 1.0:
                    a = arr.astype(np.float32, copy=False)
                    rms = float(np.sqrt(np.mean(a * a))) if a.size else 0.0
                    peak = int(np.max(np.abs(arr))) if arr.size else 0
                    logger.info(
                        f"Playback RMS: {rms:.1f}, Peak: {peak} (device={self.playback_device_resolved}, gain={self.playback_gain})"
                    )
                    self._last_level_log = now
            except Exception:
                self.stop()
                raise

        return frame

    def _drain_stderr(self):
        try:
            if not self.process or not self.process.stderr:
                return
            for line in iter(self.process.stderr.readline, b""):
                s = line.decode("utf-8", errors="ignore").strip()
                if s:
                    logger.warning(f"aplay: {s}")
        except Exception:
            return

    def stop(self):
        self._stop_event.set()
        try:
            self._pcm_queue.put_nowait(None)
        except Exception:
            pass
        if self._writer_thread is not None and self._writer_thread.is_alive():
            try:
                self._writer_thread.join(timeout=1.0)
            except Exception:
                pass
        self._writer_thread = None
        if self.process:
            self.process.terminate()
            self.process = None
        if self._wav is not None:
            try:
                self._wav.close()
            except Exception:
                pass
            self._wav = None
        super().stop()

    def _writer_loop(self):
        try:
            while not self._stop_event.is_set():
                pcm = self._pcm_queue.get()
                if pcm is None:
                    break
                if self.process and self.process.stdin:
                    try:
                        self.process.stdin.write(pcm)
                    except Exception:
                        break
                if self._wav is not None:
                    try:
                        self._wav.writeframes(pcm)
                    except Exception:
                        pass
        finally:
            try:
                if self.process and self.process.stdin:
                    self.process.stdin.flush()
            except Exception:
                pass


class PiCamera2VideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, resolution: str, framerate: str):
        super().__init__()
        self._picam2 = None
        self._frame_interval = 1.0 / max(float(framerate), 1.0)
        self._time_base = Fraction(1, 90000)
        self._start_ts = time.time()
        self._seq = 0
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
        self._seq += 1
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


async def run(pc, player, recorder, server_url, playback_device: str, playback_gain: float, playback_channels: int, dump_wav_path: str):
    # Add local tracks
    if player and player.audio:
        pc.addTrack(player.audio)
    if player and player.video:
        pc.addTrack(player.video)

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track received: {track.kind}")
        if track.kind == "audio":
            # Hook up the player
            logger.info("Starting Audio Playback...")
            local_player = AudioPlayerTrack(
                track,
                playback_device=playback_device,
                playback_gain=playback_gain,
                playback_channels=playback_channels,
                dump_wav_path=dump_wav_path,
            )

            async def pump():
                while True:
                    try:
                        await local_player.recv()
                    except Exception:
                        break

            asyncio.create_task(pump())

            @track.on("ended")
            async def on_ended():
                logger.info("Track ended")
                local_player.stop()

    # Create Offer
    for sender in pc.getSenders():
        try:
            track = getattr(sender, "track", None)
            if not track or getattr(track, "kind", None) != "video":
                continue
            if not hasattr(sender, "getParameters") or not hasattr(sender, "setParameters"):
                continue
            params = sender.getParameters()
            if params is None:
                continue
            encodings = getattr(params, "encodings", None)
            if not encodings:
                continue
            enc = encodings[0]
            if isinstance(enc, dict):
                if getattr(run, "_video_max_bitrate", 0) > 0:
                    enc["maxBitrate"] = int(getattr(run, "_video_max_bitrate"))
                if getattr(run, "_video_max_fps", 0) > 0:
                    enc["maxFramerate"] = int(getattr(run, "_video_max_fps"))
                if float(getattr(run, "_video_scale", 1.0)) > 1.0:
                    enc["scaleResolutionDownBy"] = float(getattr(run, "_video_scale"))
            else:
                if getattr(run, "_video_max_bitrate", 0) > 0:
                    setattr(enc, "maxBitrate", int(getattr(run, "_video_max_bitrate")))
                if getattr(run, "_video_max_fps", 0) > 0:
                    setattr(enc, "maxFramerate", int(getattr(run, "_video_max_fps")))
                if float(getattr(run, "_video_scale", 1.0)) > 1.0:
                    setattr(enc, "scaleResolutionDownBy", float(getattr(run, "_video_scale")))
            maybe = sender.setParameters(params)
            if asyncio.iscoroutine(maybe):
                await maybe
            logger.info(
                f"Configured video sender: maxBitrate={getattr(run, '_video_max_bitrate', 0)}, "
                f"maxFramerate={getattr(run, '_video_max_fps', 0)}, "
                f"scaleResolutionDownBy={getattr(run, '_video_scale', 1.0)}"
            )
        except Exception as e:
            logger.warning(f"Failed to configure video sender parameters: {e}")

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    if int(getattr(run, "_video_max_bitrate", 0)) > 0:
        try:
            munged = set_sdp_video_bandwidth(pc.localDescription.sdp, int(getattr(run, "_video_max_bitrate")))
            if munged and munged != pc.localDescription.sdp:
                await pc.setLocalDescription(RTCSessionDescription(sdp=munged, type=pc.localDescription.type))
                logger.info(f"Applied SDP video bandwidth: {int(getattr(run, '_video_max_bitrate'))} bps")
        except Exception as e:
            logger.warning(f"Failed to apply SDP bandwidth: {e}")

    # Send Offer
    logger.info(f"Connecting to {server_url}...")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(server_url, json={
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            }) as resp:
                if resp.status != 200:
                    logger.error(f"Server Error: {await resp.text()}")
                    return
                answer = await resp.json()
        except Exception as e:
            logger.error(f"Connection Failed: {e}")
            return

    await pc.setRemoteDescription(RTCSessionDescription(
        sdp=answer["sdp"],
        type=answer["type"]
    ))

    logger.info("Connected! Press Ctrl+C to stop.")

    # Keep alive
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry Pi WebRTC Client")
    parser.add_argument("--server", default="http://192.168.0.104:8081/offer", help="Signaling Server URL")
    parser.add_argument("--video-device", default="/dev/video0", help="Video device (e.g. /dev/video0)")
    parser.add_argument("--audio-device", default="default", help="Audio device (ALSA name)")
    parser.add_argument("--playback-device", default="default", help="Playback device for aplay (e.g. hw:3,0)")
    parser.add_argument("--playback-gain", type=float, default=4.0, help="Software gain for received audio")
    parser.add_argument("--playback-channels", type=int, choices=[1, 2], default=1, help="Playback channel count sent to aplay")
    parser.add_argument("--dump-wav-path", default="/tmp/webrtc_rx.wav", help="If set, dump received audio PCM to this wav path")
    parser.add_argument("--resolution", default="640x480", help="Video resolution")
    parser.add_argument("--framerate", default="15", help="Video framerate")
    parser.add_argument("--no-video", action="store_true", help="Disable video")
    parser.add_argument("--video-backend", choices=["auto", "v4l2", "picamera2"], default="auto", help="Video capture backend")
    parser.add_argument("--video-max-bitrate", type=int, default=350_000, help="Max video bitrate (bps) for WebRTC sender")
    parser.add_argument("--video-max-fps", type=int, default=15, help="Max video framerate for WebRTC sender")
    parser.add_argument("--video-scale", type=float, default=1.0, help="Downscale factor (>1.0 to reduce resolution)")

    args = parser.parse_args()
    args.server = sanitize_server_url(args.server)

    # Setup Media Player (Input)
    options = {}
    if not args.no_video:
        options["video_size"] = args.resolution
        options["framerate"] = args.framerate

    # On Pi, usually we use V4L2 for video and ALSA for audio
    # Format: v4l2 for video, alsa for audio
    # aiortc MediaPlayer doesn't support opening multiple formats in one go easily if they are different devices?
    # Actually it does if we use ffmpeg input string properly or create two MediaPlayers.

    # Let's create separate players for Audio and Video to be safe and combine tracks.

    pc = RTCPeerConnection()

    # Audio Input (Mic)
    try:
        audio_input = MediaPlayer(args.audio_device, format="alsa", options={"channels": "1", "sample_rate": "16000"})
        pc.addTrack(audio_input.audio)
        logger.info(f"Added Audio Input: {args.audio_device}")
    except Exception as e:
        logger.error(f"Failed to open audio: {e}")

    # Video Input (Cam)
    if not args.no_video:
        if args.video_backend in ("auto", "v4l2"):
            try:
                video_input = MediaPlayer(args.video_device, format="v4l2", options=options)
                pc.addTrack(video_input.video)
                logger.info(f"Added Video Input (v4l2): {args.video_device}")
                video_input = None
            except Exception as e:
                if args.video_backend == "v4l2":
                    logger.error(f"Failed to open video (v4l2): {e}")
                else:
                    logger.warning(f"Failed to open video (v4l2): {e}")
                    logger.info("Falling back to picamera2 backend...")
                logger.warning(v4l2_open_tip(args.video_device, args.resolution, args.framerate, e))
        if args.video_backend in ("auto", "picamera2") and not args.no_video:
            try:
                picam_track = PiCamera2VideoTrack(args.resolution, args.framerate)
                pc.addTrack(picam_track)
                logger.info("Added Video Input (picamera2)")
            except Exception as e:
                logger.error(f"Failed to open video (picamera2): {e}")
                logger.warning(
                    "Tip: 在 Raspberry Pi OS Bookworm 上运行："
                    " sudo apt update && sudo apt install -y python3-picamera2 rpicam-apps，"
                    "并在 raspi-config 启用 Camera 后重启；再用 rpicam-hello 验证；然后用 --video-backend picamera2。"
                )

    # We need a 'recorder' to consume the received track.
    # We use MediaBlackhole to consume video (if any) and our custom player for audio.
    recorder = MediaBlackhole() # Use Blackhole instead of writing to /dev/null

    loop = asyncio.get_event_loop()
    try:
        setattr(run, "_video_max_bitrate", int(args.video_max_bitrate))
        setattr(run, "_video_max_fps", int(args.video_max_fps))
        setattr(run, "_video_scale", float(args.video_scale))
        loop.run_until_complete(run(pc, None, recorder, args.server, args.playback_device, args.playback_gain, args.playback_channels, args.dump_wav_path))
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(pc.close())
