# Raspberry Pi WebRTC Client Setup

此目录包含了用于树莓派 4B 的 WebRTC 客户端脚本 `rpi_client.py`，用于连接 Omni 服务器进行音视频流传输。

## 1. 系统准备 (Raspberry Pi OS)

### 摄像头选择（推荐 rpicam-apps / picamera2）
在 Raspberry Pi OS Bookworm 上，官方命令行工具从 `libcamera-*` 迁移为 `rpicam-*`（底层仍是 libcamera），推荐使用 `picamera2` 后端（无需 legacy）。

```bash
sudo apt update
sudo apt install -y python3-picamera2 rpicam-apps
```

然后运行 `sudo raspi-config`，在 Interface Options 里启用 Camera，重启后建议先用 `rpicam-hello` 验证摄像头能正常工作。

### 启用 V4L2/Legacy（仅当你要走 v4l2 后端）
如果你必须通过 `/dev/video*`（V4L2）采集（例如用 USB 摄像头或 legacy 兼容模式），需要启用 Legacy Camera 支持：

1. 运行 `sudo raspi-config`
2. 选择 `3 Interface Options`
3. 选择 `I1 Legacy Camera`
4. 选择 `Yes` 并重启树莓派

### 安装系统依赖
在安装 Python 库之前，需要安装一些系统库（用于编译 PyAV 和 aiortc）：

```bash
sudo apt update
sudo apt install -y python3-pip python3-dev \
    libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config \
    libsrtp2-dev libopencv-dev ffmpeg alsa-utils
```

*注意：`alsa-utils` 提供了 `aplay` 命令，用于播放音频。*

## 2. 安装 Python 依赖

建议使用虚拟环境：

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements-pi.txt
```

## 3. 运行客户端

确保你的服务器（WSL 主机）已经启动了 WebRTC 服务 (`server.py`)。

```bash
# 替换为你的 WSL 主机 IP (注意端口是 8081)
python rpi_client.py --server http://192.168.0.104:8081/offer
```

推荐在 Bookworm 上显式使用 picamera2：

```bash
python rpi_client.py --server http://192.168.0.104:8081/offer --video-backend picamera2
```

### 参数说明
- `--server`: 信号服务器地址 (默认: `http://192.168.0.104:8081/offer`)
- `--video-device`: 视频设备路径 (默认: `/dev/video0`)
- `--audio-device`: 音频设备 ALSA 名称 (默认: `default`)
- `--resolution`: 视频分辨率 (默认: `640x480`)
- `--framerate`: 视频帧率 (默认: `30`)
- `--no-video`: 仅传输音频
- `--video-backend`: 视频后端 (`auto`/`v4l2`/`picamera2`)

## 4. 常见问题

- **报错 `No such file or directory: '/dev/video0'`**:
  请检查是否已启用 Legacy Camera 模式并重启。或者尝试使用 USB 摄像头。

- **报错 `picamera2 not available`**:
  Bookworm 上安装 `python3-picamera2` 与 `rpicam-apps`，并在 raspi-config 启用 Camera 后重启；运行时加 `--video-backend picamera2`。如果你在 venv 中运行，建议用 `python3 -m venv --system-site-packages venv` 让 venv 能 import 系统安装的 `picamera2`。

- **音频播放无声**:
  检查扬声器连接，并尝试运行 `aplay /usr/share/sounds/alsa/Front_Center.wav` 测试系统音频。

- **延迟高**:
  网络环境影响较大，尝试降低分辨率 `--resolution 320x240`。
