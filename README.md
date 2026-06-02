# Audio Visualizer

Python desktop audio visualizer with GPU-accelerated rendering (ModernGL / GLSL) and FFmpeg video export.

## Requirements

- Python 3.11+
- FFmpeg in PATH — [ffmpeg.org](https://ffmpeg.org/download.html)
- GPU with OpenGL 3.3+ support (any modern GPU)

## Install

```bash
git clone https://github.com/BouleMagique/audio-visualizer
cd audio-visualizer

python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Visualization modes

| Mode | Description |
|------|-------------|
| Radial | Circular bars radiating outward |
| Mirror | Symmetric vertical bars |
| Linear | Bottom-up bars |
| Oscilloscope | Lissajous figure |
| Halo | Smooth waveform ring |
| Halo Bass | Polar bass waterfall (12 rings) |
| Halo Sine | Catmull-Rom spline, PIL-rendered |
| Halo Bass 2 | EMA-smoothed neon waterfall |

## Features

- 5 built-in palettes + 2 custom palette modes (amplitude / frequency mapping)
- Background image with bass-reactive zoom pulse
- Center image overlay with glow ring and pulse
- White flash on kick / bass hits
- Halo Sine: interior fill, per-point smoothing, auto-gain, rotation
- Export to MP4 via FFmpeg — 720p / 1080p / 4K, 30 or 60 fps, H.264 + AAC 320k

## Stack

| Component | Library |
|-----------|---------|
| GPU render | ModernGL 5.x + GLSL 3.30 |
| Audio decode | soundfile |
| Audio playback | sounddevice |
| FFT / DSP | numpy + scipy |
| UI | PySide6 |
| PIL overlay | Pillow |
| Export | FFmpeg (subprocess) |
