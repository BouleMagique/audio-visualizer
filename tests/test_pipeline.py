"""Quick smoke test: generate synthetic audio, run FFT, dump one PNG frame."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path

from core.fft import FFTProcessor
from render.renderer import Renderer


def make_sine_wav(path: str, freq: float = 440.0, sr: int = 44100, dur: float = 2.0):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False).astype(np.float32)
    wave = (0.5 * np.sin(2 * np.pi * freq * t)).reshape(-1, 1)
    sf.write(path, wave, sr)


def test_fft_pipeline():
    sr = 44100
    fft_size = 2048
    num_bars = 128
    proc = FFTProcessor(sr=sr, fft_size=fft_size, num_bars=num_bars)

    rng = np.random.default_rng(42)
    samples = rng.uniform(-0.5, 0.5, fft_size).astype(np.float32)
    bars, pulse = proc.process(samples)

    assert bars.shape == (num_bars,), f"Expected {num_bars} bars, got {bars.shape}"
    assert 0.0 <= pulse <= 1.0, f"Pulse out of range: {pulse}"
    assert bars.min() >= 0.0 and bars.max() <= 1.0, "Bars out of [0,1]"
    print(f"[OK] FFT pipeline — bars max={bars.max():.3f} pulse={pulse:.3f}")


def test_render_frame():
    width, height = 320, 180
    renderer = Renderer(width, height)

    bars = np.random.default_rng(0).uniform(0, 1, 128).astype(np.float32)
    pulse = 0.5
    renderer.render_frame(bars=bars, pulse=pulse)
    raw = renderer.read_frame()

    assert len(raw) == width * height * 3, "Frame size mismatch"
    # Save PNG for visual check
    out = Path(__file__).parent / "frame_test.png"
    import struct, zlib

    def png_chunk(name, data):
        c = zlib.crc32(name + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", c)

    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    pixels = pixels[::-1]  # flip Y (OpenGL origin is bottom-left)

    rows = []
    for row in pixels:
        rows.append(b"\x00" + row.tobytes())
    compressed = zlib.compress(b"".join(rows))

    with open(out, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        f.write(png_chunk(b"IDAT", compressed))
        f.write(png_chunk(b"IEND", b""))

    renderer.release()
    print(f"[OK] Render frame — PNG saved to {out}")


if __name__ == "__main__":
    test_fft_pipeline()
    test_render_frame()
    print("\nAll tests passed.")
