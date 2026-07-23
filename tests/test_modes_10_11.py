"""Render-check for the rewritten Nuclear Shockwave (10) and Void Pull (11).

Spawns a layer per mode, drives it with synthetic bars + kicks across frames,
and dumps a PNG per mode so the visuals can be eyeballed. Verifies the shader
compiles and the u_shock_times pipeline runs end to end.
"""
import sys, os, struct, zlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from pathlib import Path
from render.renderer import Renderer
from core.layer import LayerManager


def save_png(raw: bytes, w: int, h: int, path: Path):
    def chunk(name, data):
        c = zlib.crc32(name + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", c)
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)[::-1]
    rows = b"".join(b"\x00" + row.tobytes() for row in pixels)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(rows)))
        f.write(chunk(b"IEND", b""))


def run_mode(mode: int, w=480, h=270, frames=40):
    r = Renderer(w, h)
    lm = LayerManager()
    lm.add_layer(mode=mode)
    rng = np.random.default_rng(mode)
    # Sustained-bass techno: 4-on-the-floor kick every 24 frames (~0.4 s = 150 BPM)
    # with a noisy groove (bassline/hats) between kicks that must NOT spawn waves.
    last = None
    saved = None
    st = r._state_for(lm.layers[0].id)   # inspect spawn count
    before = int((st.shock_times > -50).sum())
    spawn_log = []
    for i in range(frames):
        t = i / 60.0
        bars = rng.uniform(0.0, 0.2, 128).astype(np.float32)
        bars[:40] += 0.55                      # sustained heavy bass
        phase = i % 24
        # pulse: kick spike on phase 0 (fast attack, slow decay) + groove wiggle
        groove = 0.15 + 0.06 * np.sin(i * 1.3) + 0.04 * rng.uniform(0, 1)
        if phase == 0:
            bars[:30] += 0.35
            pulse = 0.9
        else:
            pulse = max(groove, 0.9 * (0.80 ** phase))
        n_before = int(st.shock_ptr)
        r.render_composition(lm, bars, float(pulse), time=t)
        if st.shock_ptr != n_before:
            spawn_log.append(i)
        if i == frames - 1 - 6:
            saved = r.read_frame()
        last = r.read_frame()
    n_kicks = sum(1 for i in range(frames) if i % 24 == 0)
    print(f"   mode {mode}: {len(spawn_log)} waves spawned for {n_kicks} kicks "
          f"(frames {spawn_log})")
    out = Path(__file__).parent / f"frame_mode{mode}.png"
    save_png(saved if saved is not None else last, w, h, out)
    r.release()
    print(f"[OK] mode {mode} rendered {frames} frames -> {out}")


if __name__ == "__main__":
    run_mode(10)
    run_mode(11)
    print("\nDone.")
