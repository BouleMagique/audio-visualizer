"""Halo Sine (6) / Flat Sine (9) — GPU spline path: renders, is lit, stays cheap."""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from pathlib import Path
from PIL import Image

from core.layer import LayerManager
from render.renderer import Renderer

W, H = 1280, 720


def _bars(f: int) -> np.ndarray:
    x = np.linspace(0, 1, 128)
    b = (0.8 * np.exp(-x * 4) * (0.6 + 0.4 * np.sin(f * 0.3))
         + 0.25 * np.abs(np.sin(x * 23 + f * 0.2)) * (1 - x * 0.5))
    return np.clip(b, 0, 1).astype(np.float32)


def _render(r: Renderer, mode: int, **params) -> tuple[np.ndarray, float]:
    lm = LayerManager()
    for layer in list(lm.active_layers()):
        lm.remove_layer(layer.id)
    layer = lm.add_layer(mode)
    for k, v in params.items():
        setattr(layer, k, v)
    for f in range(30):
        r.render_composition(lm, _bars(f), 0.5, time=f / 60)
    r.ctx.finish()
    t0 = time.perf_counter()
    for f in range(30, 60):
        r.render_composition(lm, _bars(f), 0.5, time=f / 60)
    r.ctx.finish()
    ms = (time.perf_counter() - t0) / 30 * 1000
    px = np.frombuffer(r.read_frame(), np.uint8).reshape(H, W, 3)
    return px, ms


def test_spline_modes():
    r = Renderer(W, H)
    out = Path(__file__).parent
    for mode, params in ((6, dict(pal_mode=1, halo_fill_opacity=0.5)),
                         (9, dict(pal_mode=1, halo_fill_opacity=0.5))):
        px, ms = _render(r, mode, **params)
        lit = float((px.max(axis=2) > 40).mean())
        assert lit > 0.01, f"mode {mode}: frame almost black ({lit:.4f} lit)"
        # Former PIL path: 11-25 ms/frame at 720p. GPU path is ~0.4 ms.
        assert ms < 8.0, f"mode {mode}: {ms:.2f} ms/frame, GPU path regressed?"
        Image.fromarray(px).save(out / f"frame_mode{mode}.png")
        print(f"[OK] mode {mode}: {ms:.2f} ms/frame, {lit:.1%} lit")
    r.release()


if __name__ == "__main__":
    test_spline_modes()
    print("\nDone.")
