"""Output window: the composition alone, fullscreen on a chosen screen (projector).

A bare QOpenGLWindow (no widget stack) with its own Renderer, redrawn on every
buffer swap with vsync on — the frame rate follows the display, not a QTimer.
Esc closes it, F or a double-click toggles fullscreen.
"""
import time
from collections import deque

import moderngl
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow

from config.defaults import NUM_BARS
from core.layer import LayerManager
from render.renderer import Renderer


class OutputWindow(QOpenGLWindow):
    closed = Signal()

    def __init__(self, lm: LayerManager, bg_path: str | None = None,
                 center_path: str | None = None):
        super().__init__()
        fmt = QSurfaceFormat(QSurfaceFormat.defaultFormat())
        fmt.setSwapInterval(1)
        self.setFormat(fmt)
        self.setTitle("Audio Visualizer — Sortie")
        self.setCursor(Qt.BlankCursor)

        self._lm = lm
        self._bg_path = bg_path
        self._center_path = center_path
        self._renderer: Renderer | None = None
        self._ctx: moderngl.Context | None = None
        self._bars = np.zeros(NUM_BARS, dtype=np.float32)
        self._pulse = 0.0
        self._audio_provider = None
        self._pending_release: list[int] = []
        self._start_time = time.perf_counter()
        self._frame_times: deque[float] = deque(maxlen=120)
        self._render_ms = 0.0

        self.frameSwapped.connect(self.update)   # continuous, paced by vsync

    # ── Feed (same API as PreviewWidget) ─────────────────────────
    def set_audio_provider(self, provider):
        self._audio_provider = provider

    def update_audio_data(self, bars: np.ndarray, pulse: float):
        self._bars = bars
        self._pulse = pulse

    def load_background(self, path: str):
        self._bg_path = path
        if self._renderer:
            self.makeCurrent()
            self._renderer.load_background(path)
            self.doneCurrent()

    def clear_background(self):
        self._bg_path = None
        if self._renderer:
            self.makeCurrent()
            self._renderer.clear_background()
            self.doneCurrent()

    def load_center_image(self, path: str):
        self._center_path = path
        if self._renderer:
            self.makeCurrent()
            self._renderer.load_center_image(path)
            self.doneCurrent()

    def clear_center_image(self):
        self._center_path = None
        if self._renderer:
            self.makeCurrent()
            self._renderer.clear_center_image()
            self.doneCurrent()

    def release_layer_state(self, layer_id: int):
        self._pending_release.append(layer_id)

    # ── Stats ────────────────────────────────────────────────────
    @property
    def fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    @property
    def render_ms(self) -> float:
        return self._render_ms

    # ── GL lifecycle ─────────────────────────────────────────────
    def _pixel_size(self) -> tuple[int, int]:
        r = self.devicePixelRatio()
        return max(1, int(self.width() * r)), max(1, int(self.height() * r))

    def initializeGL(self):
        self._ctx = moderngl.create_context()
        self._renderer = Renderer(*self._pixel_size(), ctx=self._ctx)
        if self._bg_path:
            self._renderer.load_background(self._bg_path)
        if self._center_path:
            self._renderer.load_center_image(self._center_path)

    def resizeGL(self, w: int, h: int):
        if self._renderer:
            self._renderer.resize(*self._pixel_size())

    def paintGL(self):
        if not self._renderer:
            return
        t0 = time.perf_counter()
        while self._pending_release:
            self._renderer.release_layer_state(self._pending_release.pop())
        if self._audio_provider is not None:
            self._bars, self._pulse = self._audio_provider()
        self._renderer.render_composition(self._lm, self._bars, self._pulse,
                                          time=t0 - self._start_time)
        screen_fbo = self._ctx.detect_framebuffer(self.defaultFramebufferObject())
        self._ctx.copy_framebuffer(dst=screen_fbo, src=self._renderer.fbo)
        self._render_ms = (time.perf_counter() - t0) * 1000
        self._frame_times.append(t0)

    # ── Window behaviour ─────────────────────────────────────────
    def toggle_fullscreen(self):
        if self.windowState() & Qt.WindowFullScreen:
            self.showNormal()
        else:
            self.showFullScreen()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_F:
            self.toggle_fullscreen()

    def mouseDoubleClickEvent(self, event):
        self.toggle_fullscreen()

    def closeEvent(self, event):
        if self._renderer:
            self.makeCurrent()
            self._renderer.release()
            self._renderer = None
            self.doneCurrent()
        self.closed.emit()
        super().closeEvent(event)
