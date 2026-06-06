import time as _time
import numpy as np
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import QTimer, Qt, Signal
import moderngl

from render.renderer import Renderer
from config.defaults import NUM_BARS
from core.layer import LayerManager


class PreviewWidget(QOpenGLWidget):
    """OpenGL preview driven by a LayerManager composition.

    The MainWindow owns the LayerManager and sets it via set_layer_manager().
    Dragging moves the currently selected layer.
    """

    layer_moved = Signal()   # emitted after a drag so the UI can refresh

    def __init__(self, parent=None):
        super().__init__(parent)
        self._renderer: Renderer | None = None
        self._lm: LayerManager | None = None
        self._bars = np.zeros(NUM_BARS, dtype=np.float32)
        self._pulse = 0.0
        self._bg_image_path: str | None = None
        self._center_image_path: str | None = None
        self._start_time = _time.perf_counter()

        self._dragging = False
        self._drag_origin = None        # (mouse_x, mouse_y, layer_x, layer_y)
        self._show_selection = True
        self._pending_release: list[int] = []   # layer ids to free on next paint

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(1000 // 60)

    # ── Public API ───────────────────────────────────────────────
    def set_layer_manager(self, lm: LayerManager):
        self._lm = lm

    def update_audio_data(self, bars: np.ndarray, pulse: float):
        self._bars = bars
        self._pulse = pulse
        self.update()

    # ── GL lifecycle ─────────────────────────────────────────────
    def initializeGL(self):
        self._ctx = moderngl.create_context()
        w, h = max(self.width(), 1), max(self.height(), 1)
        self._renderer = Renderer(w, h, ctx=self._ctx)
        if self._bg_image_path:
            self._renderer.load_background(self._bg_image_path)
        if self._center_image_path:
            self._renderer.load_center_image(self._center_image_path)

    def resizeGL(self, w: int, h: int):
        if not hasattr(self, "_ctx") or self._ctx is None:
            return
        if self._renderer:
            self._renderer.resize(max(w, 1), max(h, 1))

    def paintGL(self):
        if not self._renderer or not self._lm:
            return
        # Free states for removed layers now that the GL context is current
        while self._pending_release:
            self._renderer.release_layer_state(self._pending_release.pop())
        t = _time.perf_counter() - self._start_time
        self._renderer.render_composition(self._lm, self._bars, self._pulse, time=t)

        sel = self._lm.selected()
        if sel is not None and self._show_selection:
            self._renderer.draw_selection_outline(sel)

        qt_fbo = self._ctx.detect_framebuffer(self.defaultFramebufferObject())
        self._ctx.copy_framebuffer(dst=qt_fbo, src=self._renderer.fbo)

    # ── Image loading (global bg + center) ───────────────────────
    def load_background(self, path: str):
        self._bg_image_path = path
        if self._renderer:
            self._renderer.load_background(path)
        self.update()

    def clear_background(self):
        self._bg_image_path = None
        if self._renderer:
            self._renderer.clear_background()
        self.update()

    def load_center_image(self, path: str):
        self._center_image_path = path
        if self._renderer:
            self._renderer.load_center_image(path)
        self.update()

    def clear_center_image(self):
        self._center_image_path = None
        if self._renderer:
            self._renderer.clear_center_image()
        self.update()

    def release_layer_state(self, layer_id: int):
        self._pending_release.append(layer_id)

    # ── Mouse drag → move selected layer ─────────────────────────
    def mousePressEvent(self, event):
        if not self._lm or self._lm.selected() is None:
            return
        if event.button() == Qt.LeftButton:
            layer = self._lm.selected()
            self._dragging = True
            self._drag_origin = (event.position().x(), event.position().y(),
                                 layer.x, layer.y)

    def mouseMoveEvent(self, event):
        if not self._dragging or not self._lm:
            return
        layer = self._lm.selected()
        if layer is None:
            return
        mx0, my0, lx0, ly0 = self._drag_origin
        dx = event.position().x() - mx0
        dy = event.position().y() - my0
        # Pixel delta → NDC (y inverted)
        layer.x = lx0 + 2.0 * dx / max(self.width(), 1)
        layer.y = ly0 - 2.0 * dy / max(self.height(), 1)
        self.update()

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            self.layer_moved.emit()
