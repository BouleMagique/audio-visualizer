import time as _time
import numpy as np
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import QTimer
import moderngl

from render.renderer import Renderer
from config.defaults import (
    NUM_BARS, PALETTES, SENSITIVITY,
    HALO_SINE_R_BASE, HALO_SINE_AMPLITUDE, HALO_SINE_N_POINTS,
    HALO_SINE_GLOW_LAYERS, HALO_SINE_SMOOTHING_DECAY, HALO_SINE_FILL_OPACITY,
    BG_PULSE_INTENSITY, FLASH_INTENSITY,
)


class PreviewWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._renderer: Renderer | None = None
        self._bars = np.zeros(NUM_BARS, dtype=np.float32)
        self._pulse = 0.0
        self._num_bars = NUM_BARS
        self._max_bar_height = 1.0
        self._pulse_intensity = 1.0
        self._palette = list(PALETTES.values())[0]
        self._sensitivity = SENSITIVITY
        self._viz_type = 0
        self._bg_image_path: str | None = None
        self._center_image_path: str | None = None
        self._rotation: float = 0.0
        self._halo_r_base: float = HALO_SINE_R_BASE
        self._halo_amplitude: float = HALO_SINE_AMPLITUDE
        self._halo_n_points: int = HALO_SINE_N_POINTS
        self._halo_glow_layers: int = HALO_SINE_GLOW_LAYERS
        self._halo_smoothing_decay: float = HALO_SINE_SMOOTHING_DECAY
        self._halo_fill_opacity: float = HALO_SINE_FILL_OPACITY
        self._pal_mode: int = 0
        self._bg_pulse: bool = False
        self._bg_pulse_intensity: float = BG_PULSE_INTENSITY
        self._flash: bool = False
        self._flash_intensity: float = FLASH_INTENSITY
        self._start_time = _time.perf_counter()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(1000 // 60)

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
            self._renderer.release()
        self._renderer = Renderer(w, h, ctx=self._ctx)
        if self._bg_image_path:
            self._renderer.load_background(self._bg_image_path)
        if self._center_image_path:
            self._renderer.load_center_image(self._center_image_path)

    def paintGL(self):
        if not self._renderer:
            return
        self._renderer.render_frame(
            bars=self._bars,
            pulse=self._pulse,
            num_bars=self._num_bars,
            max_bar_height=self._max_bar_height,
            pulse_intensity=self._pulse_intensity,
            palette=self._palette,
            sensitivity=self._sensitivity,
            viz_type=self._viz_type,
            time=_time.perf_counter() - self._start_time,
            rotation=self._rotation,
            halo_r_base=self._halo_r_base,
            halo_amplitude=self._halo_amplitude,
            halo_n_points=self._halo_n_points,
            halo_glow_layers=self._halo_glow_layers,
            halo_smoothing_decay=self._halo_smoothing_decay,
            halo_fill_opacity=self._halo_fill_opacity,
            pal_mode=self._pal_mode,
            bg_pulse=self._bg_pulse,
            bg_pulse_intensity=self._bg_pulse_intensity,
            flash=self._flash,
            flash_intensity=self._flash_intensity,
        )
        qt_fbo = self._ctx.detect_framebuffer(self.defaultFramebufferObject())
        self._ctx.copy_framebuffer(dst=qt_fbo, src=self._renderer.fbo)

    def update_audio_data(self, bars: np.ndarray, pulse: float):
        self._bars = bars
        self._pulse = pulse
        self.update()

    def set_params(self, num_bars: int = None, max_bar_height: float = None,
                   pulse_intensity: float = None, palette: list = None,
                   sensitivity: float = None, viz_type: int = None,
                   rotation: float = None,
                   halo_r_base: float = None, halo_amplitude: float = None,
                   halo_n_points: int = None, halo_glow_layers: int = None,
                   halo_smoothing_decay: float = None,
                   halo_fill_opacity: float = None,
                   pal_mode: int = None,
                   bg_pulse: bool = None,
                   bg_pulse_intensity: float = None,
                   flash: bool = None,
                   flash_intensity: float = None):
        if num_bars is not None:
            self._num_bars = num_bars
        if max_bar_height is not None:
            self._max_bar_height = max_bar_height
        if pulse_intensity is not None:
            self._pulse_intensity = pulse_intensity
        if palette is not None:
            self._palette = palette
        if sensitivity is not None:
            self._sensitivity = sensitivity
        if viz_type is not None:
            self._viz_type = viz_type
        if rotation is not None:
            self._rotation = rotation
        if halo_r_base is not None:
            self._halo_r_base = halo_r_base
        if halo_amplitude is not None:
            self._halo_amplitude = halo_amplitude
        if halo_n_points is not None:
            self._halo_n_points = halo_n_points
        if halo_glow_layers is not None:
            self._halo_glow_layers = halo_glow_layers
        if halo_smoothing_decay is not None:
            self._halo_smoothing_decay = halo_smoothing_decay
        if halo_fill_opacity is not None:
            self._halo_fill_opacity = halo_fill_opacity
        if pal_mode is not None:
            self._pal_mode = pal_mode
        if bg_pulse is not None:
            self._bg_pulse = bg_pulse
        if bg_pulse_intensity is not None:
            self._bg_pulse_intensity = bg_pulse_intensity
        if flash is not None:
            self._flash = flash
        if flash_intensity is not None:
            self._flash_intensity = flash_intensity

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
