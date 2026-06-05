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
    HALO_SINE_SPLINE_GAP, HALO_SINE_PIXEL_SIZE,
    TUNNEL_SIDES, TUNNEL_RINGS, TUNNEL_SPEED, TUNNEL_KICK_ZOOM, TUNNEL_CHROMA,
    TUNNEL_KICK_SENSITIVITY, TUNNEL_BASS_SPEED, TUNNEL_KICK_MODE,
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
        self._halo_spline_gap: float = HALO_SINE_SPLINE_GAP
        self._halo_pixel_size: int = HALO_SINE_PIXEL_SIZE
        self._tunnel_sides: int = TUNNEL_SIDES
        self._tunnel_rings: int = TUNNEL_RINGS
        self._tunnel_speed: float = TUNNEL_SPEED
        self._tunnel_kick_zoom: float = TUNNEL_KICK_ZOOM
        self._tunnel_chroma: float = TUNNEL_CHROMA
        self._tunnel_kick_sensitivity: float = TUNNEL_KICK_SENSITIVITY
        self._tunnel_bass_speed: float = TUNNEL_BASS_SPEED
        self._tunnel_kick_mode: int = TUNNEL_KICK_MODE
        self._mirror: bool = False
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
        prev_bass    = self._renderer._prev_bass    if self._renderer else 0.0
        kick_accum   = self._renderer._kick_accum   if self._renderer else 0.0
        kick_buf     = self._renderer._kick_bass_buf if self._renderer else None
        kick_cd      = self._renderer._kick_cooldown if self._renderer else 0
        if self._renderer:
            self._renderer.release()
        self._renderer = Renderer(w, h, ctx=self._ctx)
        self._renderer._prev_bass    = prev_bass
        self._renderer._kick_accum   = kick_accum
        self._renderer._kick_bass_buf = kick_buf
        self._renderer._kick_cooldown = kick_cd
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
            halo_spline_gap=self._halo_spline_gap,
            halo_pixel_size=self._halo_pixel_size,
            tunnel_sides=self._tunnel_sides,
            tunnel_rings=self._tunnel_rings,
            tunnel_speed=self._tunnel_speed,
            tunnel_kick_zoom=self._tunnel_kick_zoom,
            tunnel_chroma=self._tunnel_chroma,
            tunnel_kick_sensitivity=self._tunnel_kick_sensitivity,
            tunnel_bass_speed=self._tunnel_bass_speed,
            tunnel_kick_mode=self._tunnel_kick_mode,
            mirror=self._mirror,
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
                   halo_spline_gap: float = None,
                   halo_pixel_size: int = None,
                   tunnel_sides: int = None,
                   tunnel_rings: int = None,
                   tunnel_speed: float = None,
                   tunnel_kick_zoom: float = None,
                   tunnel_chroma: float = None,
                   tunnel_kick_sensitivity: float = None,
                   tunnel_bass_speed: float = None,
                   tunnel_kick_mode: int = None,
                   mirror: bool = None,
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
        if halo_spline_gap is not None:
            self._halo_spline_gap = halo_spline_gap
        if halo_pixel_size is not None:
            self._halo_pixel_size = halo_pixel_size
        if tunnel_sides is not None:
            self._tunnel_sides = tunnel_sides
        if tunnel_rings is not None:
            self._tunnel_rings = tunnel_rings
        if tunnel_speed is not None:
            self._tunnel_speed = tunnel_speed
        if tunnel_kick_zoom is not None:
            self._tunnel_kick_zoom = tunnel_kick_zoom
        if tunnel_chroma is not None:
            self._tunnel_chroma = tunnel_chroma
        if tunnel_kick_sensitivity is not None:
            self._tunnel_kick_sensitivity = tunnel_kick_sensitivity
        if tunnel_bass_speed is not None:
            self._tunnel_bass_speed = tunnel_bass_speed
        if tunnel_kick_mode is not None:
            self._tunnel_kick_mode = tunnel_kick_mode
        if mirror is not None:
            self._mirror = mirror
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
