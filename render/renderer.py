import moderngl
import numpy as np
from pathlib import Path
from PIL import Image
from config.defaults import (
    CIRCLE_RADIUS_RATIO, BAR_WIDTH, PALETTES, SENSITIVITY,
    HALO_SINE_R_BASE, HALO_SINE_AMPLITUDE, HALO_SINE_N_POINTS,
    HALO_SINE_GLOW_LAYERS, HALO_SINE_SMOOTHING_DECAY, HALO_SINE_FILL_OPACITY,
    BG_PULSE_INTENSITY, FLASH_INTENSITY,
)
from render.modes import HaloSineMode


SHADER_DIR = Path(__file__).parent / "shaders"
_DEFAULT_PALETTE = list(PALETTES.values())[0]

_BASS_HIST_N  = 64    # history frames
_BASS_HIST_W  = 256   # fixed width = u_bars max
_BASS2_DECAY  = 0.93  # EMA decay for Halo Bass 2 (bidirectional smooth)


def _load_shader(name: str) -> str:
    return (SHADER_DIR / name).read_text()


class Renderer:
    def __init__(self, width: int, height: int, ctx: moderngl.Context = None):
        self.width = width
        self.height = height
        self._bg_texture: moderngl.Texture | None = None
        self._center_texture: moderngl.Texture | None = None
        self._bass_hist_tex: moderngl.Texture | None = None
        self._bass_history: np.ndarray | None = None
        self._bass_hist2_tex: moderngl.Texture | None = None
        self._bass_history2: np.ndarray | None = None
        self._bass_smooth2: np.ndarray = np.zeros(_BASS_HIST_W, dtype=np.float32)

        if ctx is None:
            self.ctx = moderngl.create_standalone_context()
            self._owns_ctx = True
        else:
            self.ctx = ctx
            self._owns_ctx = False

        self._halo_sine = HaloSineMode()
        self._build_program()
        self._build_quad()
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 3)]
        )
        self._build_bass_history()
        self._build_bass_history2()

    def _build_program(self):
        self.prog = self.ctx.program(
            vertex_shader=_load_shader("radial.vert"),
            fragment_shader=_load_shader("radial.frag"),
        )

    def _build_bass_history(self):
        self._bass_history = np.zeros((_BASS_HIST_N, _BASS_HIST_W), dtype=np.float32)
        self._bass_hist_tex = self.ctx.texture(
            (_BASS_HIST_W, _BASS_HIST_N), 1,
            self._bass_history.tobytes(), dtype='f4',
        )
        self._bass_hist_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._bass_hist_tex.repeat_x = False
        self._bass_hist_tex.repeat_y = False

    def _build_bass_history2(self):
        self._bass_smooth2 = np.zeros(_BASS_HIST_W, dtype=np.float32)
        self._bass_history2 = np.zeros((_BASS_HIST_N, _BASS_HIST_W), dtype=np.float32)
        self._bass_hist2_tex = self.ctx.texture(
            (_BASS_HIST_W, _BASS_HIST_N), 1,
            self._bass_history2.tobytes(), dtype='f4',
        )
        self._bass_hist2_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._bass_hist2_tex.repeat_x = False
        self._bass_hist2_tex.repeat_y = False

    def _build_quad(self):
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype="f4")
        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, "in_position")

    def load_background(self, path: str) -> None:
        if self._bg_texture:
            self._bg_texture.release()
        img = Image.open(path).convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)
        self._bg_texture = self.ctx.texture(img.size, 3, img.tobytes())
        self._bg_texture.build_mipmaps()
        self._bg_texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

    def clear_background(self) -> None:
        if self._bg_texture:
            self._bg_texture.release()
            self._bg_texture = None

    def load_center_image(self, path: str) -> None:
        if self._center_texture:
            self._center_texture.release()
        img = Image.open(path).convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
        self._center_texture = self.ctx.texture(img.size, 4, img.tobytes())
        self._center_texture.build_mipmaps()
        self._center_texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

    def clear_center_image(self) -> None:
        if self._center_texture:
            self._center_texture.release()
            self._center_texture = None

    def render_frame(self, bars: np.ndarray, pulse: float,
                     num_bars: int = 128,
                     max_bar_height: float = 1.0,
                     pulse_intensity: float = 1.0,
                     palette: list | None = None,
                     sensitivity: float = SENSITIVITY,
                     viz_type: int = 0,
                     time: float = 0.0,
                     rotation: float = 0.0,
                     halo_r_base: float = HALO_SINE_R_BASE,
                     halo_amplitude: float = HALO_SINE_AMPLITUDE,
                     halo_n_points: int = HALO_SINE_N_POINTS,
                     halo_glow_layers: int = HALO_SINE_GLOW_LAYERS,
                     halo_smoothing_decay: float = HALO_SINE_SMOOTHING_DECAY,
                     halo_fill_opacity: float = HALO_SINE_FILL_OPACITY,
                     pal_mode: int = 0,
                     bg_pulse: bool = False,
                     bg_pulse_intensity: float = BG_PULSE_INTENSITY,
                     flash: bool = False,
                     flash_intensity: float = FLASH_INTENSITY) -> None:
        if palette is None:
            palette = _DEFAULT_PALETTE

        self.fbo.use()
        self.ctx.clear(0.039, 0.039, 0.059, 1.0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        bar_data = np.zeros(256, dtype="f4")
        n = min(len(bars), 256)
        bar_data[:n] = bars[:n]

        if self._bg_texture:
            self._bg_texture.use(location=0)
            self.prog["u_bg_texture"].value = 0
            self.prog["u_has_bg"].value = 1
        else:
            self.prog["u_has_bg"].value = 0

        if self._center_texture:
            self._center_texture.use(location=1)
            self.prog["u_center_texture"].value = 1
            self.prog["u_has_center"].value = 1
        else:
            self.prog["u_has_center"].value = 0

        for i, rgb in enumerate(palette):
            self.prog[f"u_pal{i}"].value = tuple(rgb)

        self.prog["u_num_bars"].value = num_bars
        self.prog["u_bars"].value = tuple(bar_data.tolist())
        self.prog["u_pulse"].value = float(pulse)
        self.prog["u_pulse_intensity"].value = float(pulse_intensity)
        self.prog["u_max_bar_height"].value = float(max_bar_height)
        self.prog["u_circle_radius"].value = float(CIRCLE_RADIUS_RATIO)
        self.prog["u_bar_width"].value = float(BAR_WIDTH)
        self.prog["u_aspect"].value = float(self.width / self.height)
        self.prog["u_sensitivity"].value = float(sensitivity)
        self.prog["u_viz_type"].value    = int(viz_type)
        self.prog["u_rotation"].value    = float(rotation)
        self.prog["u_pal_mode"].value            = int(pal_mode)
        self.prog["u_halo_r_base"].value         = float(halo_r_base)
        self.prog["u_bg_pulse_enabled"].value    = int(bg_pulse)
        self.prog["u_bg_pulse_intensity"].value  = float(bg_pulse_intensity)
        self.prog["u_flash_enabled"].value       = int(flash)
        self.prog["u_flash_intensity"].value     = float(flash_intensity)

        # Lissajous / Halo animation uniforms
        n = min(len(bars), 256)
        bar_slice = bars[:n]
        total_energy = float(bar_slice.sum()) + 1e-6
        centroid = float(np.sum(bar_slice * np.arange(n)) / total_energy) / n
        lissa_energy = float(np.clip(bar_slice.mean() * sensitivity * 1.5 + 0.15, 0.1, 0.85))
        self.prog["u_time"].value = float(time)
        self.prog["u_lissa_a"].value = 1.0 + centroid * 2.0
        self.prog["u_lissa_b"].value = 1.0 + (1.0 - centroid) * 2.0
        self.prog["u_lissa_energy"].value = lissa_energy

        # Bass waterfall history (mode 5) — fast-attack, slow-release
        self._bass_history = np.roll(self._bass_history, 1, axis=0)
        self._bass_history[0, :n] = bars[:n]
        self._bass_hist_tex.write(self._bass_history.tobytes())
        self._bass_hist_tex.use(location=2)
        self.prog["u_bass_history"].value = 2

        # Bass history 2 (mode 7) — bidirectional EMA, decay 0.93
        self._bass_smooth2[:n] = (self._bass_smooth2[:n] * _BASS2_DECAY
                                  + bars[:n] * (1.0 - _BASS2_DECAY))
        self._bass_history2 = np.roll(self._bass_history2, 1, axis=0)
        self._bass_history2[0, :n] = self._bass_smooth2[:n]
        self._bass_hist2_tex.write(self._bass_history2.tobytes())
        self._bass_hist2_tex.use(location=3)
        self.prog["u_bass_history2"].value = 3

        self.vao.render(moderngl.TRIANGLE_STRIP)

        # Mode 6: PIL spline overlay composited back onto the FBO
        if viz_type == 6:
            raw       = self.fbo.read(components=3)
            pixels_bt = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
            pixels_tb = np.ascontiguousarray(pixels_bt[::-1])      # flip → top-to-bottom for PIL
            result_tb = self._halo_sine.draw_overlay(
                pixels_tb, bars=bars,
                r_base=halo_r_base, amplitude_max=halo_amplitude,
                n_points=halo_n_points, glow_layers=halo_glow_layers,
                smoothing_decay=halo_smoothing_decay,
                sensitivity=sensitivity, palette=palette,
                rotation=rotation,
                fill_opacity=halo_fill_opacity,
                pal_mode=pal_mode,
            )
            result_bt = np.ascontiguousarray(result_tb[::-1])      # flip back → bottom-to-top
            self.fbo.color_attachments[0].write(result_bt.tobytes())

    def read_frame(self) -> bytes:
        # OpenGL stores rows bottom-to-top; FFmpeg rawvideo expects top-to-bottom
        raw = self.fbo.read(components=3)
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        return np.ascontiguousarray(pixels[::-1]).tobytes()

    def release(self):
        if self._bg_texture:
            self._bg_texture.release()
        if self._center_texture:
            self._center_texture.release()
        if self._bass_hist_tex:
            self._bass_hist_tex.release()
        if self._bass_hist2_tex:
            self._bass_hist2_tex.release()
        self.vao.release()
        self.vbo.release()
        self.fbo.release()
        self.prog.release()
        if self._owns_ctx:
            self.ctx.release()
