import math
import random
import moderngl
import numpy as np
from collections import deque
from pathlib import Path
from PIL import Image
from config.defaults import (
    CIRCLE_RADIUS_RATIO, BAR_WIDTH, PALETTES, SENSITIVITY,
)
from core.layer import Layer, BGLayer, LayerManager, BLEND_ADDITIVE, apply_auto_lfo
from render.modes import HaloSineMode, FlatSineMode


SHADER_DIR = Path(__file__).parent / "shaders"
_DEFAULT_PALETTE = list(PALETTES.values())[0]

_BASS_HIST_N  = 64    # history frames
_BASS_HIST_W  = 256   # history width (modes 5/7 read the low third of the bars)
_BARS_MAX     = 512   # u_bars_tex width = max bars per layer
_BASS2_DECAY  = 0.93  # EMA decay for Halo Bass 2 (bidirectional smooth)
_SPLINE_MAX   = 256   # max control points for modes 6/9 (UI caps halo_n_points at 256)


def _load_shader(name: str) -> str:
    return (SHADER_DIR / name).read_text()


class LayerState:
    """Per-layer mutable render state: audio history, spline analysis, FBO.

    History textures are canvas-size-independent (256×64) so they survive a
    canvas resize; only the layer FBO is recreated.
    """

    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.prev_bass: float = 0.0
        self.kick_accum: float = 0.0
        self.kick_bass_buf: deque | None = None
        self.kick_cooldown: int = 0
        self.kick_bg_ema: float = 0.0

        # Mode 10 shockwaves: ring buffer of spawn times (seconds) + adaptive
        # bass-onset detector (independent of the Tunnel kick settings, so it
        # fires reliably on sustained techno).
        self.shock_times = np.full(8, -100.0, dtype="f4")
        self.shock_ptr: int = 0
        self.shock_avg: float = 0.0       # slow-tracking background pulse level
        self.shock_armed: bool = True     # hysteresis gate — one wave per kick
        self.shock_cooldown: int = 0

        # Alien Eye v2 (mode 14): CPU saccade (gaze) + blink state machines.
        self.eye_seeded = False
        self.eye_last_t = 0.0
        self.eye_gx = 0.0
        self.eye_gy = 0.0
        self.eye_tx = 0.0
        self.eye_ty = 0.0
        self.eye_next_saccade = 0.0
        self.eye_blink_amt = 0.0
        self.eye_blink_state = 0     # 0 idle, 1 closing, 2 opening
        self.eye_blink_clock = 0.0
        self.eye_next_blink = 0.0
        self.eye_double_queued = False

        self.bass_smooth2 = np.zeros(_BASS_HIST_W, dtype=np.float32)
        self.bass_history  = np.zeros((_BASS_HIST_N, _BASS_HIST_W), dtype=np.float32)
        self.bass_history2 = np.zeros((_BASS_HIST_N, _BASS_HIST_W), dtype=np.float32)

        self.bass_hist_tex = ctx.texture(
            (_BASS_HIST_W, _BASS_HIST_N), 1, self.bass_history.tobytes(), dtype='f4')
        self.bass_hist_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.bass_hist_tex.repeat_x = False
        self.bass_hist_tex.repeat_y = False

        self.bass_hist2_tex = ctx.texture(
            (_BASS_HIST_W, _BASS_HIST_N), 1, self.bass_history2.tobytes(), dtype='f4')
        self.bass_hist2_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.bass_hist2_tex.repeat_x = False
        self.bass_hist2_tex.repeat_y = False

        self.halo_sine = HaloSineMode()
        self.flat_sine = FlatSineMode()
        # Spline control values for modes 6/9 (u_spline_tex), read with texelFetch
        self.spline_tex = ctx.texture((_SPLINE_MAX, 1), 1, dtype='f4')

        self.fbo: moderngl.Framebuffer | None = None
        self.resize(width, height)

    def resize(self, width: int, height: int) -> None:
        if self.fbo:
            self.fbo.color_attachments[0].release()
            self.fbo.release()
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 3)])

    def release(self) -> None:
        self.bass_hist_tex.release()
        self.bass_hist2_tex.release()
        self.spline_tex.release()
        if self.fbo:
            self.fbo.color_attachments[0].release()
            self.fbo.release()
            self.fbo = None


class Renderer:
    def __init__(self, width: int, height: int, ctx: moderngl.Context = None):
        self.width = width
        self.height = height
        self._bg_texture: moderngl.Texture | None = None
        self._bg_img_aspect: float = 1.0
        self._center_texture: moderngl.Texture | None = None

        if ctx is None:
            self.ctx = moderngl.create_standalone_context()
            self._owns_ctx = True
        else:
            self.ctx = ctx
            self._owns_ctx = False

        self._states: dict[int, LayerState] = {}

        self._build_programs()
        self._build_quad()
        self._bars_tex = self.ctx.texture((_BARS_MAX, 1), 1, dtype="f4")
        self.fbo: moderngl.Framebuffer | None = None
        self.resize(width, height)

    # ── Setup ────────────────────────────────────────────────────
    def _build_programs(self):
        self.prog = self.ctx.program(
            vertex_shader=_load_shader("radial.vert"),
            fragment_shader=_load_shader("radial.frag"),
        )
        self.composite_prog = self.ctx.program(
            vertex_shader=_load_shader("composite.vert"),
            fragment_shader=_load_shader("composite.frag"),
        )
        self.bg_prog = self.ctx.program(
            vertex_shader=_load_shader("radial.vert"),
            fragment_shader=_load_shader("bg.frag"),
        )
        self.line_prog = self.ctx.program(
            vertex_shader=_load_shader("composite.vert"),
            fragment_shader=(
                "#version 330 core\n"
                "out vec4 fragColor;\n"
                "uniform vec3 u_color;\n"
                "void main() { fragColor = vec4(u_color, 1.0); }\n"
            ),
        )

    def _build_quad(self):
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype="f4")
        self.vbo = self.ctx.buffer(vertices)
        self.vao           = self.ctx.simple_vertex_array(self.prog, self.vbo, "in_position")
        self.composite_vao = self.ctx.simple_vertex_array(self.composite_prog, self.vbo, "in_position")
        self.bg_vao        = self.ctx.simple_vertex_array(self.bg_prog, self.vbo, "in_position")

        # Line-loop quad (CCW order) for the selection outline
        line_verts = np.array([
            -1.0, -1.0,  1.0, -1.0,  1.0, 1.0,  -1.0, 1.0,
        ], dtype="f4")
        self.line_vbo = self.ctx.buffer(line_verts)
        self.line_vao = self.ctx.simple_vertex_array(self.line_prog, self.line_vbo, "in_position")

    def resize(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        if self.fbo:
            self.fbo.color_attachments[0].release()
            self.fbo.release()
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 3)])
        for st in self._states.values():
            st.resize(width, height)

    def _state_for(self, layer_id: int) -> LayerState:
        st = self._states.get(layer_id)
        if st is None:
            st = LayerState(self.ctx, self.width, self.height)
            self._states[layer_id] = st
        return st

    def release_layer_state(self, layer_id: int) -> None:
        st = self._states.pop(layer_id, None)
        if st:
            st.release()

    # ── Image management (global bg + center) ────────────────────
    def load_background(self, path: str) -> None:
        if self._bg_texture:
            self._bg_texture.release()
        img = Image.open(path).convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)
        self._bg_img_aspect = img.size[0] / max(img.size[1], 1)
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

    # ── Composition pipeline ─────────────────────────────────────
    def render_composition(self, lm: LayerManager, bars: np.ndarray, pulse: float,
                           time: float = 0.0) -> None:
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self._render_bg(lm.bg, pulse)
        for layer in lm.active_layers():
            st = self._state_for(layer.id)
            self._render_layer(st, layer, bars, pulse, time)
            self._composite(st, layer)

    def _render_bg(self, bg: BGLayer, pulse: float) -> None:
        self.fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        has_img = 1 if (bg.image_path and self._bg_texture) else 0
        if has_img:
            self._bg_texture.use(location=0)
            self.bg_prog["u_tex"].value = 0
        self.bg_prog["u_has_image"].value     = has_img
        self.bg_prog["u_opacity"].value       = float(bg.opacity)
        self.bg_prog["u_img_aspect"].value    = float(self._bg_img_aspect)
        self.bg_prog["u_canvas_aspect"].value = float(self.width / self.height)
        zoom = 1.0 + (pulse * bg.zoom_intensity * 0.10 if bg.zoom_reactive else 0.0)
        self.bg_prog["u_zoom"].value = float(zoom)
        self.bg_vao.render(moderngl.TRIANGLE_STRIP)

    def _composite(self, st: LayerState, layer: Layer) -> None:
        self.fbo.use()
        self.ctx.enable(moderngl.BLEND)
        if layer.blend_mode == BLEND_ADDITIVE:
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        else:
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        st.fbo.color_attachments[0].use(location=0)
        self.composite_prog["u_tex"].value     = 0
        self.composite_prog["u_offset"].value  = (float(layer.x), float(layer.y))
        self.composite_prog["u_scale"].value   = float(layer.scale)
        self.composite_prog["u_opacity"].value = float(layer.opacity)
        self.composite_vao.render(moderngl.TRIANGLE_STRIP)

    # ── Single-layer viz render (into st.fbo, on black) ──────────
    def _render_layer(self, st: LayerState, layer: Layer,
                      bars: np.ndarray, pulse: float, time: float) -> None:
        # Auto-LFO: sweep flagged fields for this frame (drives preview AND export).
        apply_auto_lfo(layer, time)

        palette = layer.palette if layer.palette else _DEFAULT_PALETTE
        viz_type = layer.mode

        # Resample shared FFT bars to this layer's bar count
        if layer.num_bars != len(bars) and len(bars) > 1:
            xs = np.linspace(0, len(bars) - 1, layer.num_bars)
            lbars = np.interp(xs, np.arange(len(bars)), bars).astype(np.float32)
        else:
            lbars = bars

        # Per-band effect gains: scale the low / mid / high thirds of the render &
        # analysis bars — reaches EVERY mode (u_bars and the derived u_bass/mid/high
        # + kick). Never touches the played-back audio (separate sounddevice path).
        gl_, gm_, gh_ = layer.band_gain_low, layer.band_gain_mid, layer.band_gain_high
        if gl_ != 1.0 or gm_ != 1.0 or gh_ != 1.0:
            lbars = lbars.copy()
            nb = len(lbars)
            n_lo  = max(1, int(nb * 0.30))
            n_mid = max(n_lo + 1, int(nb * 0.70))
            lbars[:n_lo]      *= gl_
            lbars[n_lo:n_mid] *= gm_
            lbars[n_mid:]     *= gh_

        st.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        bar_data = np.zeros(_BARS_MAX, dtype="f4")
        n = min(len(lbars), _BARS_MAX)
        bar_data[:n] = lbars[:n]
        nh = min(n, _BASS_HIST_W)

        # Visual layers never draw the bg image; render on pure black for clean additive
        self.prog["u_has_bg"].value = 0
        self.prog["u_force_black_bg"].value = 1
        self.prog["u_bg_pulse_enabled"].value = 0
        self.prog["u_bg_pulse_intensity"].value = 0.0

        if self._center_texture:
            self._center_texture.use(location=1)
            self.prog["u_center_texture"].value = 1
            self.prog["u_has_center"].value = 1
        else:
            self.prog["u_has_center"].value = 0

        for i, rgb in enumerate(palette):
            self.prog[f"u_pal{i}"].value = tuple(rgb)

        self.prog["u_num_bars"].value = int(layer.num_bars)
        self._bars_tex.write(bar_data.tobytes())
        self._bars_tex.use(location=5)
        self.prog["u_bars_tex"].value = 5
        self.prog["u_pulse"].value = float(pulse)
        self.prog["u_pulse_intensity"].value = float(layer.pulse_intensity)
        self.prog["u_max_bar_height"].value = float(layer.max_bar_height)
        self.prog["u_circle_radius"].value = float(CIRCLE_RADIUS_RATIO)
        self.prog["u_bar_width"].value = float(BAR_WIDTH)
        self.prog["u_aspect"].value = float(self.width / self.height)
        self.prog["u_sensitivity"].value = float(layer.sensitivity)
        self.prog["u_viz_type"].value = int(viz_type)
        self.prog["u_rotation"].value = float(layer.rotation)
        self.prog["u_symmetry"].value = int(layer.symmetry)
        self.prog["u_mirror"].value = int(layer.mirror)
        self.prog["u_pal_mode"].value = int(layer.pal_mode)
        self.prog["u_halo_r_base"].value = float(layer.halo_r_base)
        self.prog["u_flash_enabled"].value = int(layer.flash)
        self.prog["u_flash_intensity"].value = float(layer.flash_intensity)

        # Nuclear Shockwave (mode 10) params
        self.prog["u_nuke_speed"].value = float(layer.nuke_speed)
        self.prog["u_nuke_life"].value = float(layer.nuke_life)
        self.prog["u_nuke_width"].value = float(layer.nuke_width)
        self.prog["u_nuke_flash"].value = float(layer.nuke_flash)
        self.prog["u_nuke_bg"].value = float(layer.nuke_bg)

        # Void Pull (mode 11) params
        self.prog["u_void_speed"].value = float(layer.void_speed)
        self.prog["u_void_pull"].value = float(layer.void_pull)
        self.prog["u_void_rays"].value = float(layer.void_rays)
        self.prog["u_void_arms"].value = int(layer.void_arms)

        # Psytrance modes (12-15) params
        self.prog["u_psy_speed"].value = float(layer.psy_speed)
        self.prog["u_psy_sat"].value = float(layer.psy_sat)
        self.prog["u_hue"].value = float(layer.hue)
        self.prog["u_myc_growth"].value = float(layer.myc_growth)
        self.prog["u_myc_density"].value = float(layer.myc_density)
        self.prog["u_myc_warp"].value = float(layer.myc_warp)
        self.prog["u_myc_branch"].value = float(layer.myc_branch)
        self.prog["u_myc_spore"].value = float(layer.myc_spore)
        self.prog["u_jul_zoom"].value = float(layer.jul_zoom)
        self.prog["u_jul_breathe"].value = float(layer.jul_breathe)
        self.prog["u_jul_glow"].value = float(layer.jul_glow)
        self.prog["u_jul_rotate"].value = float(layer.jul_rotate)
        self.prog["u_jul_morph"].value = float(layer.jul_morph)
        self.prog["u_jul_invert"].value = int(layer.jul_invert)
        # Alien Eye v2: advance the CPU saccade + blink state (only for mode 14).
        gx, gy, blink_amt = (self._update_eye(st, layer, time)
                             if viz_type == 14 else (0.0, 0.0, 0.0))
        self.prog["u_eye_pupil"].value = float(layer.eye_pupil)
        self.prog["u_eye_iris"].value = float(layer.eye_iris)
        self.prog["u_eye_crypts"].value = float(layer.eye_crypts)
        self.prog["u_eye_undul"].value = float(layer.eye_undul)
        self.prog["u_eye_warp"].value = float(layer.eye_warp)
        self.prog["u_eye_slit"].value = float(layer.eye_slit)
        self.prog["u_eye_dilate"].value = float(layer.eye_dilate)
        self.prog["u_eye_gaze"].value = (float(gx), float(gy))
        self.prog["u_eye_blink_amt"].value = float(blink_amt)
        self.prog["u_eye_fx_veins"].value = int(layer.eye_fx_veins)
        self.prog["u_eye_fx_noise"].value = int(layer.eye_fx_noise)
        self.prog["u_eye_fx_eyes"].value = int(layer.eye_fx_eyes)
        self.prog["u_eye_fx_holo"].value = int(layer.eye_fx_holo)
        self.prog["u_swp_scale"].value = float(layer.swp_scale)
        self.prog["u_swp_mutate"].value = float(layer.swp_mutate)
        self.prog["u_swp_turing"].value = float(layer.swp_turing)
        self.prog["u_swp_glow"].value = float(layer.swp_glow)
        self.prog["u_swp_flow"].value = float(layer.swp_flow)

        # Lissajous / Halo animation uniforms
        bar_slice = lbars[:n]
        total_energy = float(bar_slice.sum()) + 1e-6
        centroid = float(np.sum(bar_slice * np.arange(n)) / total_energy) / max(n, 1)
        lissa_energy = float(np.clip(bar_slice.mean() * layer.sensitivity * 1.5 + 0.15, 0.1, 0.85))
        self.prog["u_time"].value = float(time)
        self.prog["u_lissa_a"].value = 1.0 + centroid * 2.0
        self.prog["u_lissa_b"].value = 1.0 + (1.0 - centroid) * 2.0
        self.prog["u_lissa_energy"].value = lissa_energy

        # Bass waterfall history (mode 5)
        st.bass_history = np.roll(st.bass_history, 1, axis=0)
        st.bass_history[0, :nh] = lbars[:nh]
        st.bass_hist_tex.write(st.bass_history.tobytes())
        st.bass_hist_tex.use(location=2)
        self.prog["u_bass_history"].value = 2

        # Bass history 2 (mode 7) — bidirectional EMA
        st.bass_smooth2[:nh] = (st.bass_smooth2[:nh] * _BASS2_DECAY
                                + lbars[:nh] * (1.0 - _BASS2_DECAY))
        st.bass_history2 = np.roll(st.bass_history2, 1, axis=0)
        st.bass_history2[0, :nh] = st.bass_smooth2[:nh]
        st.bass_hist2_tex.write(st.bass_history2.tobytes())
        st.bass_hist2_tex.use(location=3)
        self.prog["u_bass_history2"].value = 3

        # Audio band decomposition + kick detection (Tunnel Arcade)
        self._update_kick(st, layer, lbars, time, pulse)

        # Spline modes: CPU analysis → control values, drawn by the shader
        if viz_type in (6, 9):
            self._update_spline(st, layer, lbars)

        self.vao.render(moderngl.TRIANGLE_STRIP)

    def _update_eye(self, st: LayerState, layer: Layer, time: float) -> tuple:
        """Advance Alien Eye v2 saccade (gaze) + blink state machines.

        A real eye fixates then jumps: the gaze target changes at random hold
        intervals and the current gaze eases toward it fast. Blink is a short
        closing→opening state machine fired at random intervals (sometimes double).
        Returns (gaze_x, gaze_y, blink_amount).
        """
        if not st.eye_seeded:
            st.eye_seeded = True
            st.eye_last_t = time
            st.eye_next_saccade = time
            st.eye_next_blink = time + 1.5
            return st.eye_gx, st.eye_gy, st.eye_blink_amt

        dt = time - st.eye_last_t
        st.eye_last_t = time
        if dt < 0.0:
            dt = 0.0
        elif dt > 0.05:
            dt = 0.05

        # ── Saccade ──
        scan_spd = max(layer.eye_scan_speed, 0.1)
        if time > st.eye_next_saccade:
            a = random.random() * 6.28318
            rad = random.random() * layer.eye_scan_amp
            st.eye_tx = math.cos(a) * rad
            st.eye_ty = math.sin(a) * rad * 0.7      # less vertical amplitude
            st.eye_next_saccade = time + (0.3 + random.random() * 0.8) / scan_spd
        k = 1.0 - 0.001 ** (dt * scan_spd * 6.0)
        st.eye_gx += (st.eye_tx - st.eye_gx) * k
        st.eye_gy += (st.eye_ty - st.eye_gy) * k

        # ── Blink ── (0 open, 1 shut) — short snap; ~1 in 8 is a double blink
        blink = layer.eye_blink
        CLOSE_DUR, OPEN_DUR = 0.06, 0.09
        if blink <= 0.0:
            st.eye_blink_amt = 0.0
        elif st.eye_blink_state == 0:               # idle
            if time > st.eye_next_blink:
                st.eye_blink_state = 1
                st.eye_blink_clock = 0.0
                st.eye_double_queued = random.random() < 0.12
        elif st.eye_blink_state == 1:               # closing
            st.eye_blink_clock += dt
            st.eye_blink_amt = min(1.0, st.eye_blink_clock / CLOSE_DUR)
            if st.eye_blink_clock >= CLOSE_DUR:
                st.eye_blink_state = 2
                st.eye_blink_clock = 0.0
        elif st.eye_blink_state == 2:               # opening
            st.eye_blink_clock += dt
            st.eye_blink_amt = max(0.0, 1.0 - st.eye_blink_clock / OPEN_DUR)
            if st.eye_blink_clock >= OPEN_DUR:
                st.eye_blink_amt = 0.0
                if st.eye_double_queued:
                    st.eye_double_queued = False
                    st.eye_blink_state = 1
                    st.eye_blink_clock = 0.0
                else:
                    st.eye_blink_state = 0
                    st.eye_next_blink = time + (1.5 + random.random() * 3.5) / max(blink, 0.1)

        return st.eye_gx, st.eye_gy, st.eye_blink_amt

    def _update_kick(self, st: LayerState, layer: Layer, bars: np.ndarray,
                     time: float, pulse: float) -> None:
        n_used = min(len(bars), layer.num_bars)
        bar_sl = bars[:n_used] if n_used > 0 else np.zeros(1, dtype="f4")
        n_bass = max(1, int(n_used * 0.30))
        n_mid  = max(n_bass + 1, int(n_used * 0.70))
        bass_v = float(np.mean(bar_sl[:n_bass]))
        mid_v  = float(np.mean(bar_sl[n_bass:n_mid]))
        high_v = float(np.mean(bar_sl[n_mid:]) if n_mid < n_used else 0.0)

        i_lo = max(0, int(n_used * layer.tunnel_kick_freq_lo / 100.0))
        i_hi = max(i_lo + 1, int(n_used * layer.tunnel_kick_freq_hi / 100.0))
        band = bar_sl[i_lo:i_hi] if i_hi <= n_used else bar_sl[i_lo:]
        band_energy = float(band.max()) if len(band) > 0 else 0.0

        sens = layer.tunnel_kick_sensitivity
        thr  = layer.tunnel_kick_threshold
        mode = layer.tunnel_kick_mode

        if mode == 1:
            if st.kick_bass_buf is None:
                st.kick_bass_buf = deque(maxlen=90)
            st.kick_bass_buf.append(bass_v)
            if st.kick_cooldown > 0:
                st.kick_cooldown -= 1
            rolling_max = max(st.kick_bass_buf) if st.kick_bass_buf else 1e-6
            if bass_v >= rolling_max * 0.75 and st.kick_cooldown == 0 and rolling_max > 1e-4:
                kick_n = float(np.clip(bass_v / max(rolling_max, 1e-6) * sens, 0.0, 1.0))
                st.kick_cooldown = 15
            else:
                kick_n = 0.0
        elif mode == 2:
            bg_alpha = 0.03
            st.kick_bg_ema = st.kick_bg_ema * (1.0 - bg_alpha) + band_energy * bg_alpha
            if st.kick_cooldown > 0:
                st.kick_cooldown -= 1
            ratio = band_energy / max(st.kick_bg_ema, 1e-5)
            if ratio >= thr and st.kick_cooldown == 0:
                kick_n = float(np.clip((ratio / max(thr, 1e-3) - 1.0) * sens, 0.0, 1.0))
                st.kick_cooldown = layer.tunnel_kick_cooldown
            else:
                kick_n = 0.0
        elif mode == 3:
            if st.kick_cooldown > 0:
                st.kick_cooldown -= 1
            if band_energy >= thr and st.kick_cooldown == 0:
                kick_n = float(np.clip(band_energy * sens, 0.0, 1.0))
                st.kick_cooldown = layer.tunnel_kick_cooldown
            else:
                kick_n = 0.0
        else:
            kick_n = float(np.clip((bass_v - st.prev_bass * 1.3) * 4.0 * sens, 0.0, 1.0))

        st.prev_bass = bass_v
        st.kick_accum = max(kick_n, st.kick_accum * 0.88)

        # Mode 10: spawn exactly one shockwave per kick. Adaptive onset on the
        # FFT pulse (sub-bass envelope) with hysteresis: the detector arms when
        # pulse falls back near its slow-tracking average, then fires once when it
        # rises a margin above that average. The slow-release pulse tail can't
        # re-trigger because the gate stays disarmed until pulse drops again.
        st.shock_avg = st.shock_avg * 0.96 + pulse * 0.04
        margin = layer.nuke_kick_threshold        # margin above background level
        hi = st.shock_avg + margin
        lo = st.shock_avg + margin * 0.4
        if st.shock_cooldown > 0:
            st.shock_cooldown -= 1
        if not st.shock_armed and pulse < lo:
            st.shock_armed = True
        if st.shock_armed and pulse > hi and pulse > 0.05 and st.shock_cooldown == 0:
            st.shock_times[st.shock_ptr] = float(time)
            st.shock_ptr = (st.shock_ptr + 1) % len(st.shock_times)
            st.shock_armed = False
            st.shock_cooldown = 6                  # safety floor (~0.1 s)

        # Band gains are already baked into lbars upstream (universal), so bass_v/
        # mid_v/high_v here already reflect them — no extra multiply.
        self.prog["u_bass"].value = bass_v
        self.prog["u_mid"].value = mid_v
        self.prog["u_high"].value = high_v
        self.prog["u_kick"].value = kick_n
        self.prog["u_kick_accum"].value = float(st.kick_accum)
        self.prog["u_shock_times"].value = tuple(st.shock_times.tolist())
        self.prog["u_tunnel_sides"].value = int(layer.tunnel_sides)
        self.prog["u_tunnel_rings"].value = int(layer.tunnel_rings)
        self.prog["u_tunnel_speed"].value = float(layer.tunnel_speed)
        self.prog["u_tunnel_kick_zoom"].value = float(layer.tunnel_kick_zoom)
        self.prog["u_tunnel_chroma"].value = float(layer.tunnel_chroma)
        self.prog["u_tunnel_bass_speed"].value = float(layer.tunnel_bass_speed)

    def _update_spline(self, st: LayerState, layer: Layer, bars: np.ndarray) -> None:
        n = max(2, min(int(layer.halo_n_points), _SPLINE_MAX))
        mode = st.halo_sine if layer.mode == 6 else st.flat_sine
        deform = mode.compute(bars, n_points=n,
                              smoothing_decay=layer.halo_smoothing_decay,
                              sensitivity=layer.sensitivity)
        data = np.zeros(_SPLINE_MAX, dtype="f4")
        data[:n] = deform
        st.spline_tex.write(data.tobytes())
        st.spline_tex.use(location=4)
        self.prog["u_spline_tex"].value = 4
        self.prog["u_spline_n"].value = n
        self.prog["u_sine_amp"].value = float(layer.halo_amplitude)
        self.prog["u_sine_gap"].value = float(layer.halo_spline_gap)
        self.prog["u_sine_glow"].value = int(layer.halo_glow_layers)
        self.prog["u_sine_fill"].value = float(layer.halo_fill_opacity)
        self.prog["u_sine_mean"].value = float(deform.mean())
        self.prog["u_center_pixel"].value = int(layer.halo_pixel_size)
        self.prog["u_res_y"].value = float(self.height)

    def draw_selection_outline(self, layer: Layer) -> None:
        """Preview-only: draw a highlight box around the selected layer."""
        self.fbo.use()
        self.ctx.disable(moderngl.BLEND)
        self.line_prog["u_offset"].value = (float(layer.x), float(layer.y))
        self.line_prog["u_scale"].value = float(layer.scale)
        self.line_prog["u_color"].value = (0.0, 0.9, 1.0)
        self.line_vao.render(moderngl.LINE_LOOP)

    # ── Output ───────────────────────────────────────────────────
    def read_frame(self) -> bytes:
        raw = self.fbo.read(components=3)
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        return np.ascontiguousarray(pixels[::-1]).tobytes()

    def release(self):
        for st in self._states.values():
            st.release()
        self._states.clear()
        if self._bg_texture:
            self._bg_texture.release()
        if self._center_texture:
            self._center_texture.release()
        self._bars_tex.release()
        self.vao.release()
        self.composite_vao.release()
        self.bg_vao.release()
        self.line_vao.release()
        self.vbo.release()
        self.line_vbo.release()
        if self.fbo:
            self.fbo.color_attachments[0].release()
            self.fbo.release()
        self.prog.release()
        self.composite_prog.release()
        self.bg_prog.release()
        self.line_prog.release()
        if self._owns_ctx:
            self.ctx.release()
