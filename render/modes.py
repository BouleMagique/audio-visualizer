import numpy as np
from PIL import Image, ImageDraw


def _catmull_rom_closed(points: list, subdivisions: int = 6) -> list:
    """Closed Catmull-Rom spline — returns list of (float, float) tuples."""
    n = len(points)
    result = []
    for i in range(n):
        p0 = points[(i - 1) % n]
        p1 = points[i]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]
        for k in range(subdivisions):
            t = k / subdivisions
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                2*p1[0] + (-p0[0]+p2[0])*t
                + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2
                + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3
            )
            y = 0.5 * (
                2*p1[1] + (-p0[1]+p2[1])*t
                + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2
                + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3
            )
            result.append((x, y))
    return result


def _pal_color(t: float, palette: list) -> tuple:
    """5-stop palette lookup → (R, G, B) as 0-255 ints."""
    t = max(0.0, min(1.0, t))
    if t < 0.25:
        a, b, f = palette[0], palette[1], t * 4.0
    elif t < 0.50:
        a, b, f = palette[1], palette[2], (t - 0.25) * 4.0
    elif t < 0.75:
        a, b, f = palette[2], palette[3], (t - 0.50) * 4.0
    else:
        a, b, f = palette[3], palette[4], (t - 0.75) * 4.0
    return tuple(int((a[i] + (b[i] - a[i]) * f) * 255) for i in range(3))


class HaloSineMode:
    """PIL-based closed Catmull-Rom spline — reactive circular waveform."""

    _N_GAIN_FRAMES = 180  # auto-gain window (~3s at 60fps)

    def __init__(self):
        self._smoothed: np.ndarray | None = None
        self._gain_buf = np.zeros(self._N_GAIN_FRAMES, dtype=np.float32)
        self._gain_ptr = 0

    def draw_overlay(
        self,
        frame_rgb: np.ndarray,    # H×W×3 uint8, top-to-bottom
        bars: np.ndarray,
        r_base: float,            # base radius as fraction of H
        amplitude_max: float,     # max radial displacement as fraction of H
        n_points: int,
        glow_layers: int,
        smoothing_decay: float,   # per-point EMA decay (0–1, fast attack)
        sensitivity: float,
        palette: list,
        rotation: float = 0.0,    # starting angle offset in radians
        fill_opacity: float = 0.0, # interior fill opacity (0–1)
        pal_mode: int = 0,         # 0 = amplitude, 1 = fréquence
    ) -> np.ndarray:
        H, W = frame_rgb.shape[:2]
        cx, cy = W / 2.0, H / 2.0
        r_glsl_px   = r_base * H / 2        # GLSL center circle radius in PIL pixels
        r_base_px   = r_glsl_px * 1.4       # spline just outside center image (~40% beyond)
        amp_max_px  = amplitude_max * H

        # Bass range: first 65% of bars
        n = len(bars)
        n_bass = max(1, int(n * 0.65))
        bass = bars[:n_bass]

        # Resample bass bins to n_half points (ceil so mirror always covers n_points)
        n_half = max(1, (n_points + 1) // 2)
        xs = np.linspace(0, n_bass - 1, n_half)
        energy_half = np.clip(
            np.interp(xs, np.arange(n_bass), bass) * sensitivity,
            0.0, 1.0,
        ).astype(np.float32)

        # Auto-gain: 95th percentile of per-frame peak over _N_GAIN_FRAMES
        frame_peak = float(energy_half.max())
        self._gain_buf[self._gain_ptr] = frame_peak
        self._gain_ptr = (self._gain_ptr + 1) % self._N_GAIN_FRAMES
        active = self._gain_buf[self._gain_buf > 0]
        p95 = float(np.percentile(active, 95)) if len(active) > 0 else 1.0
        energy_half = energy_half / max(p95, 1e-6)

        # Per-point smoothing: fast attack (max), configurable release (decay)
        if self._smoothed is None or len(self._smoothed) != n_half:
            self._smoothed = np.zeros(n_half, dtype=np.float32)
        self._smoothed = np.maximum(energy_half, self._smoothed * smoothing_decay)

        # Deformation in pixels — never negative (bass pushes outward only)
        deform_half = self._smoothed * amp_max_px

        # Mirror: bass at top (index 0), treble toward sides/bottom
        # Truncate to n_points so odd counts work correctly
        deform = np.concatenate([deform_half, deform_half[::-1]])[:n_points]

        # Control points: start at -π/2 (top) + rotation offset
        angles = np.linspace(-np.pi / 2.0 + rotation, -np.pi / 2.0 + rotation + 2.0 * np.pi,
                             n_points, endpoint=False)
        radii = r_base_px + deform

        ctrl_pts = [
            (cx + radii[i] * np.cos(angles[i]),
             cy + radii[i] * np.sin(angles[i]))
            for i in range(n_points)
        ]

        curve_pts = _catmull_rom_closed(ctrl_pts, subdivisions=6)
        if len(curve_pts) < 2:
            return frame_rgb

        # Color from mean deformation level
        mean_t = float(deform.mean()) / max(amp_max_px, 1e-6)
        base_col = _pal_color(np.clip(mean_t, 0.0, 1.0), palette)

        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))

        # Interior fill (drawn first, under glow strokes)
        if fill_opacity > 0.0:
            fill_alpha = int(np.clip(fill_opacity, 0.0, 1.0) * 255)
            fill_draw = ImageDraw.Draw(overlay)
            fill_draw.polygon(curve_pts, fill=(*base_col, fill_alpha))
            # Punch out center so the GLSL-rendered center image shows through.
            # Use r_glsl_px (= r_base * H/2) regardless of the spline scaling factor.
            r_inner = r_glsl_px
            fill_draw.ellipse(
                [cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner],
                fill=(0, 0, 0, 0),
            )
            del fill_draw

        # Glow passes: outermost (widest, dimmest) → core (narrowest, brightest)
        if pal_mode == 0:
            # Amplitude mode: single color for the whole curve
            for pass_i in range(glow_layers, 0, -1):
                frac      = 1.0 - (pass_i - 1) / max(glow_layers - 1, 1)
                width     = max(1, pass_i * 3)
                alpha     = int(40 + frac * 215)
                white_mix = frac * 0.65
                r = min(255, int(base_col[0] * (1.0 - white_mix) + 255 * white_mix))
                g = min(255, int(base_col[1] * (1.0 - white_mix) + 255 * white_mix))
                b = min(255, int(base_col[2] * (1.0 - white_mix) + 255 * white_mix))
                draw = ImageDraw.Draw(overlay)
                pts = curve_pts + [curve_pts[0]]
                draw.line(pts, fill=(r, g, b, alpha), width=width)
                del draw
        else:
            # Fréquence mode: chaque segment coloré selon sa position angulaire
            # (0=grave/haut, 1=aigu/bas, miroir → 0 à la fin)
            n_segs    = n_points
            seg_size  = max(1, len(curve_pts) // n_segs)
            for pass_i in range(glow_layers, 0, -1):
                frac      = 1.0 - (pass_i - 1) / max(glow_layers - 1, 1)
                width     = max(1, pass_i * 3)
                alpha     = int(40 + frac * 215)
                white_mix = frac * 0.65
                for seg_i in range(n_segs):
                    # freq_t : 0 en haut (grave), 1 en bas (aigu), retour à 0 (miroir)
                    freq_t    = 1.0 - abs(2.0 * seg_i / max(n_segs - 1, 1) - 1.0)
                    seg_col   = _pal_color(float(freq_t), palette)
                    r = min(255, int(seg_col[0] * (1.0 - white_mix) + 255 * white_mix))
                    g = min(255, int(seg_col[1] * (1.0 - white_mix) + 255 * white_mix))
                    b = min(255, int(seg_col[2] * (1.0 - white_mix) + 255 * white_mix))
                    start = seg_i * seg_size
                    end   = min(start + seg_size + 2, len(curve_pts))
                    seg_pts = curve_pts[start:end]
                    if len(seg_pts) >= 2:
                        draw = ImageDraw.Draw(overlay)
                        draw.line(seg_pts, fill=(r, g, b, alpha), width=width)
                        del draw
                # Fermer la boucle (dernier → premier point)
                close_col = _pal_color(0.0, palette)
                r = min(255, int(close_col[0] * (1.0 - white_mix) + 255 * white_mix))
                g = min(255, int(close_col[1] * (1.0 - white_mix) + 255 * white_mix))
                b = min(255, int(close_col[2] * (1.0 - white_mix) + 255 * white_mix))
                draw = ImageDraw.Draw(overlay)
                draw.line([curve_pts[-1], curve_pts[0]], fill=(r, g, b, alpha), width=width)
                del draw

        base_img   = Image.fromarray(frame_rgb, "RGB").convert("RGBA")
        composited = Image.alpha_composite(base_img, overlay)
        return np.array(composited.convert("RGB"))
