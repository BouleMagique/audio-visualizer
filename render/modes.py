import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Blur radius for the frequency-mode gradient fill, as a fraction of one band's width.
# Enough to dissolve the flat-shaded band edges without smearing the whole gradient.
_FILL_BLUR_FRAC = 0.6


def _catmull_rom_open(points: list, subdivisions: int = 6) -> list:
    """Open Catmull-Rom spline — phantom clamping at endpoints."""
    n = len(points)
    if n < 2:
        return list(points)
    result = []
    for i in range(n - 1):
        p0 = points[max(0, i - 1)]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[min(n - 1, i + 2)]
        for k in range(subdivisions):
            t = k / subdivisions
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (2*p1[0] + (-p0[0]+p2[0])*t
                       + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2
                       + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
            y = 0.5 * (2*p1[1] + (-p0[1]+p2[1])*t
                       + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2
                       + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
            result.append((x, y))
    result.append(points[-1])
    return result


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


def _pal_colors(ts: np.ndarray, palette: list) -> np.ndarray:
    """Vectorised _pal_color: (n,) of t in 0..1 → (n, 3) uint8."""
    stops = np.asarray(palette, dtype=np.float32)[:, :3]      # 5 × 3
    t = np.clip(np.asarray(ts, dtype=np.float32), 0.0, 1.0) * 4.0
    i = np.minimum(t.astype(np.intp), 3)
    f = (t - i)[:, None]
    return ((stops[i] + (stops[i + 1] - stops[i]) * f) * 255).astype(np.uint8)


def _bbox(pts: list, W: int, H: int) -> tuple:
    """Integer bounding box of pts, clamped to the frame: (x0, y0, x1, y1)."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0 = max(0, int(min(xs)) - 1)
    y0 = max(0, int(min(ys)) - 1)
    x1 = max(x0 + 1, min(W, int(max(xs)) + 2))
    y1 = max(y0 + 1, min(H, int(max(ys)) + 2))
    return x0, y0, x1, y1


def _blur_scale(blur_px: float, box: tuple) -> int:
    """Resolution divisor for building a gradient that will be blurred by blur_px.

    A gradient carries no fine detail, so it is cheaper to draw it small and let one
    bilinear upscale do most of the softening than to rasterise and blur at full size.
    """
    x0, y0, x1, y1 = box
    scale = max(1, int(blur_px))
    while scale > 1 and ((x1 - x0) // scale < 8 or (y1 - y0) // scale < 8):
        scale -= 1
    return scale


def _expand_radial(pts: list, cx: float, cy: float, px: float) -> list:
    """Push points outward from (cx, cy) by px, so a blur has colour to pull from."""
    p = np.asarray(pts, dtype=np.float64)
    d = p - (cx, cy)
    r = np.hypot(d[:, 0], d[:, 1])
    k = np.where(r > 1e-6, (r + px) / np.maximum(r, 1e-6), 1.0)
    out = (cx, cy) + d * k[:, None]
    return [(float(x), float(y)) for x, y in out]


def _finish_gradient(small: Image.Image, box: tuple, scale: int, blur_px: float,
                     mask_pts: list, alpha: int, hole: tuple | None = None) -> Image.Image:
    """Upscale a small flat-shaded gradient, then cut it to the exact fill silhouette.

    Softening comes from the bilinear upscale plus a light Gaussian at the small size, so
    the flat band edges dissolve. The shape is applied afterwards at full resolution, which
    keeps the fill's outline crisp. `small` must cover more than mask_pts, otherwise the
    blur pulls its own transparent surround inward and leaves a dark rim.
    """
    x0, y0, x1, y1 = box
    if blur_px >= 0.5:
        small = small.filter(ImageFilter.GaussianBlur(max(0.5, blur_px / scale)))
    layer = small.resize((x1 - x0, y1 - y0), Image.BILINEAR) if scale > 1 else small

    mask = Image.new("L", layer.size, 0)
    md   = ImageDraw.Draw(mask)
    md.polygon([(x - x0, y - y0) for x, y in mask_pts], fill=alpha)
    if hole is not None:
        hx, hy, hr = hole
        md.ellipse([hx - hr - x0, hy - hr - y0, hx + hr - x0, hy + hr - y0], fill=0)
    del md
    layer.putalpha(mask)
    return layer


def _wedge_fan(curve_pts: list, cx: float, cy: float, box: tuple, scale: int,
               n_segs: int, palette: list, freq_of) -> Image.Image:
    """Opaque fan of wedges from (cx, cy) out to curve_pts, coloured along the curve.

    Drawn at 1/scale resolution — the result is a gradient that gets blurred and upscaled,
    so rasterising it at full size would only burn time.
    """
    x0, y0, x1, y1 = box
    layer = Image.new("RGBA", (max(1, (x1 - x0) // scale), max(1, (y1 - y0) // scale)),
                      (0, 0, 0, 0))
    draw  = ImageDraw.Draw(layer)
    seg_size = max(1, len(curve_pts) // max(n_segs, 1))
    apex = ((cx - x0) / scale, (cy - y0) / scale)
    for seg_i in range(n_segs):
        col   = _pal_color(float(np.clip(freq_of(seg_i, n_segs), 0.0, 1.0)), palette)
        start = seg_i * seg_size
        end   = min(start + seg_size + 1, len(curve_pts))
        pts   = [apex] + [((x - x0) / scale, (y - y0) / scale)
                          for x, y in curve_pts[start:end]]
        if len(pts) >= 3:
            draw.polygon(pts, fill=(*col, 255))
    # Close the ring: last control point back round to the first
    draw.polygon(
        [apex,
         ((curve_pts[-1][0] - x0) / scale, (curve_pts[-1][1] - y0) / scale),
         ((curve_pts[0][0] - x0) / scale, (curve_pts[0][1] - y0) / scale)],
        fill=(*_pal_color(float(np.clip(freq_of(0, n_segs), 0.0, 1.0)), palette), 255),
    )
    del draw
    return layer


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
        spline_gap: float = 1.4,   # spline base radius as multiple of center circle radius
    ) -> np.ndarray:
        H, W = frame_rgb.shape[:2]
        cx, cy = W / 2.0, H / 2.0
        r_glsl_px   = r_base * H / 2        # GLSL center circle radius in PIL pixels
        r_base_px   = r_glsl_px * spline_gap
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
            # Punch out center so the GLSL-rendered center image shows through.
            # Use r_glsl_px (= r_base * H/2) regardless of the spline scaling factor.
            r_inner = r_glsl_px
            if pal_mode == 0:
                fill_draw = ImageDraw.Draw(overlay)
                fill_draw.polygon(curve_pts, fill=(*base_col, fill_alpha))
                fill_draw.ellipse(
                    [cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner],
                    fill=(0, 0, 0, 0),
                )
                del fill_draw
            else:
                # Frequency mode: the fill follows the same angular gradient as the
                # strokes, as a fan of wedges from the centre. The wedges are flat-shaded,
                # so their edges would read as radial bands — the layer is drawn opaque and
                # oversized, blurred, then cut to the real silhouette.
                band_px = 2.0 * np.pi * max(r_base_px, 1.0) / max(n_points, 1)
                blur_px = max(1.0, band_px * _FILL_BLUR_FRAC)
                grown   = _expand_radial(curve_pts, cx, cy, blur_px + 2.0)
                box     = _bbox(grown + [(cx, cy)], W, H)
                scale   = _blur_scale(blur_px, box)
                fill_img = _wedge_fan(
                    grown, cx, cy, box, scale, n_points, palette,
                    lambda seg_i, n: 1.0 - abs(2.0 * seg_i / max(n - 1, 1) - 1.0),
                )
                fill_img = _finish_gradient(fill_img, box, scale, blur_px,
                                            curve_pts, fill_alpha,
                                            hole=(cx, cy, r_inner))
                # Straight paste, no mask: the overlay is still empty here, so this copies
                # the RGBA verbatim instead of squaring the alpha.
                overlay.paste(fill_img, (box[0], box[1]))

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

        # Paste rather than alpha_composite: same result, without converting the whole
        # 1920×1080 frame to RGBA and back on every frame.
        base_img = Image.fromarray(frame_rgb, "RGB")
        base_img.paste(overlay, (0, 0), overlay)
        return np.array(base_img)


class FlatSineMode:
    """PIL-based horizontal Catmull-Rom waveform, mirrored top/bottom."""

    _N_GAIN_FRAMES = 180

    def __init__(self):
        self._smoothed: np.ndarray | None = None
        self._gain_buf = np.zeros(self._N_GAIN_FRAMES, dtype=np.float32)
        self._gain_ptr = 0

    def draw_overlay(
        self,
        frame_rgb: np.ndarray,
        bars: np.ndarray,
        amplitude_max: float,
        n_points: int,
        glow_layers: int,
        smoothing_decay: float,
        sensitivity: float,
        palette: list,
        fill_opacity: float = 0.0,
        pal_mode: int = 0,
    ) -> np.ndarray:
        H, W = frame_rgb.shape[:2]
        cy     = H / 2.0
        amp_px = amplitude_max * H

        n  = len(bars)
        xs = np.linspace(0, n - 1, n_points)
        energy = np.clip(
            np.interp(xs, np.arange(n), bars) * sensitivity, 0.0, 1.0
        ).astype(np.float32)

        # Auto-gain (95th percentile over rolling window)
        frame_peak = float(energy.max())
        self._gain_buf[self._gain_ptr] = frame_peak
        self._gain_ptr = (self._gain_ptr + 1) % self._N_GAIN_FRAMES
        active = self._gain_buf[self._gain_buf > 0]
        p95 = float(np.percentile(active, 95)) if len(active) > 0 else 1.0
        energy = energy / max(p95, 1e-6)

        # Per-point smoothing
        if self._smoothed is None or len(self._smoothed) != n_points:
            self._smoothed = np.zeros(n_points, dtype=np.float32)
        self._smoothed = np.maximum(energy, self._smoothed * smoothing_decay)
        deform = self._smoothed * amp_px

        x_pts = np.linspace(0.0, float(W - 1), n_points)
        top_pts    = [(float(x_pts[i]), cy - deform[i]) for i in range(n_points)]
        bottom_pts = [(float(x_pts[i]), cy + deform[i]) for i in range(n_points)]

        top_curve    = _catmull_rom_open(top_pts)
        bottom_curve = _catmull_rom_open(bottom_pts)

        if not top_curve or not bottom_curve:
            return frame_rgb

        mean_t  = float(deform.mean()) / max(amp_px, 1e-6)
        base_col = _pal_color(np.clip(mean_t, 0.0, 1.0), palette)

        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))

        # Fill between curves
        if fill_opacity > 0.0:
            fill_alpha = int(np.clip(fill_opacity, 0.0, 1.0) * 255)
            if pal_mode == 0:
                draw = ImageDraw.Draw(overlay)
                draw.polygon(top_curve + bottom_curve[::-1], fill=(*base_col, fill_alpha))
                del draw
            else:
                # Frequency mode: same left-to-right gradient as the strokes. It only varies
                # with x, so it is built exactly — one palette lookup per column, broadcast
                # down the rows. No flat bands to blur away, and cheaper than rasterising
                # and blurring slabs.
                x0, y0, x1, y1 = _bbox(top_curve + bottom_curve, W, H)
                cols = _pal_colors(np.arange(x0, x1) / max(W - 1, 1), palette)
                rgb  = np.broadcast_to(cols[None, :, :], (y1 - y0, x1 - x0, 3))
                fill_img = Image.fromarray(np.ascontiguousarray(rgb), "RGB")

                mask = Image.new("L", fill_img.size, 0)
                md   = ImageDraw.Draw(mask)
                md.polygon([(x - x0, y - y0)
                            for x, y in top_curve + bottom_curve[::-1]], fill=fill_alpha)
                del md
                fill_img.putalpha(mask)
                overlay.paste(fill_img, (x0, y0))

        # Glow passes
        n_segs   = n_points
        seg_size = max(1, len(top_curve) // n_segs)

        for pass_i in range(glow_layers, 0, -1):
            frac      = 1.0 - (pass_i - 1) / max(glow_layers - 1, 1)
            width     = max(1, pass_i * 3)
            alpha     = int(40 + frac * 215)
            white_mix = frac * 0.65

            if pal_mode == 1:
                # Fréquence: color each segment by horizontal position
                for seg_i in range(n_segs):
                    freq_t  = seg_i / max(n_segs - 1, 1)
                    seg_col = _pal_color(freq_t, palette)
                    r = min(255, int(seg_col[0] * (1.0 - white_mix) + 255 * white_mix))
                    g = min(255, int(seg_col[1] * (1.0 - white_mix) + 255 * white_mix))
                    b = min(255, int(seg_col[2] * (1.0 - white_mix) + 255 * white_mix))
                    s = seg_i * seg_size
                    e = min(s + seg_size + 2, len(top_curve))
                    if e - s >= 2:
                        draw = ImageDraw.Draw(overlay)
                        draw.line(top_curve[s:e],    fill=(r, g, b, alpha), width=width)
                        draw.line(bottom_curve[s:e], fill=(r, g, b, alpha), width=width)
                        del draw
            else:
                r = min(255, int(base_col[0] * (1.0 - white_mix) + 255 * white_mix))
                g = min(255, int(base_col[1] * (1.0 - white_mix) + 255 * white_mix))
                b = min(255, int(base_col[2] * (1.0 - white_mix) + 255 * white_mix))
                draw = ImageDraw.Draw(overlay)
                draw.line(top_curve,    fill=(r, g, b, alpha), width=width)
                draw.line(bottom_curve, fill=(r, g, b, alpha), width=width)
                del draw

        base_img = Image.fromarray(frame_rgb, "RGB")
        base_img.paste(overlay, (0, 0), overlay)
        return np.array(base_img)
