"""Layer composition model for the audio visualizer.

A composition is a stack of layers rendered bottom→top:
  - one BGLayer (background image / colour + reactive zoom, locked at the bottom)
  - up to MAX_VISUAL_LAYERS visual layers, each an independent instance of a viz mode

Layer objects are pure configuration + identity. The per-layer mutable render
state (audio history, PIL mode instances, FBO) lives renderer-side, keyed by
Layer.id — see render/renderer.py:LayerState.
"""
import math
from dataclasses import dataclass, field
from itertools import count

from config.defaults import (
    NUM_BARS, MAX_BAR_HEIGHT, PULSE_INTENSITY, SENSITIVITY, PALETTES,
    HALO_SINE_R_BASE, HALO_SINE_AMPLITUDE, HALO_SINE_N_POINTS,
    HALO_SINE_GLOW_LAYERS, HALO_SINE_SMOOTHING_DECAY, HALO_SINE_FILL_OPACITY,
    HALO_SINE_SPLINE_GAP, HALO_SINE_PIXEL_SIZE,
    TUNNEL_SIDES, TUNNEL_RINGS, TUNNEL_SPEED, TUNNEL_KICK_ZOOM, TUNNEL_CHROMA,
    TUNNEL_KICK_SENSITIVITY, TUNNEL_BASS_SPEED, TUNNEL_KICK_MODE,
    TUNNEL_KICK_FREQ_LO, TUNNEL_KICK_FREQ_HI,
    TUNNEL_KICK_THRESHOLD, TUNNEL_KICK_COOLDOWN,
    NUKE_KICK_THRESHOLD, NUKE_SPEED, NUKE_LIFE, NUKE_WIDTH, NUKE_FLASH, NUKE_BG,
    VOID_SPEED, VOID_PULL, VOID_RAYS, VOID_ARMS,
    SYMMETRY, BAND_GAIN_LOW, BAND_GAIN_MID, BAND_GAIN_HIGH, HUE,
    AUTO_SPEED, AUTO_AMOUNT,
    PSY_SPEED, PSY_SAT,
    MYC_GROWTH, MYC_DENSITY, MYC_WARP, MYC_BRANCH, MYC_SPORE,
    JUL_ZOOM, JUL_BREATHE, JUL_GLOW, JUL_ROTATE, JUL_MORPH, JUL_INVERT,
    EYE_PUPIL, EYE_IRIS, EYE_CRYPTS, EYE_UNDUL, EYE_WARP, EYE_SLIT,
    EYE_SCAN_SPEED, EYE_SCAN_AMP, EYE_DILATE, EYE_BLINK,
    EYE_FX_VEINS, EYE_FX_NOISE, EYE_FX_EYES, EYE_FX_HOLO,
    SWP_SCALE, SWP_MUTATE, SWP_TURING, SWP_GLOW, SWP_FLOW,
    FLASH_INTENSITY, BG_PULSE_INTENSITY, VIZ_TYPES, DEFAULT_VIZ_TYPE,
)

MAX_VISUAL_LAYERS = 6
BLEND_ADDITIVE = "additive"
BLEND_NORMAL   = "normal"

_id_counter = count(1)


def _default_palette() -> list:
    return [list(c) for c in list(PALETTES.values())[0]]


def apply_auto_lfo(layer, t: float) -> None:
    """Evaluate the layer's auto-LFO sweeps at time t and write them onto its fields.

    Called once per frame by the renderer, so the sweep drives BOTH the live preview
    and the frame-by-frame export identically (the UI is not involved). Each swept
    field oscillates around the MIDPOINT of its range, so amount = 1 covers the full
    min↔max independent of the field's own range. `pause` soft-clips the sine toward a
    square, dwelling at the extremes (0 = clean continuous sweep).
    """
    if not layer.auto:
        return
    speed, amount, pause = layer.auto_speed, layer.auto_amount, layer.auto_pause
    for field_name, spec in layer.auto.items():
        lo, hi, phase = spec
        s = math.sin(t * speed * 2.0 + phase)
        if pause > 0.001:
            g = pause * 6.0
            s = math.tanh(s * g) / math.tanh(g)
        mid = (hi + lo) * 0.5
        v = mid + s * (hi - lo) * 0.5 * amount
        v = max(lo, min(hi, v))
        cur = getattr(layer, field_name, None)
        setattr(layer, field_name, int(round(v)) if isinstance(cur, int) else v)


@dataclass
class Layer:
    """A single visual layer — an instance of a viz mode with its own transform."""
    mode: int = DEFAULT_VIZ_TYPE        # viz_type
    name: str = "Layer"

    # ── Composition / transform ──
    x: float = 0.0                      # NDC offset, 0 = centre
    y: float = 0.0
    scale: float = 1.0                  # 0.1 → 2.0
    opacity: float = 1.0               # 0.0 → 1.0
    blend_mode: str = BLEND_ADDITIVE
    visible: bool = True
    solo: bool = False

    # ── Generic viz params ──
    num_bars: int = NUM_BARS
    max_bar_height: float = MAX_BAR_HEIGHT
    pulse_intensity: float = PULSE_INTENSITY
    sensitivity: float = SENSITIVITY
    rotation: float = 0.0              # radians
    hue: float = HUE                   # global hue shift 0-1, all modes
    symmetry: int = SYMMETRY           # kaleidoscope sectors (1 = off), all modes
    band_gain_low: float = BAND_GAIN_LOW    # effect-only gain on u_bass
    band_gain_mid: float = BAND_GAIN_MID    # effect-only gain on u_mid
    band_gain_high: float = BAND_GAIN_HIGH  # effect-only gain on u_high
    # Auto-LFO: {field_name: [lo, hi, phase]} in field units, evaluated per frame
    # by apply_auto_lfo() in the renderer — so it drives both preview AND export.
    auto: dict = field(default_factory=dict)
    auto_speed: float = AUTO_SPEED
    auto_amount: float = AUTO_AMOUNT
    auto_pause: float = 0.0
    pal_mode: int = 0                  # 0 = amplitude, 1 = fréquence
    mirror: bool = False
    flash: bool = False
    flash_intensity: float = FLASH_INTENSITY
    palette: list = field(default_factory=_default_palette)
    palette_name: str = "Défaut"   # built-in presets removed; "Défaut" + 2 custom

    # ── Halo Sine / Flat Sine ──
    halo_r_base: float = HALO_SINE_R_BASE
    halo_amplitude: float = HALO_SINE_AMPLITUDE
    halo_n_points: int = HALO_SINE_N_POINTS
    halo_glow_layers: int = HALO_SINE_GLOW_LAYERS
    halo_smoothing_decay: float = HALO_SINE_SMOOTHING_DECAY
    halo_fill_opacity: float = HALO_SINE_FILL_OPACITY
    halo_spline_gap: float = HALO_SINE_SPLINE_GAP
    halo_pixel_size: int = HALO_SINE_PIXEL_SIZE

    # ── Tunnel Arcade ──
    tunnel_sides: int = TUNNEL_SIDES
    tunnel_rings: int = TUNNEL_RINGS
    tunnel_speed: float = TUNNEL_SPEED
    tunnel_kick_zoom: float = TUNNEL_KICK_ZOOM
    tunnel_chroma: float = TUNNEL_CHROMA
    tunnel_kick_sensitivity: float = TUNNEL_KICK_SENSITIVITY
    tunnel_bass_speed: float = TUNNEL_BASS_SPEED
    tunnel_kick_mode: int = TUNNEL_KICK_MODE
    tunnel_kick_freq_lo: int = TUNNEL_KICK_FREQ_LO
    tunnel_kick_freq_hi: int = TUNNEL_KICK_FREQ_HI
    tunnel_kick_threshold: float = TUNNEL_KICK_THRESHOLD / 100.0
    tunnel_kick_cooldown: int = TUNNEL_KICK_COOLDOWN

    # ── Nuclear Shockwave (mode 10) ──
    nuke_kick_threshold: float = NUKE_KICK_THRESHOLD / 1000.0
    nuke_speed: float = NUKE_SPEED
    nuke_life: float = NUKE_LIFE
    nuke_width: float = NUKE_WIDTH
    nuke_flash: float = NUKE_FLASH
    nuke_bg: float = NUKE_BG

    # ── Void Pull (mode 11) ──
    void_speed: float = VOID_SPEED
    void_pull: float = VOID_PULL
    void_rays: float = VOID_RAYS
    void_arms: int = VOID_ARMS

    # ── Psytrance modes (12-15) — shared "psy global" params ──
    psy_speed: float = PSY_SPEED
    psy_sat: float = PSY_SAT

    # ── Mycelium (mode 12) ──
    myc_growth: float = MYC_GROWTH
    myc_density: float = MYC_DENSITY
    myc_warp: float = MYC_WARP
    myc_branch: float = MYC_BRANCH
    myc_spore: float = MYC_SPORE

    # ── Julia Morph (mode 13) ──
    jul_zoom: float = JUL_ZOOM
    jul_breathe: float = JUL_BREATHE
    jul_glow: float = JUL_GLOW
    jul_rotate: float = JUL_ROTATE
    jul_morph: float = JUL_MORPH
    jul_invert: int = JUL_INVERT

    # ── Alien Eye v2 (mode 14) ──
    eye_pupil: float = EYE_PUPIL
    eye_iris: float = EYE_IRIS
    eye_crypts: int = EYE_CRYPTS
    eye_undul: float = EYE_UNDUL
    eye_warp: float = EYE_WARP
    eye_slit: float = EYE_SLIT
    eye_scan_speed: float = EYE_SCAN_SPEED
    eye_scan_amp: float = EYE_SCAN_AMP
    eye_dilate: float = EYE_DILATE
    eye_blink: float = EYE_BLINK
    eye_fx_veins: int = EYE_FX_VEINS
    eye_fx_noise: int = EYE_FX_NOISE
    eye_fx_eyes: int = EYE_FX_EYES
    eye_fx_holo: int = EYE_FX_HOLO

    # ── Swamp (mode 15) ──
    swp_scale: float = SWP_SCALE
    swp_mutate: float = SWP_MUTATE
    swp_turing: float = SWP_TURING
    swp_glow: float = SWP_GLOW
    swp_flow: float = SWP_FLOW

    id: int = field(default_factory=lambda: next(_id_counter))

    def mode_name(self) -> str:
        for name, idx in VIZ_TYPES.items():
            if idx == self.mode:
                return name
        return f"Mode {self.mode}"


@dataclass
class BGLayer:
    """Background layer — image (cover) or solid black, with bass-reactive zoom.

    Locked at the bottom of the stack, blend mode forced to normal.
    """
    image_path: str | None = None
    opacity: float = 1.0
    zoom_reactive: bool = False
    zoom_intensity: float = BG_PULSE_INTENSITY
    id: int = 0   # reserved id


class LayerManager:
    """Holds the BG layer + ordered visual layer stack and selection state."""

    def __init__(self):
        self.bg = BGLayer()
        self.layers: list[Layer] = []
        self._selected_id: int | None = None

    # ── Stack management ──
    def add_layer(self, mode: int = DEFAULT_VIZ_TYPE) -> Layer | None:
        if len(self.layers) >= MAX_VISUAL_LAYERS:
            return None
        layer = Layer(mode=mode)
        layer.name = layer.mode_name()
        self.layers.append(layer)
        self._selected_id = layer.id
        return layer

    def remove_layer(self, layer_id: int) -> None:
        self.layers = [l for l in self.layers if l.id != layer_id]
        if self._selected_id == layer_id:
            self._selected_id = self.layers[-1].id if self.layers else None

    def move_layer(self, from_idx: int, to_idx: int) -> None:
        if not (0 <= from_idx < len(self.layers)):
            return
        to_idx = max(0, min(to_idx, len(self.layers) - 1))
        layer = self.layers.pop(from_idx)
        self.layers.insert(to_idx, layer)

    def get(self, layer_id: int) -> Layer | None:
        for l in self.layers:
            if l.id == layer_id:
                return l
        return None

    def index_of(self, layer_id: int) -> int:
        for i, l in enumerate(self.layers):
            if l.id == layer_id:
                return i
        return -1

    # ── Selection ──
    @property
    def selected_id(self) -> int | None:
        return self._selected_id

    @selected_id.setter
    def selected_id(self, value: int | None) -> None:
        self._selected_id = value

    def selected(self) -> Layer | None:
        return self.get(self._selected_id) if self._selected_id is not None else None

    # ── Render order ──
    def active_layers(self) -> list[Layer]:
        """Layers to render, bottom→top, honouring solo and visibility."""
        soloed = [l for l in self.layers if l.solo]
        pool = soloed if soloed else self.layers
        return [l for l in pool if l.visible]
