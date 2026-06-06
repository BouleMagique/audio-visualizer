"""Layer composition model for the audio visualizer.

A composition is a stack of layers rendered bottom→top:
  - one BGLayer (background image / colour + reactive zoom, locked at the bottom)
  - up to MAX_VISUAL_LAYERS visual layers, each an independent instance of a viz mode

Layer objects are pure configuration + identity. The per-layer mutable render
state (audio history, PIL mode instances, FBO) lives renderer-side, keyed by
Layer.id — see render/renderer.py:LayerState.
"""
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
    FLASH_INTENSITY, BG_PULSE_INTENSITY, VIZ_TYPES,
)

MAX_VISUAL_LAYERS = 6
BLEND_ADDITIVE = "additive"
BLEND_NORMAL   = "normal"

_id_counter = count(1)


def _default_palette() -> list:
    return [list(c) for c in list(PALETTES.values())[0]]


@dataclass
class Layer:
    """A single visual layer — an instance of a viz mode with its own transform."""
    mode: int = 0                       # viz_type
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
    pal_mode: int = 0                  # 0 = amplitude, 1 = fréquence
    mirror: bool = False
    flash: bool = False
    flash_intensity: float = FLASH_INTENSITY
    palette: list = field(default_factory=_default_palette)
    palette_name: str = field(default_factory=lambda: list(PALETTES.keys())[0])

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
    def add_layer(self, mode: int = 0) -> Layer | None:
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
