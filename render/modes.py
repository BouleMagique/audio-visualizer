"""CPU-side analysis for the spline modes (Halo Sine 6, Flat Sine 9).

Only the per-frame control values are computed here — auto-gain, per-point
smoothing, mirroring. The curves, glow, fill, center image and ring are all
drawn in radial.frag from these values (u_spline[]), so nothing is read back
from the GPU.
"""
import numpy as np


class _SplineAnalysis:
    """Rolling 95th-percentile auto-gain + fast-attack / decay smoothing."""

    _N_GAIN_FRAMES = 180  # auto-gain window (~3s at 60fps)

    def __init__(self):
        self._smoothed: np.ndarray | None = None
        self._gain_buf = np.zeros(self._N_GAIN_FRAMES, dtype=np.float32)
        self._gain_ptr = 0

    def _gain_and_smooth(self, energy: np.ndarray, smoothing_decay: float) -> np.ndarray:
        frame_peak = float(energy.max())
        self._gain_buf[self._gain_ptr] = frame_peak
        self._gain_ptr = (self._gain_ptr + 1) % self._N_GAIN_FRAMES
        active = self._gain_buf[self._gain_buf > 0]
        p95 = float(np.percentile(active, 95)) if len(active) > 0 else 1.0
        energy = energy / max(p95, 1e-6)

        if self._smoothed is None or len(self._smoothed) != len(energy):
            self._smoothed = np.zeros(len(energy), dtype=np.float32)
        self._smoothed = np.maximum(energy, self._smoothed * smoothing_decay)
        return self._smoothed


class HaloSineMode(_SplineAnalysis):
    """Closed reactive circle: bass mirrored at the top, treble toward the bottom."""

    def compute(self, bars: np.ndarray, n_points: int, smoothing_decay: float,
                sensitivity: float) -> np.ndarray:
        """→ (n_points,) radial deformation, 1.0 = amplitude_max (can exceed 1)."""
        # Bass range: first 65% of bars
        n_bass = max(1, int(len(bars) * 0.65))
        bass = bars[:n_bass]

        # Resample bass bins to n_half points (ceil so mirror always covers n_points)
        n_half = max(1, (n_points + 1) // 2)
        xs = np.linspace(0, n_bass - 1, n_half)
        energy_half = np.clip(
            np.interp(xs, np.arange(n_bass), bass) * sensitivity, 0.0, 1.0,
        ).astype(np.float32)

        half = self._gain_and_smooth(energy_half, smoothing_decay)
        # Mirror: bass at top (index 0), treble toward sides/bottom
        return np.concatenate([half, half[::-1]])[:n_points]


class FlatSineMode(_SplineAnalysis):
    """Horizontal waveform, mirrored top/bottom around the center line."""

    def compute(self, bars: np.ndarray, n_points: int, smoothing_decay: float,
                sensitivity: float) -> np.ndarray:
        """→ (n_points,) vertical deformation, 1.0 = amplitude_max."""
        n = len(bars)
        xs = np.linspace(0, n - 1, n_points)
        energy = np.clip(
            np.interp(xs, np.arange(n), bars) * sensitivity, 0.0, 1.0
        ).astype(np.float32)
        return self._gain_and_smooth(energy, smoothing_decay)
