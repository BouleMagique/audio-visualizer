import numpy as np
from scipy.fft import rfft
from scipy.signal.windows import hann
from config.defaults import (FFT_SIZE, FREQ_MIN, FREQ_MAX, NUM_BARS, SMOOTHING_DECAY,
                             FREQ_BASS_SPLIT, FREQ_BASS_SPLIT_HZ, CQT_BINS_PER_OCTAVE)

# Normalization: dBFS mapped to [0, 1]. bars / fft_size makes the result scale-invariant
# across FFT sizes. Typical music peaks at -40 to -15 dBFS per bin after freq weighting.
_DB_FLOOR = -70.0
_DB_CEIL  = -10.0
_DB_RANGE = _DB_CEIL - _DB_FLOOR


class FFTProcessor:
    def __init__(self, sr: int, fft_size: int = FFT_SIZE, num_bars: int = NUM_BARS,
                 freq_min: float = FREQ_MIN, freq_max: float = FREQ_MAX,
                 smoothing_decay: float = SMOOTHING_DECAY,
                 bass_split: float = FREQ_BASS_SPLIT,
                 bass_split_hz: float = FREQ_BASS_SPLIT_HZ,
                 use_cqt: bool = False,
                 bins_per_octave: int = CQT_BINS_PER_OCTAVE):
        self.sr = sr
        self.fft_size = fft_size
        self.num_bars = num_bars
        self.freq_min = freq_min
        # Below Nyquist there is no spectrum to bin: a 22 kHz file would otherwise
        # put the top bars past the last FFT bin and read nothing.
        self.freq_max = min(freq_max, sr * 0.5 * 0.99)
        self.smoothing_decay = smoothing_decay
        self.bass_split = bass_split
        self.bass_split_hz = bass_split_hz
        self.use_cqt = use_cqt
        self.bins_per_octave = bins_per_octave

        self.window = hann(fft_size, sym=False).astype(np.float32)
        self._smoothed = np.zeros(num_bars, dtype=np.float32)
        if use_cqt:
            self._build_cqt_bins()
        else:
            self._build_log_bins()
        self._pack_bin_ranges(fft_size // 2 + 1)

    def _pack_bin_ranges(self, n_freqs: int):
        """Turn _bin_ranges into flat arrays so process() can bin without a loop."""
        lo = np.array([r[0] for r in self._bin_ranges], dtype=np.intp)
        hi = np.array([r[1] for r in self._bin_ranges], dtype=np.intp)
        # Ranges the loop would have skipped (empty, or past the end of the spectrum)
        # stay flagged here and are zeroed in process().
        self._bin_valid  = (lo < hi) & (hi <= n_freqs)
        self._bin_lo     = np.clip(lo, 0, n_freqs)
        self._bin_hi     = np.clip(hi, 0, n_freqs)
        self._bin_counts = np.maximum(self._bin_hi - self._bin_lo, 1).astype(np.float64)

    def _build_cqt_bins(self):
        freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sr)

        # Q = f_center / bandwidth, constant for all bins
        # Q = 1 / (2^(1/bins_per_octave) - 1)
        Q = 1.0 / (2.0 ** (1.0 / self.bins_per_octave) - 1.0)

        # k spans exactly freq_min..freq_max, so the centers stay in range whatever
        # bins_per_octave is: high values give more CQT bins than bars (sub-sampled),
        # low values fewer (over-sampled, neighbouring bars overlap).
        n_octaves = np.log2(self.freq_max / self.freq_min)
        k_indices = np.linspace(0.0, n_octaves * self.bins_per_octave, self.num_bars)
        f_centers = self.freq_min * 2.0 ** (k_indices / self.bins_per_octave)

        self._bin_ranges = []
        for f_c in f_centers:
            half_bw = f_c / (2.0 * Q)
            lo = np.searchsorted(freqs, max(f_c - half_bw, freqs[1]))
            lo = min(lo, len(freqs) - 1)          # keep the slice non-empty
            hi = np.searchsorted(freqs, f_c + half_bw)
            hi = min(max(hi, lo + 1), len(freqs))
            self._bin_ranges.append((lo, hi))

        self._freq_weights = np.sqrt(f_centers / self.freq_min).astype(np.float32)

    def _build_log_bins(self):
        freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sr)

        # Two-region log scale: allocate bass_split% of bars to [freq_min, bass_split_hz]
        # and the rest to [bass_split_hz, freq_max]. Gives ~2× resolution in the bass band
        # vs a single log scale spanning the full range.
        xover = float(np.clip(self.bass_split_hz, self.freq_min * 2, self.freq_max / 2))
        n_bass = max(1, round(self.num_bars * self.bass_split))
        n_high = self.num_bars - n_bass

        edges_bass = np.logspace(np.log10(self.freq_min), np.log10(xover), n_bass + 1)
        edges_high = np.logspace(np.log10(xover), np.log10(self.freq_max), n_high + 1)
        # Drop the duplicate crossover point at the junction
        log_edges = np.concatenate([edges_bass[:-1], edges_high])  # num_bars + 1 edges

        self._bin_ranges = []
        bar_freqs = np.zeros(self.num_bars)

        for i in range(self.num_bars):
            lo = np.searchsorted(freqs, log_edges[i])
            hi = np.searchsorted(freqs, log_edges[i + 1])
            hi = max(hi, lo + 1)
            self._bin_ranges.append((lo, hi))
            bar_freqs[i] = np.sqrt(log_edges[i] * log_edges[i + 1])

        self._freq_weights = np.sqrt(bar_freqs / self.freq_min).astype(np.float32)

    def process(self, samples: np.ndarray) -> tuple[np.ndarray, float]:
        windowed = samples * self.window
        spectrum = np.abs(rfft(windowed))

        # Bin means as differences of a running sum: one vectorised pass instead of a
        # Python loop over up to 256 ranges. Ranges may overlap, which is fine — each
        # one is read independently.
        # float64 accumulation: spectrum is float32, and a running sum over 4000+ bins
        # loses too much precision to differencing at single precision.
        csum = np.concatenate(([0.0], np.cumsum(spectrum, dtype=np.float64)))
        bars = (csum[self._bin_hi] - csum[self._bin_lo]) / self._bin_counts
        # Empty or out-of-range slices would mean() to NaN, and NaN sticks forever in
        # the renderers' EMA state — never let one out of here.
        bars = np.where(self._bin_valid, bars, 0.0).astype(np.float32)

        # Apply frequency emphasis before log (multiplicative in linear = additive in dB)
        bars *= self._freq_weights

        # Convert to dBFS, normalized by FFT size so the result is independent of
        # fft_size and comparable to dBFS levels (0 dBFS = full scale sine per bin)
        bars_db = 20.0 * np.log10(bars / self.fft_size + 1e-7)

        # Map [_DB_FLOOR, _DB_CEIL] → [0, 1]
        bars_norm = (bars_db - _DB_FLOOR) / _DB_RANGE
        bars_norm = np.clip(bars_norm, 0.0, 1.0).astype(np.float32)

        # Temporal smoothing: fast attack, slow release
        self._smoothed = np.maximum(bars_norm, self._smoothed * self.smoothing_decay)

        # Sub-bass energy (lowest 10% of bars) drives the circle pulse
        sub_n = max(1, self.num_bars // 10)
        pulse = float(self._smoothed[:sub_n].mean())

        return self._smoothed.copy(), pulse

    def reset(self):
        self._smoothed[:] = 0.0
