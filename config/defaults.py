NUM_BARS = 128
MAX_BAR_HEIGHT = 1.0
SMOOTHING_DECAY = 0.85
PULSE_INTENSITY = 1.0
SENSITIVITY = 1.0
FFT_SIZE = 8192
FREQ_MIN = 20
FREQ_MAX = 16000
FPS = 60
EXPORT_RESOLUTION = "1080p"

RESOLUTIONS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4K": (3840, 2160),
}

# Each palette: list of 5 RGB tuples (float 0-1), from low to high amplitude
PALETTES = {
    "Violet / Magenta": [
        (0.322, 0.000, 1.000),
        (0.627, 0.125, 0.941),
        (1.000, 0.000, 0.800),
        (1.000, 0.400, 1.000),
        (1.000, 1.000, 1.000),
    ],
    "Feu": [
        (0.500, 0.000, 0.000),
        (1.000, 0.150, 0.000),
        (1.000, 0.550, 0.000),
        (1.000, 0.900, 0.100),
        (1.000, 1.000, 1.000),
    ],
    "Océan": [
        (0.000, 0.050, 0.500),
        (0.000, 0.350, 0.900),
        (0.000, 0.750, 1.000),
        (0.300, 1.000, 1.000),
        (1.000, 1.000, 1.000),
    ],
    "Néon vert": [
        (0.000, 0.150, 0.000),
        (0.000, 0.550, 0.100),
        (0.050, 1.000, 0.150),
        (0.500, 1.000, 0.500),
        (1.000, 1.000, 1.000),
    ],
    "Monochrome": [
        (0.080, 0.080, 0.080),
        (0.280, 0.280, 0.280),
        (0.550, 0.550, 0.550),
        (0.820, 0.820, 0.820),
        (1.000, 1.000, 1.000),
    ],
}

VIZ_TYPES = {
    "Radial": 0,
    "Miroir": 1,
    "Linéaire": 2,
    "Oscilloscope": 3,
    "Halo": 4,
    "Halo Bass": 5,
    "Halo Sine": 6,
    "Halo Bass 2": 7,
    "Tunnel Arcade": 8,
}

# Halo Sine mode defaults
HALO_SINE_R_BASE          = 0.35   # base radius as fraction of H
HALO_SINE_AMPLITUDE       = 0.18   # max radial displacement as fraction of H
HALO_SINE_N_POINTS        = 128    # spline control points
HALO_SINE_GLOW_LAYERS     = 3      # PIL glow passes
HALO_SINE_WAVE_SPEED      = 1.5    # traveling wave speed in rad/s
HALO_SINE_SMOOTHING_DECAY = 0.80   # per-point EMA decay (fast attack, slow release)
HALO_SINE_FILL_OPACITY    = 0.0    # interior fill opacity (0 = no fill)
HALO_SINE_SPLINE_GAP      = 1.4    # spline base radius as multiple of center circle radius
HALO_SINE_PIXEL_SIZE      = 1      # center image pixelation block size (1 = off)

# Tunnel Arcade mode defaults
TUNNEL_SIDES      = 8     # polygon sides (4 / 6 / 8 / 12)
TUNNEL_RINGS      = 4     # visible ring lines
TUNNEL_SPEED            = 0.3   # base advance speed
TUNNEL_KICK_ZOOM        = 1.0   # kick zoom intensity multiplier
TUNNEL_CHROMA           = 1.0   # chromatic aberration strength multiplier
TUNNEL_KICK_SENSITIVITY = 2.0   # multiplier on raw kick value before clamping
TUNNEL_BASS_SPEED       = 2.5   # bass reactivity on tunnel advance speed
TUNNEL_KICK_MODE        = 0     # 0 = delta, 1 = adaptive threshold

BG_PULSE_INTENSITY = 0.5   # BG zoom depth on kick (0-1)
FLASH_INTENSITY    = 0.5   # white flash opacity on kick (0-1)

BG_COLOR = (0.039, 0.039, 0.059, 1.0)
CIRCLE_RADIUS_RATIO = 0.18
BAR_WIDTH = 0.6

# Two-region log scale (mode FFT)
FREQ_BASS_SPLIT    = 0.60   # 60 % of bars → 20–FREQ_BASS_SPLIT_HZ
FREQ_BASS_SPLIT_HZ = 300.0  # Hz crossover

# CQT mode
CQT_BINS_PER_OCTAVE = 24    # 2 bins per semitone; Q ≈ 34.5
