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
    "Sine Plat": 9,
    "Nuclear Shockwave": 10,
    "Void Pull": 11,
}

# Modes proposés dans le sélecteur, dans l'ordre d'affichage.
# Les modes absents de cette liste restent implémentés (VIZ_TYPES, renderer)
# mais ne sont plus sélectionnables depuis l'UI.
VIZ_TYPES_VISIBLE = (
    "Halo Sine",
    "Halo",
    "Radial",
    "Linéaire",
    "Sine Plat",
    "Miroir",
    "Tunnel Arcade",
)

DEFAULT_VIZ_TYPE = VIZ_TYPES[VIZ_TYPES_VISIBLE[0]]

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
TUNNEL_KICK_MODE        = 0     # 0=delta 1=adaptive 2=kick spectral 3=freq seuil
TUNNEL_KICK_FREQ_LO     = 0     # band lower bound as % of num_bars
TUNNEL_KICK_FREQ_HI     = 12    # band upper bound as % of num_bars (~sub-bass)
TUNNEL_KICK_THRESHOLD   = 250   # ×0.01 → mode 2: ratio×BG (2.5×), mode 3: abs (0-1)
TUNNEL_KICK_COOLDOWN    = 20    # refractory period in frames

# Nuclear Shockwave mode defaults (mode 10)
NUKE_KICK_THRESHOLD = 30     # ×0.001 → pulse margin above adaptive avg to fire a wave
NUKE_SPEED          = 0.95   # shockwave expansion speed
NUKE_LIFE           = 2.2    # shockwave lifetime (seconds)
NUKE_WIDTH          = 1.0    # ring thickness multiplier
NUKE_FLASH          = 0.6    # whiteout intensity on impact
NUKE_BG             = 1.0    # nebula / starfield background brightness

# Void Pull mode defaults (mode 11)
VOID_SPEED = 0.10   # constant rotation speed (slow, hypnotic)
VOID_PULL  = 0.60   # gravity contraction strength on kick/bass (driven by pulse)
VOID_RAYS  = 2.8    # high-frequency light-ray intensity
VOID_ARMS  = 3      # spiral arm count

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
