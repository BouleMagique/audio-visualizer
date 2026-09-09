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
    "Mycelium": 12,
    "Julia Morph": 13,
    "Alien Eye": 14,
    "Swamp": 15,
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
    "Mycelium",
    "Julia Morph",
    "Alien Eye",
    "Swamp",
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

# ── Global effect-reactivity controls (all reactive modes) ──
# Kaleidoscope symmetry: folds the coordinate plane into N mirror sectors before
# the mode renders. 1 = off, 2..12 = mandala. Applies to every mode.
SYMMETRY = 1
# Per-band gains on the energy FED TO THE EFFECT ONLY (u_bass/u_mid/u_high).
# These NEVER touch the played-back audio — only the reactivity computation.
BAND_GAIN_LOW  = 1.0
BAND_GAIN_MID  = 1.0
BAND_GAIN_HIGH = 1.0
# Global hue shift 0-1 (full turn) — post hue-rotation on the final colour, all modes.
HUE = 0.0

# ── Psytrance modes (12-15) — ported from psytrance_visualizer_enhanced.html ──
# Shared "psy global" params, applied to all four psytrance modes.
PSY_SPEED = 1.0   # global animation speed multiplier (proto `spd`)
PSY_SAT   = 1.0   # global saturation post (proto `sat`)
# Auto-LFO (proto `~` toggles): sinusoidal auto-sweep of flagged sliders.
AUTO_SPEED  = 0.3   # LFO speed
AUTO_AMOUNT = 0.5   # LFO amplitude as fraction of each slider's half-range

# Mycelium (mode 12) — organic filament network
MYC_GROWTH  = 1.0   # overall growth intensity / brightness
MYC_DENSITY = 2.5   # filament density (base noise frequency)
MYC_WARP    = 0.6   # domain-warp strength (filament tortuousness)
MYC_BRANCH  = 0.06  # filament line thickness
MYC_SPORE   = 1.0   # luminous spores popping on kick

# Julia Morph (mode 13) — morphing Julia fractal / mandala
JUL_ZOOM     = 1.5   # base zoom level
JUL_BREATHE  = 0.13  # zoom breathing speed (proto `iterInf`)
JUL_GLOW     = 2.0   # edge halo intensity (proto `edgeGlow`)
JUL_ROTATE   = 0.05  # global rotation speed
JUL_MORPH    = 0.12  # constant-c morph amplitude (proto `cShift`)
JUL_INVERT   = 0     # invert palette (0/1)

# Alien Eye v2 (mode 14) — realistic almond eye, saccade scan, undulating crypts
EYE_PUPIL      = 0.14  # resting pupil radius
EYE_IRIS       = 0.40  # iris radius
EYE_CRYPTS     = 40    # crypt (radial fibre) density
EYE_UNDUL      = 1.0   # crypt undulation amplitude
EYE_WARP       = 1.0   # bass domain-warp strength on the iris
EYE_SLIT       = 0.0   # pupil shape: 0 = round, 1 = vertical reptilian slit
EYE_SCAN_SPEED = 1.0   # saccade speed (CPU gaze state machine)
EYE_SCAN_AMP   = 0.5   # saccade amplitude (0 = eye fixed on centre)
EYE_DILATE     = 1.0   # pupil dilation strength on kick
EYE_BLINK      = 1.0   # blink frequency (CPU blink state machine)
EYE_FX_VEINS   = 0     # sclera FX toggle: reactive neon veins
EYE_FX_NOISE   = 0     # sclera FX toggle: undulating fractal noise
EYE_FX_EYES    = 0     # sclera FX toggle: repeated small eyes / sacred motifs
EYE_FX_HOLO    = 0     # sclera FX toggle: pulsing holographic cloud

# Swamp (mode 15) — reaction-diffusion / Turing patterns
SWP_SCALE  = 4.0   # pattern scale
SWP_MUTATE = 1.0   # mutation speed
SWP_TURING = 0.4   # Turing threshold (blob finesse)
SWP_GLOW   = 1.0   # contour network glow (proto `cellGlow`)
SWP_FLOW   = 0.05  # global flow speed

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
