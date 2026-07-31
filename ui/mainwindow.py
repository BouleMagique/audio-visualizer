import math
import copy
import time
import numpy as np
import sounddevice as sd
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QComboBox,
    QProgressBar, QGroupBox, QFormLayout, QSizePolicy, QSpinBox, QCheckBox,
    QColorDialog, QListWidget, QListWidgetItem, QAbstractItemView, QScrollArea,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QColor

from core.audio import AudioFile
from core.fft import FFTProcessor
from core.layer import Layer, LayerManager, MAX_VISUAL_LAYERS, BLEND_ADDITIVE, BLEND_NORMAL
from ui.preview import PreviewWidget
from config.defaults import (
    NUM_BARS, MAX_BAR_HEIGHT, SMOOTHING_DECAY, PULSE_INTENSITY,
    SENSITIVITY, FPS, RESOLUTIONS, FFT_SIZE, PALETTES, VIZ_TYPES,
    VIZ_TYPES_VISIBLE, DEFAULT_VIZ_TYPE,
    FREQ_BASS_SPLIT, FREQ_BASS_SPLIT_HZ, CQT_BINS_PER_OCTAVE,
    HALO_SINE_R_BASE, HALO_SINE_AMPLITUDE, HALO_SINE_N_POINTS,
    HALO_SINE_GLOW_LAYERS, HALO_SINE_SMOOTHING_DECAY, HALO_SINE_FILL_OPACITY,
    HALO_SINE_SPLINE_GAP, HALO_SINE_PIXEL_SIZE,
    TUNNEL_SIDES, TUNNEL_RINGS, TUNNEL_SPEED, TUNNEL_KICK_ZOOM, TUNNEL_CHROMA,
    TUNNEL_KICK_SENSITIVITY, TUNNEL_BASS_SPEED, TUNNEL_KICK_MODE,
    TUNNEL_KICK_FREQ_LO, TUNNEL_KICK_FREQ_HI,
    TUNNEL_KICK_THRESHOLD, TUNNEL_KICK_COOLDOWN,
    NUKE_KICK_THRESHOLD, NUKE_SPEED, NUKE_LIFE, NUKE_WIDTH, NUKE_FLASH, NUKE_BG,
    VOID_SPEED, VOID_PULL, VOID_RAYS, VOID_ARMS,
    BG_PULSE_INTENSITY, FLASH_INTENSITY,
)

FFT_ANALYSIS_BARS = 256   # global analysis resolution; layers resample from this

_RADIAL_MODES     = {0, 4, 5, 6, 7}
_ROTATION_MODES   = {0, 4, 5, 6, 7}
_BAR_MODES        = {0, 1, 2, 4, 5, 7}
_HEIGHT_MODES     = {0, 1, 2, 4, 5, 7}
_HALO_MODES       = {4, 5, 6, 7, 8}
_HALO_SINE_MODES  = {6, 9}
_TUNNEL_MODES     = {8}
_MIRROR_MODES     = {0, 4}
_NUKE_MODES       = {10}
_VOID_MODES       = {11}

_AMP_LABELS  = ["Silence", "Faible", "Moyen", "Fort", "Saturation"]
_FREQ_LABELS = ["Grave", "Basse", "Médium", "Présence", "Aigu"]
_PERSO_AMP   = "Perso. Amplitude"
_PERSO_FREQ  = "Perso. Fréquence"


class ExportWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(str)
    error    = Signal(str)

    def __init__(self, audio, layer_manager, output_dir, resolution, fps,
                 smoothing_decay, bass_split, bass_split_hz,
                 use_cqt, bins_per_octave,
                 bg_image_path, center_image_path):
        super().__init__()
        self._audio = audio
        self._lm = layer_manager
        self._kwargs = dict(
            output_dir=output_dir, resolution=resolution, fps=fps,
            smoothing_decay=smoothing_decay,
            bass_split=bass_split, bass_split_hz=bass_split_hz,
            use_cqt=use_cqt, bins_per_octave=bins_per_octave,
            bg_image_path=bg_image_path, center_image_path=center_image_path,
        )

    def run(self):
        try:
            from export.ffmpeg_export import FFmpegExporter
            exporter = FFmpegExporter(
                audio=self._audio,
                layer_manager=self._lm,
                progress_cb=lambda cur, tot: self.progress.emit(cur, tot),
                **self._kwargs,
            )
            out = exporter.export()
            self.finished.emit(out)
        except Exception as e:
            self.error.emit(str(e))


_THREAD_STOP_TIMEOUT_MS = 2000


class AudioPlaybackThread(QThread):
    frame_ready      = Signal(object, float)
    position_changed = Signal(int, int)

    def __init__(self, audio: AudioFile, fft: FFTProcessor, fps: int,
                 start_frame: int = 0, volume: float = 1.0):
        super().__init__()
        self._audio       = audio
        self._fft         = fft
        self._fps         = fps
        self._running     = False
        self._start_frame = start_frame
        self._volume      = volume

    def update_fft(self, fft: FFTProcessor):
        self._fft = fft

    def run(self):
        self._running = True
        hop   = max(1, int(self._audio.sr / self._fps))
        total = int(len(self._audio.mono) / hop)

        start_sample = self._start_frame * hop
        audio_out    = self._audio.mono[start_sample:] * self._volume
        sd.play(audio_out, samplerate=self._audio.sr)

        t0 = time.perf_counter()
        i  = self._start_frame
        while self._running and i < total:
            samples = self._audio.get_frame_samples(i, hop, FFT_SIZE)
            bars, pulse = self._fft.process(samples)
            self.frame_ready.emit(bars, pulse)
            self.position_changed.emit(i, total)

            # The next frame is whichever one the elapsed time points at, not i+1: a slow
            # render drops frames instead of pushing the whole preview behind the audio.
            # int(1000/fps) also truncates (16 ms for 60 fps, not 16.667), which on its own
            # walked the preview 3% off the audio clock over a track.
            # Target the next boundary strictly ahead of now (+1): when rendering runs
            # late this still yields for the rest of the frame instead of spinning with
            # no sleep at all, which would starve the GUI thread of the GIL.
            elapsed = time.perf_counter() - t0
            nxt     = max(i + 1, self._start_frame + int(elapsed * self._fps) + 1)
            delay   = (nxt - self._start_frame) / self._fps - elapsed
            if delay > 0:
                self.msleep(max(1, int(delay * 1000)))
            i = nxt
        sd.stop()

    def stop(self):
        self._running = False
        sd.stop()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Visualizer")
        self.resize(1340, 820)

        self._audio: AudioFile | None = None
        self._fft:   FFTProcessor | None = None
        self._playback_thread: AudioPlaybackThread | None = None
        self._export_thread:   QThread | None = None
        self._bg_image_path:     str | None = None
        self._center_image_path: str | None = None
        self._volume: float    = 1.0
        self._seek_frame: int  = 0
        self._seeking: bool    = False
        self._loading: bool    = False    # guards UI→layer writes during load

        self._lm = LayerManager()
        self._lm.add_layer(mode=DEFAULT_VIZ_TYPE)

        # Editing buffers for the custom palettes of the selected layer
        _default = [list(c) for c in list(PALETTES.values())[0]]
        self._custom_palette_amp  = [list(c) for c in _default]
        self._custom_palette_freq = [list(c) for c in _default]

        self._build_ui()
        self._preview.set_layer_manager(self._lm)
        self._refresh_layer_list()
        self._load_layer_into_ui(self._lm.selected())

    # ── UI builders ──────────────────────────────────────────────
    def _make_slider_row(self, layout: QFormLayout, label: str,
                         lo: int, hi: int, val: int, callback=None) -> tuple:
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)

        sl = QSlider(Qt.Horizontal)
        sl.setRange(lo, hi)
        sl.setValue(val)
        sl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        sb = QSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(val)
        sb.setButtonSymbols(QSpinBox.NoButtons)
        sb.setMinimumWidth(90)
        sb.setFixedWidth(90)

        sl.valueChanged.connect(sb.setValue)
        sb.valueChanged.connect(sl.setValue)
        if callback:
            sl.valueChanged.connect(callback)

        h.addWidget(sl)
        h.addWidget(sb)

        lbl = QLabel(label)
        layout.addRow(lbl, container)
        return sl, lbl, container

    def _set_row_visible(self, row: tuple, visible: bool):
        row[1].setVisible(visible)
        row[2].setVisible(visible)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(348)
        root.addWidget(scroll)

        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setAlignment(Qt.AlignTop)
        scroll.setWidget(panel)

        # ── Audio ──
        ag = QGroupBox("Audio")
        al = QVBoxLayout(ag)
        self._file_label = QLabel("Aucun fichier")
        self._file_label.setWordWrap(True)
        btn_open = QPushButton("Ouvrir fichier audio…")
        btn_open.clicked.connect(self._open_file)
        al.addWidget(self._file_label)
        al.addWidget(btn_open)
        pl.addWidget(ag)

        # ── Layers ──
        lg = QGroupBox("Calques")
        lgl = QVBoxLayout(lg)
        self._layer_list = QListWidget()
        self._layer_list.setDragDropMode(QAbstractItemView.InternalMove)
        self._layer_list.setMaximumHeight(150)
        self._layer_list.currentRowChanged.connect(self._on_layer_selected)
        self._layer_list.model().rowsMoved.connect(self._on_layers_reordered)
        lgl.addWidget(self._layer_list)

        lbtns = QHBoxLayout()
        btn_add = QPushButton("＋")
        btn_add.clicked.connect(self._add_layer)
        btn_del = QPushButton("🗑")
        btn_del.clicked.connect(self._remove_selected_layer)
        self._btn_layer_vis = QPushButton("👁")
        self._btn_layer_vis.clicked.connect(self._toggle_selected_visible)
        self._btn_layer_solo = QPushButton("S")
        self._btn_layer_solo.clicked.connect(self._toggle_selected_solo)
        for b in (btn_add, btn_del, self._btn_layer_vis, self._btn_layer_solo):
            b.setFixedWidth(44)
            lbtns.addWidget(b)
        lbtns.addStretch()
        lgl.addLayout(lbtns)
        pl.addWidget(lg)

        # ── Transform du calque ──
        tg = QGroupBox("Calque — transform")
        tf = QFormLayout(tg)
        self._row_scale = self._make_slider_row(
            tf, "Échelle (×0.01)", 10, 200, 100, self._on_params_changed)
        self._row_opacity = self._make_slider_row(
            tf, "Opacité (×0.01)", 0, 100, 100, self._on_params_changed)
        self._combo_blend = QComboBox()
        self._combo_blend.addItems(["Additif", "Normal"])
        self._combo_blend.currentIndexChanged.connect(self._on_params_changed)
        tf.addRow("Fusion", self._combo_blend)
        pl.addWidget(tg)

        # ── Image de fond (BG layer) ──
        bg_g = QGroupBox("Calque de fond")
        bg_l = QVBoxLayout(bg_g)
        self._bg_label = QLabel("Aucune image")
        self._bg_label.setWordWrap(True)
        bg_row = QHBoxLayout()
        btn_bg = QPushButton("Choisir…")
        btn_bg.clicked.connect(self._open_bg_image)
        self._btn_bg_clear = QPushButton("Effacer")
        self._btn_bg_clear.setEnabled(False)
        self._btn_bg_clear.clicked.connect(self._clear_bg_image)
        bg_row.addWidget(btn_bg)
        bg_row.addWidget(self._btn_bg_clear)
        bg_l.addWidget(self._bg_label)
        bg_l.addLayout(bg_row)
        bg_form = QFormLayout()
        self._row_bg_opacity = self._make_slider_row(
            bg_form, "Opacité (×0.01)", 0, 100, 100, self._on_bg_changed)
        self._chk_bg_zoom = QCheckBox("Zoom réactif bass")
        self._chk_bg_zoom.stateChanged.connect(self._on_bg_changed)
        bg_form.addRow("", self._chk_bg_zoom)
        self._row_bg_zoom_int = self._make_slider_row(
            bg_form, "Intensité (×0.01)", 1, 100, int(BG_PULSE_INTENSITY * 100),
            self._on_bg_changed)
        bg_l.addLayout(bg_form)
        pl.addWidget(bg_g)

        # ── Image centre (Halo) ──
        self._center_group = QGroupBox("Image centre (Halo)")
        cg_l = QVBoxLayout(self._center_group)
        self._center_label = QLabel("Aucune image")
        self._center_label.setWordWrap(True)
        cg_row = QHBoxLayout()
        btn_ctr = QPushButton("Choisir…")
        btn_ctr.clicked.connect(self._open_center_image)
        self._btn_ctr_clear = QPushButton("Effacer")
        self._btn_ctr_clear.setEnabled(False)
        self._btn_ctr_clear.clicked.connect(self._clear_center_image)
        cg_row.addWidget(btn_ctr)
        cg_row.addWidget(self._btn_ctr_clear)
        cg_l.addWidget(self._center_label)
        cg_l.addLayout(cg_row)
        pl.addWidget(self._center_group)

        # ── Visuel ──
        vg = QGroupBox("Visuel")
        vl = QFormLayout(vg)
        self._combo_viz = QComboBox()
        for name in VIZ_TYPES_VISIBLE:
            self._combo_viz.addItem(name)
        self._combo_viz.currentIndexChanged.connect(self._on_viz_changed)
        vl.addRow("Type", self._combo_viz)
        self._combo_palette = QComboBox()
        for name in PALETTES:
            self._combo_palette.addItem(name)
        self._combo_palette.addItem(_PERSO_AMP)
        self._combo_palette.addItem(_PERSO_FREQ)
        self._combo_palette.currentIndexChanged.connect(self._on_palette_changed)
        vl.addRow("Palette", self._combo_palette)
        pl.addWidget(vg)

        # ── Palette Perso. Amplitude ──
        self._custom_amp_group = QGroupBox("Palette amplitude")
        amp_l = QVBoxLayout(self._custom_amp_group)
        amp_l.setSpacing(3)
        self._amp_buttons: list[QPushButton] = []
        for i, lbl in enumerate(_AMP_LABELS):
            btn = QPushButton(lbl)
            btn.clicked.connect(lambda checked, idx=i: self._pick_amp_color(idx))
            amp_l.addWidget(btn)
            self._amp_buttons.append(btn)
        pl.addWidget(self._custom_amp_group)
        self._custom_amp_group.setVisible(False)
        self._refresh_pal_buttons(self._amp_buttons, self._custom_palette_amp)

        # ── Palette Perso. Fréquence ──
        self._custom_freq_group = QGroupBox("Palette fréquence")
        freq_l = QVBoxLayout(self._custom_freq_group)
        freq_l.setSpacing(3)
        self._freq_buttons: list[QPushButton] = []
        for i, lbl in enumerate(_FREQ_LABELS):
            btn = QPushButton(lbl)
            btn.clicked.connect(lambda checked, idx=i: self._pick_freq_color(idx))
            freq_l.addWidget(btn)
            self._freq_buttons.append(btn)
        pl.addWidget(self._custom_freq_group)
        self._custom_freq_group.setVisible(False)
        self._refresh_pal_buttons(self._freq_buttons, self._custom_palette_freq)

        # ── Paramètres visuels ──
        pg = QGroupBox("Paramètres")
        pf = QFormLayout(pg)
        self._row_bars = self._make_slider_row(
            pf, "Barres", 32, 256, NUM_BARS, self._on_params_changed)
        self._row_height = self._make_slider_row(
            pf, "Hauteur max (×0.01)", 30, 250, int(MAX_BAR_HEIGHT * 100),
            self._on_params_changed)
        self._row_smooth = self._make_slider_row(
            pf, "Smoothing (×0.01)", 50, 99, int(SMOOTHING_DECAY * 100),
            self._on_global_smoothing_changed)
        self._row_pulse = self._make_slider_row(
            pf, "Pulse (×0.01)", 0, 200, int(PULSE_INTENSITY * 100),
            self._on_params_changed)
        self._row_sensitivity = self._make_slider_row(
            pf, "Sensibilité (×0.01)", 25, 300, int(SENSITIVITY * 100),
            self._on_params_changed)
        self._row_rotation = self._make_slider_row(
            pf, "Rotation (°)", 0, 360, 0, self._on_params_changed)
        self._chk_mirror = QCheckBox("Miroir (½ cercle = spectre entier)")
        self._chk_mirror.stateChanged.connect(self._on_params_changed)
        pf.addRow("", self._chk_mirror)
        pl.addWidget(pg)

        # ── Répartition fréquentielle (global) ──
        fq = QGroupBox("Répartition fréquentielle (global)")
        ff = QFormLayout(fq)
        self._cqt_check = QCheckBox("Mode CQT")
        self._cqt_check.stateChanged.connect(self._on_freq_mode_changed)
        ff.addRow("", self._cqt_check)
        self._row_bass_split = self._make_slider_row(
            ff, "Basses (% barres)", 20, 80, int(FREQ_BASS_SPLIT * 100),
            self._on_freq_changed)
        self._row_bass_hz = self._make_slider_row(
            ff, "Coupure basses (Hz)", 80, 1500, int(FREQ_BASS_SPLIT_HZ),
            self._on_freq_changed)
        self._row_cqt_bpo = self._make_slider_row(
            ff, "Bins/octave", 6, 48, CQT_BINS_PER_OCTAVE, self._on_freq_changed)
        pl.addWidget(fq)
        self._update_freq_ui(cqt=False)

        # ── Halo Sine ──
        self._halo_sine_group = QGroupBox("Halo Sine")
        hsf = QFormLayout(self._halo_sine_group)
        self._row_hs_r_base = self._make_slider_row(
            hsf, "Rayon base (×0.01)", 5, 70, int(HALO_SINE_R_BASE * 100),
            self._on_params_changed)
        self._row_hs_amplitude = self._make_slider_row(
            hsf, "Amplitude (×0.01)", 1, 40, int(HALO_SINE_AMPLITUDE * 100),
            self._on_params_changed)
        self._row_hs_points = self._make_slider_row(
            hsf, "Points spline", 16, 256, HALO_SINE_N_POINTS, self._on_params_changed)
        self._row_hs_glow = self._make_slider_row(
            hsf, "Couches de glow", 1, 8, HALO_SINE_GLOW_LAYERS, self._on_params_changed)
        self._row_hs_decay = self._make_slider_row(
            hsf, "Lissage (×0.01)", 0, 99, int(HALO_SINE_SMOOTHING_DECAY * 100),
            self._on_params_changed)
        self._row_hs_fill = self._make_slider_row(
            hsf, "Remplissage (×0.01)", 0, 100, int(HALO_SINE_FILL_OPACITY * 100),
            self._on_params_changed)
        self._row_hs_gap = self._make_slider_row(
            hsf, "Distance sine (×0.01)", 10, 250, int(HALO_SINE_SPLINE_GAP * 100),
            self._on_params_changed)
        self._row_hs_pixel = self._make_slider_row(
            hsf, "Pixelisation", 1, 64, HALO_SINE_PIXEL_SIZE, self._on_params_changed)
        pl.addWidget(self._halo_sine_group)

        # ── Tunnel Arcade ──
        self._tunnel_group = QGroupBox("Tunnel Arcade")
        tg_f = QFormLayout(self._tunnel_group)
        self._combo_tunnel_sides = QComboBox()
        for s in ["4", "6", "8", "12"]:
            self._combo_tunnel_sides.addItem(s)
        self._combo_tunnel_sides.setCurrentText(str(TUNNEL_SIDES))
        self._combo_tunnel_sides.currentIndexChanged.connect(self._on_params_changed)
        tg_f.addRow("Côtés", self._combo_tunnel_sides)
        self._combo_tunnel_kick_mode = QComboBox()
        self._combo_tunnel_kick_mode.addItems([
            "Delta", "Seuil adaptatif", "Kick spectral", "Fréquence seuil"])
        self._combo_tunnel_kick_mode.setCurrentIndex(TUNNEL_KICK_MODE)
        self._combo_tunnel_kick_mode.currentIndexChanged.connect(self._on_tunnel_kick_mode_changed)
        tg_f.addRow("Détection", self._combo_tunnel_kick_mode)
        self._row_tunnel_kick_freq_lo = self._make_slider_row(
            tg_f, "Bande lo (%)", 0, 49, TUNNEL_KICK_FREQ_LO, self._on_params_changed)
        self._row_tunnel_kick_freq_hi = self._make_slider_row(
            tg_f, "Bande hi (%)", 1, 50, TUNNEL_KICK_FREQ_HI, self._on_params_changed)
        self._row_tunnel_kick_threshold = self._make_slider_row(
            tg_f, "Seuil (×0.01)", 10, 800, TUNNEL_KICK_THRESHOLD, self._on_params_changed)
        self._row_tunnel_kick_cooldown = self._make_slider_row(
            tg_f, "Cooldown (frames)", 5, 120, TUNNEL_KICK_COOLDOWN, self._on_params_changed)
        for _r in (self._row_tunnel_kick_freq_lo, self._row_tunnel_kick_freq_hi,
                   self._row_tunnel_kick_threshold, self._row_tunnel_kick_cooldown):
            self._set_row_visible(_r, False)
        self._row_tunnel_rings = self._make_slider_row(
            tg_f, "Anneaux", 1, 16, TUNNEL_RINGS, self._on_params_changed)
        self._row_tunnel_speed = self._make_slider_row(
            tg_f, "Vitesse (×0.01)", 10, 400, int(TUNNEL_SPEED * 100), self._on_params_changed)
        self._row_tunnel_kick_zoom = self._make_slider_row(
            tg_f, "Kick zoom (×0.01)", 0, 300, int(TUNNEL_KICK_ZOOM * 100), self._on_params_changed)
        self._row_tunnel_chroma = self._make_slider_row(
            tg_f, "Aberration (×0.01)", 0, 300, int(TUNNEL_CHROMA * 100), self._on_params_changed)
        self._row_tunnel_kick_sens = self._make_slider_row(
            tg_f, "Kick sens. (×0.01)", 10, 500, int(TUNNEL_KICK_SENSITIVITY * 100), self._on_params_changed)
        self._row_tunnel_bass_speed = self._make_slider_row(
            tg_f, "Vitesse audio (×0.01)", 0, 1000, int(TUNNEL_BASS_SPEED * 100), self._on_params_changed)
        pl.addWidget(self._tunnel_group)

        # ── Nuclear Shockwave (mode 10) ──
        self._nuke_group = QGroupBox("Nuclear Shockwave")
        nk_f = QFormLayout(self._nuke_group)
        self._row_nuke_thresh = self._make_slider_row(
            nk_f, "Seuil kick (×0.001)", 5, 150, NUKE_KICK_THRESHOLD, self._on_params_changed)
        self._row_nuke_speed = self._make_slider_row(
            nk_f, "Vitesse onde (×0.01)", 20, 300, int(NUKE_SPEED * 100), self._on_params_changed)
        self._row_nuke_life = self._make_slider_row(
            nk_f, "Durée vie (×0.1 s)", 5, 60, int(NUKE_LIFE * 10), self._on_params_changed)
        self._row_nuke_width = self._make_slider_row(
            nk_f, "Épaisseur (×0.01)", 30, 400, int(NUKE_WIDTH * 100), self._on_params_changed)
        self._row_nuke_flash = self._make_slider_row(
            nk_f, "Flash (×0.01)", 0, 100, int(NUKE_FLASH * 100), self._on_params_changed)
        self._row_nuke_bg = self._make_slider_row(
            nk_f, "Fond (×0.01)", 0, 300, int(NUKE_BG * 100), self._on_params_changed)
        pl.addWidget(self._nuke_group)

        # ── Void Pull (mode 11) ──
        self._void_group = QGroupBox("Void Pull")
        vd_f = QFormLayout(self._void_group)
        self._row_void_speed = self._make_slider_row(
            vd_f, "Vitesse rotation (×0.01)", 0, 100, int(VOID_SPEED * 100), self._on_params_changed)
        self._row_void_pull = self._make_slider_row(
            vd_f, "Distorsion kick/bass (×0.01)", 0, 150, int(VOID_PULL * 100), self._on_params_changed)
        self._row_void_rays = self._make_slider_row(
            vd_f, "Rayons aigus (×0.01)", 0, 600, int(VOID_RAYS * 100), self._on_params_changed)
        self._row_void_arms = self._make_slider_row(
            vd_f, "Bras spirale", 1, 8, VOID_ARMS, self._on_params_changed)
        pl.addWidget(self._void_group)

        # ── Effets beats (par calque) ──
        beats_g = QGroupBox("Effets beats")
        beats_f = QFormLayout(beats_g)
        self._chk_flash = QCheckBox("Flash")
        self._chk_flash.stateChanged.connect(self._on_params_changed)
        beats_f.addRow("", self._chk_flash)
        self._row_flash_intensity = self._make_slider_row(
            beats_f, "Intensité (×0.01)", 1, 100, int(FLASH_INTENSITY * 100),
            self._on_params_changed)
        self._chk_flash.stateChanged.connect(
            lambda s: self._set_row_visible(self._row_flash_intensity, bool(s)))
        self._set_row_visible(self._row_flash_intensity, False)
        pl.addWidget(beats_g)

        # ── Export ──
        eg = QGroupBox("Export")
        ef = QFormLayout(eg)
        self._combo_res = QComboBox()
        for r in RESOLUTIONS:
            self._combo_res.addItem(r)
        self._combo_res.setCurrentText("1080p")
        ef.addRow("Résolution", self._combo_res)
        self._combo_fps = QComboBox()
        self._combo_fps.addItems(["30", "60"])
        self._combo_fps.setCurrentText("60")
        ef.addRow("FPS", self._combo_fps)
        pl.addWidget(eg)

        self._btn_play = QPushButton("▶ Preview")
        self._btn_play.setEnabled(False)
        self._btn_play.clicked.connect(self._toggle_playback)
        pl.addWidget(self._btn_play)

        # ── Transport ──
        self._seek_slider = QSlider(Qt.Horizontal)
        self._seek_slider.setRange(0, 1000)
        self._seek_slider.setValue(0)
        self._seek_slider.setEnabled(False)
        self._seek_slider.sliderPressed.connect(self._on_seek_pressed)
        self._seek_slider.sliderReleased.connect(self._on_seek_released)
        pl.addWidget(self._seek_slider)

        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("Volume"))
        self._vol_slider = QSlider(Qt.Horizontal)
        self._vol_slider.setRange(0, 100)
        self._vol_slider.setValue(100)
        self._vol_slider.valueChanged.connect(self._on_volume_changed)
        # Restarting playback is what actually applies a new volume, and it tears down
        # the audio stream and the FFT. Do it once the slider is let go, never on every
        # value emitted during a drag.
        self._vol_slider.sliderReleased.connect(self._on_volume_released)
        self._vol_label = QLabel("100%")
        self._vol_label.setFixedWidth(36)
        vol_row.addWidget(self._vol_slider)
        vol_row.addWidget(self._vol_label)
        pl.addLayout(vol_row)

        self._btn_export = QPushButton("⬇ Exporter MP4")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._start_export)
        pl.addWidget(self._btn_export)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        pl.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        pl.addWidget(self._status)

        self._preview = PreviewWidget()
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._preview.layer_moved.connect(self._on_layer_moved)
        root.addWidget(self._preview)

    # ── Layer list management ────────────────────────────────────
    def _layer_item_label(self, layer: Layer) -> str:
        vis  = "👁" if layer.visible else "🚫"
        solo = " ⓢ" if layer.solo else ""
        return f"{vis} {layer.name}{solo}"

    def _refresh_layer_list(self):
        self._loading = True
        self._layer_list.clear()
        # Display topmost layer first (render order is bottom→top)
        for layer in reversed(self._lm.layers):
            item = QListWidgetItem(self._layer_item_label(layer))
            item.setData(Qt.UserRole, layer.id)
            self._layer_list.addItem(item)
        # Sync selection
        sel = self._lm.selected_id
        for row in range(self._layer_list.count()):
            if self._layer_list.item(row).data(Qt.UserRole) == sel:
                self._layer_list.setCurrentRow(row)
                break
        self._loading = False

    def _refresh_selected_item_label(self):
        layer = self._lm.selected()
        if layer is None:
            return
        row = self._layer_list.currentRow()
        if row >= 0:
            self._layer_list.item(row).setText(self._layer_item_label(layer))

    def _on_layer_selected(self, row: int):
        if self._loading or row < 0:
            return
        item = self._layer_list.item(row)
        if item is None:
            return
        self._lm.selected_id = item.data(Qt.UserRole)
        self._load_layer_into_ui(self._lm.selected())

    def _on_layers_reordered(self, *args):
        if self._loading:
            return
        # Rebuild render order from list (top item = topmost = last in layers)
        ids_top_to_bottom = [self._layer_list.item(r).data(Qt.UserRole)
                             for r in range(self._layer_list.count())]
        id_to_layer = {l.id: l for l in self._lm.layers}
        self._lm.layers = [id_to_layer[i] for i in reversed(ids_top_to_bottom)
                           if i in id_to_layer]

    def _add_layer(self):
        if len(self._lm.layers) >= MAX_VISUAL_LAYERS:
            self._status.setText(f"Maximum {MAX_VISUAL_LAYERS} calques.")
            return
        self._lm.add_layer(mode=DEFAULT_VIZ_TYPE)
        self._refresh_layer_list()
        self._load_layer_into_ui(self._lm.selected())

    def _remove_selected_layer(self):
        layer = self._lm.selected()
        if layer is None:
            return
        self._preview.release_layer_state(layer.id)
        self._lm.remove_layer(layer.id)
        self._refresh_layer_list()
        if self._lm.selected():
            self._load_layer_into_ui(self._lm.selected())

    def _toggle_selected_visible(self):
        layer = self._lm.selected()
        if layer:
            layer.visible = not layer.visible
            self._refresh_selected_item_label()

    def _toggle_selected_solo(self):
        layer = self._lm.selected()
        if layer:
            layer.solo = not layer.solo
            self._refresh_selected_item_label()

    def _on_layer_moved(self):
        # Drag in preview updated x/y; nothing else to refresh for now
        pass

    # ── Load / write selected layer ──────────────────────────────
    def _load_layer_into_ui(self, layer: Layer | None):
        if layer is None:
            return
        self._loading = True

        # transform
        self._sl(self._row_scale).setValue(int(layer.scale * 100))
        self._sl(self._row_opacity).setValue(int(layer.opacity * 100))
        self._combo_blend.setCurrentIndex(0 if layer.blend_mode == BLEND_ADDITIVE else 1)

        # viz + palette
        self._combo_viz.setCurrentText(layer.mode_name())
        self._custom_palette_amp  = [list(c) for c in layer.palette] if layer.palette_name == _PERSO_AMP else self._custom_palette_amp
        self._custom_palette_freq = [list(c) for c in layer.palette] if layer.palette_name == _PERSO_FREQ else self._custom_palette_freq
        self._refresh_pal_buttons(self._amp_buttons, self._custom_palette_amp)
        self._refresh_pal_buttons(self._freq_buttons, self._custom_palette_freq)
        idx = self._combo_palette.findText(layer.palette_name)
        self._combo_palette.setCurrentIndex(idx if idx >= 0 else 0)

        # generic
        self._sl(self._row_bars).setValue(layer.num_bars)
        self._sl(self._row_height).setValue(int(layer.max_bar_height * 100))
        self._sl(self._row_pulse).setValue(int(layer.pulse_intensity * 100))
        self._sl(self._row_sensitivity).setValue(int(layer.sensitivity * 100))
        self._sl(self._row_rotation).setValue(int(math.degrees(layer.rotation)))
        self._chk_mirror.setChecked(layer.mirror)

        # halo
        self._sl(self._row_hs_r_base).setValue(int(layer.halo_r_base * 100))
        self._sl(self._row_hs_amplitude).setValue(int(layer.halo_amplitude * 100))
        self._sl(self._row_hs_points).setValue(layer.halo_n_points)
        self._sl(self._row_hs_glow).setValue(layer.halo_glow_layers)
        self._sl(self._row_hs_decay).setValue(int(layer.halo_smoothing_decay * 100))
        self._sl(self._row_hs_fill).setValue(int(layer.halo_fill_opacity * 100))
        self._sl(self._row_hs_gap).setValue(int(layer.halo_spline_gap * 100))
        self._sl(self._row_hs_pixel).setValue(layer.halo_pixel_size)

        # tunnel
        self._combo_tunnel_sides.setCurrentText(str(layer.tunnel_sides))
        self._combo_tunnel_kick_mode.setCurrentIndex(layer.tunnel_kick_mode)
        self._sl(self._row_tunnel_kick_freq_lo).setValue(layer.tunnel_kick_freq_lo)
        self._sl(self._row_tunnel_kick_freq_hi).setValue(layer.tunnel_kick_freq_hi)
        self._sl(self._row_tunnel_kick_threshold).setValue(int(layer.tunnel_kick_threshold * 100))
        self._sl(self._row_tunnel_kick_cooldown).setValue(layer.tunnel_kick_cooldown)
        self._sl(self._row_tunnel_rings).setValue(layer.tunnel_rings)
        self._sl(self._row_tunnel_speed).setValue(int(layer.tunnel_speed * 100))
        self._sl(self._row_tunnel_kick_zoom).setValue(int(layer.tunnel_kick_zoom * 100))
        self._sl(self._row_tunnel_chroma).setValue(int(layer.tunnel_chroma * 100))
        self._sl(self._row_tunnel_kick_sens).setValue(int(layer.tunnel_kick_sensitivity * 100))
        self._sl(self._row_tunnel_bass_speed).setValue(int(layer.tunnel_bass_speed * 100))

        # nuclear shockwave
        self._sl(self._row_nuke_thresh).setValue(int(round(layer.nuke_kick_threshold * 1000)))
        self._sl(self._row_nuke_speed).setValue(int(layer.nuke_speed * 100))
        self._sl(self._row_nuke_life).setValue(int(layer.nuke_life * 10))
        self._sl(self._row_nuke_width).setValue(int(layer.nuke_width * 100))
        self._sl(self._row_nuke_flash).setValue(int(layer.nuke_flash * 100))
        self._sl(self._row_nuke_bg).setValue(int(layer.nuke_bg * 100))

        # void pull
        self._sl(self._row_void_speed).setValue(int(layer.void_speed * 100))
        self._sl(self._row_void_pull).setValue(int(layer.void_pull * 100))
        self._sl(self._row_void_rays).setValue(int(layer.void_rays * 100))
        self._sl(self._row_void_arms).setValue(layer.void_arms)

        # flash
        self._chk_flash.setChecked(layer.flash)
        self._sl(self._row_flash_intensity).setValue(int(layer.flash_intensity * 100))

        self._loading = False
        self._update_ui_for_viz(layer.mode)
        self._update_tunnel_kick_visibility()
        self._set_row_visible(self._row_flash_intensity, layer.flash)

    def _write_ui_to_layer(self, layer: Layer):
        layer.scale = self._sl(self._row_scale).value() / 100
        layer.opacity = self._sl(self._row_opacity).value() / 100
        layer.blend_mode = BLEND_ADDITIVE if self._combo_blend.currentIndex() == 0 else BLEND_NORMAL
        layer.mode = VIZ_TYPES[self._combo_viz.currentText()]
        layer.name = layer.mode_name()
        layer.palette = self._current_palette()
        layer.palette_name = self._combo_palette.currentText()
        layer.pal_mode = self._current_pal_mode()
        layer.num_bars = self._sl(self._row_bars).value()
        layer.max_bar_height = self._sl(self._row_height).value() / 100
        layer.pulse_intensity = self._sl(self._row_pulse).value() / 100
        layer.sensitivity = self._sl(self._row_sensitivity).value() / 100
        layer.rotation = math.radians(self._sl(self._row_rotation).value())
        layer.mirror = self._chk_mirror.isChecked()
        layer.halo_r_base = self._sl(self._row_hs_r_base).value() / 100
        layer.halo_amplitude = self._sl(self._row_hs_amplitude).value() / 100
        layer.halo_n_points = self._sl(self._row_hs_points).value()
        layer.halo_glow_layers = self._sl(self._row_hs_glow).value()
        layer.halo_smoothing_decay = self._sl(self._row_hs_decay).value() / 100
        layer.halo_fill_opacity = self._sl(self._row_hs_fill).value() / 100
        layer.halo_spline_gap = self._sl(self._row_hs_gap).value() / 100
        layer.halo_pixel_size = self._sl(self._row_hs_pixel).value()
        layer.tunnel_sides = int(self._combo_tunnel_sides.currentText())
        layer.tunnel_kick_mode = self._combo_tunnel_kick_mode.currentIndex()
        layer.tunnel_kick_freq_lo = self._sl(self._row_tunnel_kick_freq_lo).value()
        layer.tunnel_kick_freq_hi = self._sl(self._row_tunnel_kick_freq_hi).value()
        layer.tunnel_kick_threshold = self._sl(self._row_tunnel_kick_threshold).value() / 100.0
        layer.tunnel_kick_cooldown = self._sl(self._row_tunnel_kick_cooldown).value()
        layer.tunnel_rings = self._sl(self._row_tunnel_rings).value()
        layer.tunnel_speed = self._sl(self._row_tunnel_speed).value() / 100
        layer.tunnel_kick_zoom = self._sl(self._row_tunnel_kick_zoom).value() / 100
        layer.tunnel_chroma = self._sl(self._row_tunnel_chroma).value() / 100
        layer.tunnel_kick_sensitivity = self._sl(self._row_tunnel_kick_sens).value() / 100
        layer.tunnel_bass_speed = self._sl(self._row_tunnel_bass_speed).value() / 100
        layer.nuke_kick_threshold = self._sl(self._row_nuke_thresh).value() / 1000.0
        layer.nuke_speed = self._sl(self._row_nuke_speed).value() / 100
        layer.nuke_life = self._sl(self._row_nuke_life).value() / 10
        layer.nuke_width = self._sl(self._row_nuke_width).value() / 100
        layer.nuke_flash = self._sl(self._row_nuke_flash).value() / 100
        layer.nuke_bg = self._sl(self._row_nuke_bg).value() / 100
        layer.void_speed = self._sl(self._row_void_speed).value() / 100
        layer.void_pull = self._sl(self._row_void_pull).value() / 100
        layer.void_rays = self._sl(self._row_void_rays).value() / 100
        layer.void_arms = self._sl(self._row_void_arms).value()
        layer.flash = self._chk_flash.isChecked()
        layer.flash_intensity = self._sl(self._row_flash_intensity).value() / 100

    # ── Visibility helpers ───────────────────────────────────────
    def _update_freq_ui(self, cqt: bool):
        self._set_row_visible(self._row_bass_split, not cqt)
        self._set_row_visible(self._row_bass_hz,    not cqt)
        self._set_row_visible(self._row_cqt_bpo,    cqt)

    def _update_tunnel_kick_visibility(self):
        advanced = self._combo_tunnel_kick_mode.currentIndex() in (2, 3)
        self._set_row_visible(self._row_tunnel_kick_freq_lo,   advanced)
        self._set_row_visible(self._row_tunnel_kick_freq_hi,   advanced)
        self._set_row_visible(self._row_tunnel_kick_threshold, advanced)
        self._set_row_visible(self._row_tunnel_kick_cooldown,  advanced)

    def _update_ui_for_viz(self, viz_type: int):
        is_radial    = viz_type in _RADIAL_MODES
        has_rotation = viz_type in _ROTATION_MODES
        is_bar       = viz_type in _BAR_MODES
        has_height   = viz_type in _HEIGHT_MODES
        is_halo      = viz_type in _HALO_MODES
        is_halo_sine = viz_type in _HALO_SINE_MODES
        is_tunnel    = viz_type in _TUNNEL_MODES
        has_mirror   = viz_type in _MIRROR_MODES

        self._set_row_visible(self._row_bars,      is_bar)
        self._set_row_visible(self._row_height,    has_height)
        self._set_row_visible(self._row_pulse,     is_radial)
        self._set_row_visible(self._row_rotation,  has_rotation)
        self._center_group.setVisible(is_halo)
        self._halo_sine_group.setVisible(is_halo_sine)
        is_halo_circle = viz_type == 6
        self._set_row_visible(self._row_hs_r_base, is_halo_circle)
        self._set_row_visible(self._row_hs_gap,    is_halo_circle)
        self._set_row_visible(self._row_hs_pixel,  is_halo_circle)
        self._tunnel_group.setVisible(is_tunnel)
        self._nuke_group.setVisible(viz_type in _NUKE_MODES)
        self._void_group.setVisible(viz_type in _VOID_MODES)
        self._chk_mirror.setVisible(has_mirror)
        pal_name = self._combo_palette.currentText()
        self._custom_amp_group.setVisible(pal_name == _PERSO_AMP)
        self._custom_freq_group.setVisible(pal_name == _PERSO_FREQ)

    # ── Accessors ────────────────────────────────────────────────
    def _sl(self, row): return row[0]

    def _current_palette(self):
        name = self._combo_palette.currentText()
        if name == _PERSO_AMP:
            return [list(c) for c in self._custom_palette_amp]
        if name == _PERSO_FREQ:
            return [list(c) for c in self._custom_palette_freq]
        return [list(c) for c in PALETTES[name]]

    def _current_pal_mode(self) -> int:
        return 1 if self._combo_palette.currentText() == _PERSO_FREQ else 0

    def _refresh_pal_buttons(self, buttons: list, palette: list):
        for i, c in enumerate(palette):
            r, g, b = c[0], c[1], c[2]
            hex_col = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
            lum     = 0.299*r + 0.587*g + 0.114*b
            txt_col = "#000000" if lum > 0.5 else "#ffffff"
            buttons[i].setStyleSheet(
                f"background-color:{hex_col};color:{txt_col};border:none;padding:4px;")

    def _pick_amp_color(self, idx: int):
        c = self._custom_palette_amp[idx]
        initial = QColor(int(c[0]*255), int(c[1]*255), int(c[2]*255))
        col = QColorDialog.getColor(initial, self, f"Amplitude — stop {idx + 1}")
        if col.isValid():
            self._custom_palette_amp[idx] = [col.redF(), col.greenF(), col.blueF()]
            self._refresh_pal_buttons(self._amp_buttons, self._custom_palette_amp)
            self._on_params_changed()

    def _pick_freq_color(self, idx: int):
        c = self._custom_palette_freq[idx]
        initial = QColor(int(c[0]*255), int(c[1]*255), int(c[2]*255))
        col = QColorDialog.getColor(initial, self, f"Fréquence — stop {idx + 1}")
        if col.isValid():
            self._custom_palette_freq[idx] = [col.redF(), col.greenF(), col.blueF()]
            self._refresh_pal_buttons(self._freq_buttons, self._custom_palette_freq)
            self._on_params_changed()

    def _on_palette_changed(self):
        pal_name = self._combo_palette.currentText()
        self._custom_amp_group.setVisible(pal_name == _PERSO_AMP)
        self._custom_freq_group.setVisible(pal_name == _PERSO_FREQ)
        self._on_params_changed()

    def _use_cqt(self) -> bool:
        return self._cqt_check.isChecked()

    def _make_fft(self) -> FFTProcessor:
        return FFTProcessor(
            sr=self._audio.sr,
            num_bars=FFT_ANALYSIS_BARS,
            smoothing_decay=self._sl(self._row_smooth).value() / 100,
            bass_split=self._sl(self._row_bass_split).value() / 100,
            bass_split_hz=float(self._sl(self._row_bass_hz).value()),
            use_cqt=self._use_cqt(),
            bins_per_octave=self._sl(self._row_cqt_bpo).value(),
        )

    # ── Slots ────────────────────────────────────────────────────
    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir audio", "", "Audio (*.wav *.flac *.mp3 *.ogg *.aiff)")
        if not path:
            return
        self._audio = AudioFile(path)
        self._fft   = self._make_fft()
        self._file_label.setText(Path(path).name)
        self._btn_play.setEnabled(True)
        self._btn_export.setEnabled(True)
        self._seek_frame = 0
        self._seek_slider.setValue(0)
        self._seek_slider.setEnabled(True)
        self._status.setText(f"Durée : {self._audio.duration:.1f}s · {self._audio.sr} Hz")

    def _open_bg_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Image de fond", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)")
        if not path:
            return
        self._bg_image_path = path
        self._lm.bg.image_path = path
        self._bg_label.setText(Path(path).name)
        self._btn_bg_clear.setEnabled(True)
        self._preview.load_background(path)

    def _clear_bg_image(self):
        self._bg_image_path = None
        self._lm.bg.image_path = None
        self._bg_label.setText("Aucune image")
        self._btn_bg_clear.setEnabled(False)
        self._preview.clear_background()

    def _on_bg_changed(self):
        if self._loading:
            return
        self._lm.bg.opacity = self._sl(self._row_bg_opacity).value() / 100
        self._lm.bg.zoom_reactive = self._chk_bg_zoom.isChecked()
        self._lm.bg.zoom_intensity = self._sl(self._row_bg_zoom_int).value() / 100

    def _open_center_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Image centre", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)")
        if not path:
            return
        self._center_image_path = path
        self._center_label.setText(Path(path).name)
        self._btn_ctr_clear.setEnabled(True)
        self._preview.load_center_image(path)

    def _clear_center_image(self):
        self._center_image_path = None
        self._center_label.setText("Aucune image")
        self._btn_ctr_clear.setEnabled(False)
        self._preview.clear_center_image()

    def _on_viz_changed(self):
        if self._loading:
            return
        self._update_ui_for_viz(VIZ_TYPES[self._combo_viz.currentText()])
        self._on_params_changed()
        self._refresh_selected_item_label()

    def _on_params_changed(self):
        if self._loading:
            return
        layer = self._lm.selected()
        if layer is not None:
            self._write_ui_to_layer(layer)

    def _on_global_smoothing_changed(self):
        if self._loading:
            return
        if self._fft:
            self._fft.smoothing_decay = self._sl(self._row_smooth).value() / 100

    def _on_freq_mode_changed(self):
        self._update_freq_ui(cqt=self._use_cqt())
        self._on_freq_changed()

    def _on_freq_changed(self):
        if self._loading:
            return
        if self._audio:
            new_fft = self._make_fft()
            if self._playback_thread and self._playback_thread.isRunning():
                self._playback_thread.update_fft(new_fft)
            self._fft = new_fft

    def _on_tunnel_kick_mode_changed(self):
        self._update_tunnel_kick_visibility()
        self._on_params_changed()

    def _on_volume_changed(self, val: int):
        self._volume = val / 100.0
        self._vol_label.setText(f"{val}%")
        # No restart here: a drag emits dozens of these per second, and each restart
        # stops and recreates the playback thread and the shared audio stream.
        if not self._vol_slider.isSliderDown():
            self._on_volume_released()

    def _on_volume_released(self):
        if self._playback_thread and self._playback_thread.isRunning():
            self._restart_playback_from(self._seek_frame)

    def _on_seek_pressed(self):
        self._seeking = True

    def _on_seek_released(self):
        if not self._audio:
            self._seeking = False
            return
        hop   = max(1, int(self._audio.sr / int(self._combo_fps.currentText())))
        total = int(len(self._audio.mono) / hop)
        self._seek_frame = int(self._seek_slider.value() / 1000 * total)
        self._seeking = False
        if self._playback_thread and self._playback_thread.isRunning():
            self._restart_playback_from(self._seek_frame)

    def _on_position_changed(self, frame: int, total: int):
        if not self._seeking:
            self._seek_slider.setValue(int(frame / max(total, 1) * 1000))
        self._seek_frame = frame

    def _restart_playback_from(self, frame: int):
        if self._playback_thread:
            self._playback_thread.stop()
            # Bounded wait: the GUI thread must never be able to hang here, whatever
            # state the audio stream is in.
            self._playback_thread.wait(_THREAD_STOP_TIMEOUT_MS)
        fps = int(self._combo_fps.currentText())
        self._fft = self._make_fft()
        self._playback_thread = AudioPlaybackThread(
            self._audio, self._fft, fps, start_frame=frame, volume=self._volume)
        self._playback_thread.frame_ready.connect(self._preview.update_audio_data)
        self._playback_thread.position_changed.connect(self._on_position_changed)
        self._playback_thread.finished.connect(self._on_playback_done)
        self._playback_thread.start()

    def _toggle_playback(self):
        if self._playback_thread and self._playback_thread.isRunning():
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if not self._audio:
            return
        self._fft = self._make_fft()
        fps = int(self._combo_fps.currentText())
        self._playback_thread = AudioPlaybackThread(
            self._audio, self._fft, fps, start_frame=self._seek_frame, volume=self._volume)
        self._playback_thread.frame_ready.connect(self._preview.update_audio_data)
        self._playback_thread.position_changed.connect(self._on_position_changed)
        self._playback_thread.finished.connect(self._on_playback_done)
        self._playback_thread.start()
        self._btn_play.setText("⏹ Stop")

    def _stop_playback(self):
        if self._playback_thread:
            self._playback_thread.stop()
            self._playback_thread.wait(_THREAD_STOP_TIMEOUT_MS)
        self._btn_play.setText("▶ Preview")

    def _on_playback_done(self):
        self._btn_play.setText("▶ Preview")
        self._seek_frame = 0
        self._seek_slider.setValue(0)

    def _start_export(self):
        if not self._audio:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Dossier de sortie")
        if not out_dir:
            return

        self._stop_playback()
        self._btn_export.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.setText("Export en cours…")

        worker = ExportWorker(
            audio=self._audio,
            layer_manager=copy.deepcopy(self._lm),
            output_dir=out_dir,
            resolution=self._combo_res.currentText(),
            fps=int(self._combo_fps.currentText()),
            smoothing_decay=self._sl(self._row_smooth).value() / 100,
            bass_split=self._sl(self._row_bass_split).value() / 100,
            bass_split_hz=float(self._sl(self._row_bass_hz).value()),
            use_cqt=self._use_cqt(),
            bins_per_octave=self._sl(self._row_cqt_bpo).value(),
            bg_image_path=self._bg_image_path,
            center_image_path=self._center_image_path,
        )
        self._export_thread = QThread()
        worker.moveToThread(self._export_thread)
        self._export_thread.started.connect(worker.run)
        worker.progress.connect(self._on_export_progress)
        worker.finished.connect(self._on_export_done)
        worker.error.connect(self._on_export_error)
        worker.finished.connect(self._export_thread.quit)
        worker.error.connect(self._export_thread.quit)
        self._export_worker = worker
        self._export_thread.start()

    def _on_export_progress(self, cur, total):
        self._progress.setMaximum(total)
        self._progress.setValue(cur)

    def _on_export_done(self, path):
        self._progress.setVisible(False)
        self._btn_export.setEnabled(True)
        self._status.setText(f"Export terminé :\n{path}")

    def _on_export_error(self, msg):
        self._progress.setVisible(False)
        self._btn_export.setEnabled(True)
        self._status.setText(f"Erreur export :\n{msg}")

    def closeEvent(self, event):
        self._stop_playback()
        super().closeEvent(event)
