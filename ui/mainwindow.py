import math
import numpy as np
import sounddevice as sd
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QComboBox,
    QProgressBar, QGroupBox, QFormLayout, QSizePolicy, QSpinBox, QCheckBox,
    QColorDialog,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QColor

from core.audio import AudioFile
from core.fft import FFTProcessor
from ui.preview import PreviewWidget
from config.defaults import (
    NUM_BARS, MAX_BAR_HEIGHT, SMOOTHING_DECAY, PULSE_INTENSITY,
    SENSITIVITY, FPS, RESOLUTIONS, FFT_SIZE, PALETTES, VIZ_TYPES,
    FREQ_BASS_SPLIT, FREQ_BASS_SPLIT_HZ, CQT_BINS_PER_OCTAVE,
    HALO_SINE_R_BASE, HALO_SINE_AMPLITUDE, HALO_SINE_N_POINTS,
    HALO_SINE_GLOW_LAYERS, HALO_SINE_SMOOTHING_DECAY, HALO_SINE_FILL_OPACITY,
    BG_PULSE_INTENSITY, FLASH_INTENSITY,
)

_RADIAL_MODES     = {0, 4, 5, 6, 7, 8}
_ROTATION_MODES   = {0, 4, 5, 6, 7, 8}
_BAR_MODES        = {0, 1, 2, 4, 5, 7, 8}
_HEIGHT_MODES     = {0, 1, 2, 4, 5, 7, 8}
_HALO_MODES       = {4, 5, 6, 7, 8}
_HALO_SINE_MODES  = {6}

_AMP_LABELS  = ["Silence", "Faible", "Moyen", "Fort", "Saturation"]
_FREQ_LABELS = ["Grave", "Basse", "Médium", "Présence", "Aigu"]


class ExportWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(str)
    error    = Signal(str)

    def __init__(self, audio, output_dir, resolution, fps,
                 num_bars, max_bar_height, smoothing_decay, pulse_intensity,
                 palette, sensitivity, viz_type, bg_image_path,
                 center_image_path, rotation,
                 bass_split, bass_split_hz,
                 use_cqt, bins_per_octave,
                 halo_r_base, halo_amplitude, halo_n_points, halo_glow_layers,
                 halo_smoothing_decay, halo_fill_opacity, pal_mode: int = 0,
                 bg_pulse: bool = False, bg_pulse_intensity: float = 0.5,
                 flash: bool = False, flash_intensity: float = 0.5):
        super().__init__()
        self._audio = audio
        self._kwargs = dict(
            output_dir=output_dir, resolution=resolution, fps=fps,
            num_bars=num_bars, max_bar_height=max_bar_height,
            smoothing_decay=smoothing_decay, pulse_intensity=pulse_intensity,
            palette=palette, sensitivity=sensitivity, viz_type=viz_type,
            bg_image_path=bg_image_path,
            center_image_path=center_image_path,
            rotation=rotation,
            bass_split=bass_split,
            bass_split_hz=bass_split_hz,
            use_cqt=use_cqt,
            bins_per_octave=bins_per_octave,
            halo_r_base=halo_r_base,
            halo_amplitude=halo_amplitude,
            halo_n_points=halo_n_points,
            halo_glow_layers=halo_glow_layers,
            halo_smoothing_decay=halo_smoothing_decay,
            halo_fill_opacity=halo_fill_opacity,
            pal_mode=pal_mode,
            bg_pulse=bg_pulse,
            bg_pulse_intensity=bg_pulse_intensity,
            flash=flash,
            flash_intensity=flash_intensity,
        )

    def run(self):
        try:
            from export.ffmpeg_export import FFmpegExporter
            exporter = FFmpegExporter(
                audio=self._audio,
                progress_cb=lambda cur, tot: self.progress.emit(cur, tot),
                **self._kwargs,
            )
            out = exporter.export()
            self.finished.emit(out)
        except Exception as e:
            self.error.emit(str(e))


class AudioPlaybackThread(QThread):
    frame_ready = Signal(object, float)

    def __init__(self, audio: AudioFile, fft: FFTProcessor, fps: int):
        super().__init__()
        self._audio = audio
        self._fft   = fft
        self._fps   = fps
        self._running = False

    def update_fft(self, fft: FFTProcessor):
        self._fft = fft

    def run(self):
        self._running = True
        hop   = int(self._audio.sr / self._fps)
        total = int(len(self._audio.mono) / hop)
        sd.play(self._audio.mono, samplerate=self._audio.sr)
        for i in range(total):
            if not self._running:
                break
            samples = self._audio.get_frame_samples(i, hop, FFT_SIZE)
            bars, pulse = self._fft.process(samples)
            self.frame_ready.emit(bars, pulse)
            self.msleep(int(1000 / self._fps))
        sd.stop()

    def stop(self):
        self._running = False
        sd.stop()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Visualizer")
        self.resize(1300, 800)

        self._audio: AudioFile | None = None
        self._fft:   FFTProcessor | None = None
        self._playback_thread: AudioPlaybackThread | None = None
        self._export_thread:   QThread | None = None
        self._bg_image_path:     str | None = None
        self._center_image_path: str | None = None
        _default = [list(c) for c in list(PALETTES.values())[0]]
        self._custom_palette_amp  = [list(c) for c in _default]
        self._custom_palette_freq = [list(c) for c in _default]

        self._build_ui()
        self._update_ui_for_viz(0)

    # ── constructeurs d'UI ───────────────────────────────────────
    def _make_slider_row(self, layout: QFormLayout, label: str,
                         lo: int, hi: int, val: int,
                         callback=None) -> tuple:
        """Retourne (QSlider, lbl_widget, container_widget) pour show/hide."""
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
        sb.setMinimumWidth(72)
        sb.setFixedWidth(72)

        # connexion bidirectionnelle sans boucle infinie (Qt vérifie si la valeur change)
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

        panel = QWidget()
        panel.setFixedWidth(330)
        pl = QVBoxLayout(panel)
        pl.setAlignment(Qt.AlignTop)
        root.addWidget(panel)

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

        # ── Image de fond ──
        bg_g = QGroupBox("Image de fond")
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
        pl.addWidget(bg_g)

        # ── Image centre (Halo uniquement) ──
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
        for name in VIZ_TYPES:
            self._combo_viz.addItem(name)
        self._combo_viz.currentIndexChanged.connect(self._on_viz_changed)
        vl.addRow("Type", self._combo_viz)
        self._combo_palette = QComboBox()
        for name in PALETTES:
            self._combo_palette.addItem(name)
        self._combo_palette.addItem("Perso. Amplitude")
        self._combo_palette.addItem("Perso. Fréquence")
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
            self._on_params_changed)

        self._row_pulse = self._make_slider_row(
            pf, "Pulse (×0.01)", 0, 200, int(PULSE_INTENSITY * 100),
            self._on_params_changed)

        self._row_sensitivity = self._make_slider_row(
            pf, "Sensibilité (×0.01)", 25, 300, int(SENSITIVITY * 100),
            self._on_params_changed)

        self._row_rotation = self._make_slider_row(
            pf, "Rotation (°)", 0, 360, 0, self._on_params_changed)

        pl.addWidget(pg)

        # ── Répartition fréquentielle ──
        fq = QGroupBox("Répartition fréquentielle")
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
            ff, "Bins/octave", 6, 48, CQT_BINS_PER_OCTAVE,
            self._on_freq_changed)

        pl.addWidget(fq)
        self._update_freq_ui(cqt=False)

        # ── Halo Sine (mode 6 uniquement) ──
        self._halo_sine_group = QGroupBox("Halo Sine")
        hsf = QFormLayout(self._halo_sine_group)

        self._row_hs_r_base = self._make_slider_row(
            hsf, "Rayon base (×0.01)", 5, 70, int(HALO_SINE_R_BASE * 100),
            self._on_params_changed)

        self._row_hs_amplitude = self._make_slider_row(
            hsf, "Amplitude (×0.01)", 1, 40, int(HALO_SINE_AMPLITUDE * 100),
            self._on_params_changed)

        self._row_hs_points = self._make_slider_row(
            hsf, "Points spline", 16, 256, HALO_SINE_N_POINTS,
            self._on_params_changed)

        self._row_hs_glow = self._make_slider_row(
            hsf, "Couches de glow", 1, 8, HALO_SINE_GLOW_LAYERS,
            self._on_params_changed)

        self._row_hs_decay = self._make_slider_row(
            hsf, "Lissage (×0.01)", 0, 99, int(HALO_SINE_SMOOTHING_DECAY * 100),
            self._on_params_changed)

        self._row_hs_fill = self._make_slider_row(
            hsf, "Remplissage (×0.01)", 0, 100, int(HALO_SINE_FILL_OPACITY * 100),
            self._on_params_changed)

        pl.addWidget(self._halo_sine_group)

        # ── Effets beats ──
        beats_g = QGroupBox("Effets beats")
        beats_f = QFormLayout(beats_g)

        self._chk_bg_pulse = QCheckBox("Pulse BG")
        self._chk_bg_pulse.stateChanged.connect(self._on_params_changed)
        beats_f.addRow("", self._chk_bg_pulse)

        self._row_bg_pulse_intensity = self._make_slider_row(
            beats_f, "Intensité (×0.01)", 1, 100,
            int(BG_PULSE_INTENSITY * 100), self._on_params_changed)
        self._set_row_visible(self._row_bg_pulse_intensity, False)

        self._chk_flash = QCheckBox("Flash")
        self._chk_flash.stateChanged.connect(self._on_params_changed)
        beats_f.addRow("", self._chk_flash)

        self._row_flash_intensity = self._make_slider_row(
            beats_f, "Intensité (×0.01)", 1, 100,
            int(FLASH_INTENSITY * 100), self._on_params_changed)
        self._set_row_visible(self._row_flash_intensity, False)

        self._chk_bg_pulse.stateChanged.connect(
            lambda s: self._set_row_visible(self._row_bg_pulse_intensity, bool(s)))
        self._chk_flash.stateChanged.connect(
            lambda s: self._set_row_visible(self._row_flash_intensity, bool(s)))

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
        root.addWidget(self._preview)

    # ── Visibilité UI fréquentielle ──────────────────────────────
    def _update_freq_ui(self, cqt: bool):
        self._set_row_visible(self._row_bass_split, not cqt)
        self._set_row_visible(self._row_bass_hz,    not cqt)
        self._set_row_visible(self._row_cqt_bpo,    cqt)

    # ── Visibilité par mode ──────────────────────────────────────
    def _update_ui_for_viz(self, viz_type: int):
        is_radial    = viz_type in _RADIAL_MODES
        has_rotation = viz_type in _ROTATION_MODES
        is_bar       = viz_type in _BAR_MODES
        has_height   = viz_type in _HEIGHT_MODES
        is_halo      = viz_type in _HALO_MODES
        is_halo_sine = viz_type in _HALO_SINE_MODES

        self._set_row_visible(self._row_bars,      is_bar)
        self._set_row_visible(self._row_height,    has_height)
        self._set_row_visible(self._row_pulse,     is_radial)
        self._set_row_visible(self._row_rotation,  has_rotation)
        self._center_group.setVisible(is_halo)
        self._halo_sine_group.setVisible(is_halo_sine)
        pal_name = self._combo_palette.currentText()
        self._custom_amp_group.setVisible(pal_name == "Perso. Amplitude")
        self._custom_freq_group.setVisible(pal_name == "Perso. Fréquence")

    # ── Accesseurs ───────────────────────────────────────────────
    def _sl(self, row): return row[0]   # QSlider from a row tuple

    def _current_palette(self):
        name = self._combo_palette.currentText()
        if name == "Perso. Amplitude":
            return [list(c) for c in self._custom_palette_amp]
        if name == "Perso. Fréquence":
            return [list(c) for c in self._custom_palette_freq]
        return PALETTES[name]

    def _current_pal_mode(self) -> int:
        return 1 if self._combo_palette.currentText() == "Perso. Fréquence" else 0

    def _refresh_pal_buttons(self, buttons: list, palette: list):
        for i, c in enumerate(palette):
            r, g, b = c[0], c[1], c[2]
            hex_col = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
            lum     = 0.299*r + 0.587*g + 0.114*b
            txt_col = "#000000" if lum > 0.5 else "#ffffff"
            buttons[i].setStyleSheet(
                f"background-color:{hex_col};color:{txt_col};border:none;padding:4px;"
            )

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
        self._custom_amp_group.setVisible(pal_name == "Perso. Amplitude")
        self._custom_freq_group.setVisible(pal_name == "Perso. Fréquence")
        self._on_params_changed()

    def _current_viz_type(self):
        return VIZ_TYPES[self._combo_viz.currentText()]

    def _current_rotation_rad(self):
        return math.radians(self._sl(self._row_rotation).value())

    def _use_cqt(self) -> bool:
        return self._cqt_check.isChecked()

    def _make_fft(self) -> FFTProcessor:
        # Mode 6 does its own per-point smoothing — bypass FFT smoothing
        decay = (0.0 if self._current_viz_type() == 6
                 else self._sl(self._row_smooth).value() / 100)
        return FFTProcessor(
            sr=self._audio.sr,
            num_bars=self._sl(self._row_bars).value(),
            smoothing_decay=decay,
            bass_split=self._sl(self._row_bass_split).value() / 100,
            bass_split_hz=float(self._sl(self._row_bass_hz).value()),
            use_cqt=self._use_cqt(),
            bins_per_octave=self._sl(self._row_cqt_bpo).value(),
        )

    # ── Slots ────────────────────────────────────────────────────
    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir audio", "",
            "Audio (*.wav *.flac *.mp3 *.ogg *.aiff)")
        if not path:
            return
        self._audio = AudioFile(path)
        self._fft   = self._make_fft()
        self._file_label.setText(Path(path).name)
        self._btn_play.setEnabled(True)
        self._btn_export.setEnabled(True)
        self._status.setText(f"Durée : {self._audio.duration:.1f}s · {self._audio.sr} Hz")

    def _open_bg_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Image de fond", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)")
        if not path:
            return
        self._bg_image_path = path
        self._bg_label.setText(Path(path).name)
        self._btn_bg_clear.setEnabled(True)
        self._preview.load_background(path)

    def _clear_bg_image(self):
        self._bg_image_path = None
        self._bg_label.setText("Aucune image")
        self._btn_bg_clear.setEnabled(False)
        self._preview.clear_background()

    def _open_center_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Image centre", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)")
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
        self._update_ui_for_viz(self._current_viz_type())
        self._on_params_changed()

    def _on_params_changed(self):
        if self._audio and self._playback_thread and self._playback_thread.isRunning():
            new_fft = self._make_fft()
            self._playback_thread.update_fft(new_fft)
            self._fft = new_fft
        elif self._fft:
            self._fft.smoothing_decay = self._sl(self._row_smooth).value() / 100

        self._preview.set_params(
            num_bars=self._sl(self._row_bars).value(),
            max_bar_height=self._sl(self._row_height).value() / 100,
            pulse_intensity=self._sl(self._row_pulse).value() / 100,
            palette=self._current_palette(),
            sensitivity=self._sl(self._row_sensitivity).value() / 100,
            viz_type=self._current_viz_type(),
            rotation=self._current_rotation_rad(),
            halo_r_base=self._sl(self._row_hs_r_base).value() / 100,
            halo_amplitude=self._sl(self._row_hs_amplitude).value() / 100,
            halo_n_points=self._sl(self._row_hs_points).value(),
            halo_glow_layers=self._sl(self._row_hs_glow).value(),
            halo_smoothing_decay=self._sl(self._row_hs_decay).value() / 100,
            halo_fill_opacity=self._sl(self._row_hs_fill).value() / 100,
            pal_mode=self._current_pal_mode(),
            bg_pulse=self._chk_bg_pulse.isChecked(),
            bg_pulse_intensity=self._sl(self._row_bg_pulse_intensity).value() / 100,
            flash=self._chk_flash.isChecked(),
            flash_intensity=self._sl(self._row_flash_intensity).value() / 100,
        )

    def _on_freq_mode_changed(self):
        self._update_freq_ui(cqt=self._use_cqt())
        self._on_freq_changed()

    def _on_freq_changed(self):
        if self._audio:
            new_fft = self._make_fft()
            if self._playback_thread and self._playback_thread.isRunning():
                self._playback_thread.update_fft(new_fft)
            self._fft = new_fft

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
        self._playback_thread = AudioPlaybackThread(self._audio, self._fft, fps)
        self._playback_thread.frame_ready.connect(self._preview.update_audio_data)
        self._playback_thread.finished.connect(self._on_playback_done)
        self._playback_thread.start()
        self._btn_play.setText("⏹ Stop")

    def _stop_playback(self):
        if self._playback_thread:
            self._playback_thread.stop()
            self._playback_thread.wait()
        self._btn_play.setText("▶ Preview")

    def _on_playback_done(self):
        self._btn_play.setText("▶ Preview")

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
            output_dir=out_dir,
            resolution=self._combo_res.currentText(),
            fps=int(self._combo_fps.currentText()),
            num_bars=self._sl(self._row_bars).value(),
            max_bar_height=self._sl(self._row_height).value() / 100,
            smoothing_decay=self._sl(self._row_smooth).value() / 100,
            pulse_intensity=self._sl(self._row_pulse).value() / 100,
            palette=self._current_palette(),
            sensitivity=self._sl(self._row_sensitivity).value() / 100,
            viz_type=self._current_viz_type(),
            bg_image_path=self._bg_image_path,
            center_image_path=self._center_image_path,
            rotation=self._current_rotation_rad(),
            bass_split=self._sl(self._row_bass_split).value() / 100,
            bass_split_hz=float(self._sl(self._row_bass_hz).value()),
            use_cqt=self._use_cqt(),
            bins_per_octave=self._sl(self._row_cqt_bpo).value(),
            halo_r_base=self._sl(self._row_hs_r_base).value() / 100,
            halo_amplitude=self._sl(self._row_hs_amplitude).value() / 100,
            halo_n_points=self._sl(self._row_hs_points).value(),
            halo_glow_layers=self._sl(self._row_hs_glow).value(),
            halo_smoothing_decay=self._sl(self._row_hs_decay).value() / 100,
            halo_fill_opacity=self._sl(self._row_hs_fill).value() / 100,
            pal_mode=self._current_pal_mode(),
            bg_pulse=self._chk_bg_pulse.isChecked(),
            bg_pulse_intensity=self._sl(self._row_bg_pulse_intensity).value() / 100,
            flash=self._chk_flash.isChecked(),
            flash_intensity=self._sl(self._row_flash_intensity).value() / 100,
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
