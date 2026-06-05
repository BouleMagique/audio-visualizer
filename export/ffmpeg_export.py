import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Callable

from core.audio import AudioFile
from core.fft import FFTProcessor
from render.renderer import Renderer
from config.defaults import (
    RESOLUTIONS, FPS, FFT_SIZE, SENSITIVITY,
    HALO_SINE_R_BASE, HALO_SINE_AMPLITUDE, HALO_SINE_N_POINTS,
    HALO_SINE_GLOW_LAYERS, HALO_SINE_SMOOTHING_DECAY, HALO_SINE_FILL_OPACITY,
    HALO_SINE_SPLINE_GAP, HALO_SINE_PIXEL_SIZE,
    TUNNEL_SIDES, TUNNEL_RINGS, TUNNEL_SPEED, TUNNEL_KICK_ZOOM, TUNNEL_CHROMA,
    TUNNEL_KICK_SENSITIVITY,
    CQT_BINS_PER_OCTAVE, BG_PULSE_INTENSITY, FLASH_INTENSITY,
)


class FFmpegExporter:
    def __init__(self, audio: AudioFile, output_dir: str = ".",
                 resolution: str = "1080p", fps: int = FPS,
                 num_bars: int = 128, max_bar_height: float = 1.0,
                 smoothing_decay: float = 0.85, pulse_intensity: float = 1.0,
                 palette: list | None = None,
                 sensitivity: float = SENSITIVITY,
                 viz_type: int = 0,
                 bg_image_path: str | None = None,
                 center_image_path: str | None = None,
                 rotation: float = 0.0,
                 bass_split: float = 0.60,
                 bass_split_hz: float = 300.0,
                 use_cqt: bool = False,
                 bins_per_octave: int = CQT_BINS_PER_OCTAVE,
                 halo_r_base: float = HALO_SINE_R_BASE,
                 halo_amplitude: float = HALO_SINE_AMPLITUDE,
                 halo_n_points: int = HALO_SINE_N_POINTS,
                 halo_glow_layers: int = HALO_SINE_GLOW_LAYERS,
                 halo_smoothing_decay: float = HALO_SINE_SMOOTHING_DECAY,
                 halo_fill_opacity: float = HALO_SINE_FILL_OPACITY,
                 halo_spline_gap: float = HALO_SINE_SPLINE_GAP,
                 halo_pixel_size: int = HALO_SINE_PIXEL_SIZE,
                 tunnel_sides: int = TUNNEL_SIDES,
                 tunnel_rings: int = TUNNEL_RINGS,
                 tunnel_speed: float = TUNNEL_SPEED,
                 tunnel_kick_zoom: float = TUNNEL_KICK_ZOOM,
                 tunnel_chroma: float = TUNNEL_CHROMA,
                 tunnel_kick_sensitivity: float = TUNNEL_KICK_SENSITIVITY,
                 pal_mode: int = 0,
                 bg_pulse: bool = False,
                 bg_pulse_intensity: float = BG_PULSE_INTENSITY,
                 flash: bool = False,
                 flash_intensity: float = FLASH_INTENSITY,
                 progress_cb: Callable[[int, int], None] = None):
        self.audio = audio
        self.output_dir = Path(output_dir)
        self.width, self.height = RESOLUTIONS[resolution]
        self.fps = fps
        self.num_bars = num_bars
        self.max_bar_height = max_bar_height
        self.pulse_intensity = pulse_intensity
        self.palette = palette
        self.sensitivity = sensitivity
        self.viz_type = viz_type
        self.bg_image_path = bg_image_path
        self.center_image_path = center_image_path
        self.rotation = rotation
        self.halo_r_base = halo_r_base
        self.halo_amplitude = halo_amplitude
        self.halo_n_points = halo_n_points
        self.halo_glow_layers = halo_glow_layers
        self.halo_smoothing_decay = halo_smoothing_decay
        self.halo_fill_opacity = halo_fill_opacity
        self.halo_spline_gap = halo_spline_gap
        self.halo_pixel_size = halo_pixel_size
        self.tunnel_sides = tunnel_sides
        self.tunnel_rings = tunnel_rings
        self.tunnel_speed = tunnel_speed
        self.tunnel_kick_zoom = tunnel_kick_zoom
        self.tunnel_chroma = tunnel_chroma
        self.tunnel_kick_sensitivity = tunnel_kick_sensitivity
        self.pal_mode = pal_mode
        self.bg_pulse = bg_pulse
        self.bg_pulse_intensity = bg_pulse_intensity
        self.flash = flash
        self.flash_intensity = flash_intensity
        self.progress_cb = progress_cb

        self.hop_size = int(audio.sr / fps)
        # Mode 6 does per-point smoothing in PIL — bypass FFT smoothing
        fft_decay = 0.0 if viz_type == 6 else smoothing_decay
        self.fft = FFTProcessor(
            sr=audio.sr,
            fft_size=FFT_SIZE,
            num_bars=num_bars,
            smoothing_decay=fft_decay,
            bass_split=bass_split,
            bass_split_hz=bass_split_hz,
            use_cqt=use_cqt,
            bins_per_octave=bins_per_octave,
        )
        self.total_frames = int(np.ceil(len(audio.mono) / self.hop_size))

    def _build_ffmpeg_cmd(self, output_path: str) -> list[str]:
        return [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "rgb24",
            "-r", str(self.fps),
            "-i", "-",
            "-i", str(self.audio.path),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "320k",
            "-shortest",
            output_path,
        ]

    def export(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(self.output_dir / f"{self.audio.path.stem}_{ts}.mp4")

        renderer = Renderer(self.width, self.height)
        if self.bg_image_path:
            renderer.load_background(self.bg_image_path)
        if self.center_image_path:
            renderer.load_center_image(self.center_image_path)

        self.fft.reset()

        with subprocess.Popen(self._build_ffmpeg_cmd(out_path), stdin=subprocess.PIPE) as proc:
            for frame_idx in range(self.total_frames):
                samples = self.audio.get_frame_samples(frame_idx, self.hop_size, FFT_SIZE)
                bars, pulse = self.fft.process(samples)

                renderer.render_frame(
                    bars=bars,
                    pulse=pulse,
                    num_bars=self.num_bars,
                    max_bar_height=self.max_bar_height,
                    pulse_intensity=self.pulse_intensity,
                    palette=self.palette,
                    sensitivity=self.sensitivity,
                    viz_type=self.viz_type,
                    time=frame_idx / self.fps,
                    rotation=self.rotation,
                    halo_r_base=self.halo_r_base,
                    halo_amplitude=self.halo_amplitude,
                    halo_n_points=self.halo_n_points,
                    halo_glow_layers=self.halo_glow_layers,
                    halo_smoothing_decay=self.halo_smoothing_decay,
                    halo_fill_opacity=self.halo_fill_opacity,
                    halo_spline_gap=self.halo_spline_gap,
                    halo_pixel_size=self.halo_pixel_size,
                    tunnel_sides=self.tunnel_sides,
                    tunnel_rings=self.tunnel_rings,
                    tunnel_speed=self.tunnel_speed,
                    tunnel_kick_zoom=self.tunnel_kick_zoom,
                    tunnel_chroma=self.tunnel_chroma,
                    tunnel_kick_sensitivity=self.tunnel_kick_sensitivity,
                    pal_mode=self.pal_mode,
                    bg_pulse=self.bg_pulse,
                    bg_pulse_intensity=self.bg_pulse_intensity,
                    flash=self.flash,
                    flash_intensity=self.flash_intensity,
                )
                proc.stdin.write(renderer.read_frame())

                if self.progress_cb:
                    self.progress_cb(frame_idx + 1, self.total_frames)

            proc.stdin.close()
            proc.wait()

        renderer.release()
        return out_path
