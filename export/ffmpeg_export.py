import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Callable

from core.audio import AudioFile
from core.fft import FFTProcessor
from core.layer import LayerManager
from render.renderer import Renderer
from config.defaults import (
    RESOLUTIONS, FPS, FFT_SIZE, CQT_BINS_PER_OCTAVE,
)

# Global analysis resolution; layers resample this to their own bar count.
FFT_ANALYSIS_BARS = 256


class FFmpegExporter:
    """Renders a LayerManager composition frame-by-frame and pipes to FFmpeg."""

    def __init__(self, audio: AudioFile, layer_manager: LayerManager,
                 output_dir: str = ".",
                 resolution: str = "1080p", fps: int = FPS,
                 smoothing_decay: float = 0.85,
                 bass_split: float = 0.60,
                 bass_split_hz: float = 300.0,
                 use_cqt: bool = False,
                 bins_per_octave: int = CQT_BINS_PER_OCTAVE,
                 bg_image_path: str | None = None,
                 center_image_path: str | None = None,
                 progress_cb: Callable[[int, int], None] = None):
        self.audio = audio
        self.lm = layer_manager
        self.output_dir = Path(output_dir)
        self.width, self.height = RESOLUTIONS[resolution]
        self.fps = fps
        self.bg_image_path = bg_image_path
        self.center_image_path = center_image_path
        self.progress_cb = progress_cb

        self.hop_size = int(audio.sr / fps)
        self.fft = FFTProcessor(
            sr=audio.sr,
            fft_size=FFT_SIZE,
            num_bars=FFT_ANALYSIS_BARS,
            smoothing_decay=smoothing_decay,
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

                renderer.render_composition(
                    self.lm, bars, pulse, time=frame_idx / self.fps)
                proc.stdin.write(renderer.read_frame())

                if self.progress_cb:
                    self.progress_cb(frame_idx + 1, self.total_frames)

            proc.stdin.close()
            proc.wait()

        renderer.release()
        return out_path
