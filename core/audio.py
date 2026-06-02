import numpy as np
import soundfile as sf
from pathlib import Path


class AudioFile:
    def __init__(self, path: str):
        self.path = Path(path)
        self.samples, self.sr = sf.read(str(path), dtype="float32", always_2d=True)
        # Mix down to mono
        self.mono = self.samples.mean(axis=1)
        self.duration = len(self.mono) / self.sr

    def get_frame_samples(self, frame_idx: int, hop_size: int, fft_size: int) -> np.ndarray:
        start = frame_idx * hop_size
        end = start + fft_size
        chunk = self.mono[start:end] if end <= len(self.mono) else np.zeros(fft_size, dtype=np.float32)
        if len(chunk) < fft_size:
            chunk = np.pad(chunk, (0, fft_size - len(chunk)))
        return chunk.astype(np.float32)

    @property
    def num_frames(self) -> int:
        from config.defaults import FPS
        hop = self.sr / FPS
        return int(np.ceil(len(self.mono) / hop))
