"""Live audio input: capture what the machine plays (or any input) for the preview.

Sources
-------
* Linux (PipeWire / PulseAudio): every Pulse source is listed via `pactl`, sink
  monitors first ("Sortie PC : …" = what YouTube / Spotify / Traktor play). They are
  opened through PortAudio's `pulse` ALSA device with PULSE_SOURCE pointing at the
  source, since PortAudio cannot see monitors on its own.
* Elsewhere (macOS + BlackHole, Windows + Stereo Mix / VB-Cable): PortAudio input devices.

The audio callback only appends to a ring buffer; analysis happens on the render
thread, on the most recent samples, so the picture always reflects the latest audio.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

LIVE_SR = 48000
_BLOCK = 256            # 5.3 ms callback blocks
_RING = 16384           # > largest analysis window (4096) with room to spare


@dataclass(frozen=True)
class LiveSource:
    label: str
    device: str | int           # PortAudio device (name or index)
    pulse_source: str | None = None


def _pulse_sources() -> list[LiveSource]:
    if not sys.platform.startswith("linux") or not shutil.which("pactl"):
        return []
    try:
        out = subprocess.run(["pactl", "-f", "json", "list", "sources"],
                             capture_output=True, text=True, timeout=3).stdout
        sources = json.loads(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    monitors, inputs = [], []
    for s in sources:
        name = s.get("name", "")
        desc = s.get("description") or name
        if name.endswith(".monitor"):
            monitors.append(LiveSource(f"Sortie PC : {desc.removeprefix('Monitor of ')}",
                                       "pulse", name))
        else:
            inputs.append(LiveSource(f"Entrée : {desc}", "pulse", name))
    return monitors + inputs


def _default_monitor() -> str | None:
    """Monitor of the default output — what the user hears right now."""
    try:
        sink = subprocess.run(["pactl", "get-default-sink"], capture_output=True,
                              text=True, timeout=3).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{sink}.monitor" if sink else None


def list_sources() -> tuple[list[LiveSource], int]:
    """→ (sources, index of the one to preselect)."""
    sources = _pulse_sources()
    if sources:
        mon = _default_monitor()
        idx = next((i for i, s in enumerate(sources) if s.pulse_source == mon), 0)
        return sources, idx

    sources = []
    default_in = sd.default.device[0] if sd.default.device else -1
    idx = 0
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            if i == default_in:
                idx = len(sources)
            sources.append(LiveSource(d["name"], i))
    # BlackHole / loopback drivers are what a live set wants by default
    for j, s in enumerate(sources):
        if "blackhole" in s.label.lower() or "loopback" in s.label.lower():
            idx = j
            break
    return sources, idx


class LiveInput:
    """Ring-buffered mono capture. latest(n) is safe to call from any thread."""

    def __init__(self, source: LiveSource, sr: int = LIVE_SR):
        self.source = source
        self.sr = sr
        self._ring = np.zeros(_RING, dtype=np.float32)
        self._pos = 0                       # total samples written
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self.peak = 0.0                     # last block peak, for a level meter

    def start(self) -> None:
        if self.source.pulse_source:
            # Read by the Pulse client library when the ALSA `pulse` device connects
            os.environ["PULSE_SOURCE"] = self.source.pulse_source
            channels = 2
        else:
            channels = min(2, sd.query_devices(self.source.device)["max_input_channels"])
        self._stream = sd.InputStream(
            device=self.source.device, channels=channels,
            samplerate=self.sr, blocksize=_BLOCK, latency="low", dtype="float32",
            callback=self._callback)
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def input_latency(self) -> float:
        return float(self._stream.latency) if self._stream is not None else 0.0

    def _callback(self, indata, frames, time_info, status) -> None:
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata[:, 0]
        self.peak = float(np.abs(mono).max()) if frames else 0.0
        with self._lock:
            i = self._pos % _RING
            first = min(frames, _RING - i)
            self._ring[i:i + first] = mono[:first]
            self._ring[:frames - first] = mono[first:]
            self._pos += frames

    def latest(self, n: int) -> np.ndarray:
        """The last n samples, oldest first."""
        with self._lock:
            end = self._pos % _RING
            if end >= n:
                return self._ring[end - n:end].copy()
            return np.concatenate((self._ring[end - n:], self._ring[:end]))
