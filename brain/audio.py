"""one output stream for everything the character makes: voice, sound effects, music.

the point of mixing these ourselves rather than firing three players is that they have to
behave like one creature. music ducks under the voice, an effect lands on top without
cutting either, and the voice channel's level is published every block so the body can
move and the light can breathe with the speech. the lamp's glow following its own voice is
the single cheapest thing that makes the character read as one thing instead of a speaker
sat next to a puppet.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import sounddevice as sd

SR = 22050          # piper lessac medium is 22.05 khz, so nothing is resampled
BLOCK = 512         # 23 ms, small enough that the envelope leads the motion


@dataclass
class Channel:
    data: np.ndarray
    pos: int = 0
    gain: float = 1.0
    loop: bool = False
    fade: float = 1.0           # current fade multiplier
    fade_to: float = 1.0
    done: bool = False

    def take(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        if self.done:
            return out
        if self.loop:
            idx = (np.arange(self.pos, self.pos + n) % len(self.data))
            out[:] = self.data[idx]
            self.pos = (self.pos + n) % len(self.data)
        else:
            end = min(self.pos + n, len(self.data))
            out[: end - self.pos] = self.data[self.pos : end]
            self.pos = end
            if self.pos >= len(self.data):
                self.done = True
        # per block linear fade toward the target, used for ducking and clean stops
        ramp = np.linspace(self.fade, self.fade_to, n, dtype=np.float32)
        self.fade = self.fade_to
        return out * self.gain * ramp


class AudioOut:
    """mixes the channels, ducks music under speech, reports the voice level."""

    def __init__(self, on_level: Callable[[float], None] | None = None, device=None):
        self.ch: dict[str, Channel] = {}
        self.lock = threading.Lock()
        self.on_level = on_level
        self.stream = sd.OutputStream(samplerate=SR, channels=1, blocksize=BLOCK,
                                      dtype="float32", callback=self._cb, device=device)

    def start(self) -> None:
        self.stream.start()

    def close(self) -> None:
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

    def _cb(self, outdata, frames, time_info, status) -> None:
        mix = np.zeros(frames, dtype=np.float32)
        voice_level = 0.0
        with self.lock:
            speaking = "voice" in self.ch and not self.ch["voice"].done
            for name, c in list(self.ch.items()):
                if name == "music":
                    c.fade_to = 0.30 if speaking else 1.0   # duck under the voice
                block = c.take(frames)
                if name == "voice":
                    voice_level = float(np.sqrt(np.mean(block * block)))
                mix += block
                if c.done:
                    del self.ch[name]
        np.clip(mix, -1.0, 1.0, out=mix)
        outdata[:, 0] = mix
        if self.on_level is not None:
            # 0.12 rms is a comfortable speaking level, so that maps to a full swing
            self.on_level(min(1.0, voice_level / 0.12))

    def play(self, name: str, data: np.ndarray, gain: float = 1.0, loop: bool = False) -> None:
        with self.lock:
            self.ch[name] = Channel(np.ascontiguousarray(data, dtype=np.float32),
                                    gain=gain, loop=loop)

    def stop(self, name: str) -> None:
        with self.lock:
            self.ch.pop(name, None)

    def busy(self, name: str = "voice") -> bool:
        with self.lock:
            c = self.ch.get(name)
            return c is not None and not c.done


# ---- procedural sounds -------------------------------------------------------------
# synthesized rather than shipped as wav files: no asset licensing to explain in the
# submission, a few hundred bytes of code instead of megabytes, and the character's
# effects and music end up sharing one tuning so they sound related.

def _env(n: int, attack: float = 0.01, release: float = 0.25) -> np.ndarray:
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    r = np.exp(-t / max(release, 1e-4))
    return (a * r).astype(np.float32)


def _tone(freq, dur, kind="sine", detune=0.0):
    t = np.arange(int(SR * dur)) / SR
    f = np.asarray(freq, dtype=np.float32)
    if f.ndim == 0:
        f = np.full_like(t, float(f), dtype=np.float32)
    phase = 2 * np.pi * np.cumsum(f) / SR
    wave = np.sin(phase)
    if kind == "tri":
        wave = 2 / np.pi * np.arcsin(np.sin(phase))
    if detune:
        wave = 0.5 * wave + 0.5 * np.sin(phase * (1 + detune))
    return wave.astype(np.float32)


def chirp_up(dur=0.22) -> np.ndarray:
    n = int(SR * dur)
    f = np.linspace(520, 1180, n)
    return _tone(f, dur) * _env(n, 0.005, 0.10) * 0.5


def chirp_down(dur=0.30) -> np.ndarray:
    n = int(SR * dur)
    f = np.linspace(760, 380, n)
    return _tone(f, dur) * _env(n, 0.008, 0.16) * 0.4


def blip(dur=0.10) -> np.ndarray:
    n = int(SR * dur)
    return _tone(880, dur, "tri") * _env(n, 0.003, 0.05) * 0.35


def thinking(dur=0.9) -> np.ndarray:
    # two close tones beating slowly, reads as "working on it"
    n = int(SR * dur)
    return (_tone(392, dur, detune=0.006) * _env(n, 0.08, 0.6) * 0.22)


def found_it(dur=0.45) -> np.ndarray:
    # a small rising third, the sound of recognising something
    parts = [_tone(587, 0.12), _tone(740, 0.12), _tone(880, 0.21)]
    out = np.concatenate([p * _env(len(p), 0.004, 0.09) for p in parts])
    return out * 0.4


def music_bed(bars: float = 4.0, bpm: float = 68.0) -> np.ndarray:
    """a slow warm loop, quiet enough to sit under speech. kept simple and diatonic so
    it never fights the voice for attention."""
    beat = 60.0 / bpm
    seq = [262, 330, 392, 330, 294, 349, 440, 349]      # c major wandering, two bars
    notes = []
    for i in range(int(len(seq) * bars / 2)):
        f = seq[i % len(seq)] / 2
        d = beat * 0.95
        n = int(SR * d)
        notes.append(_tone(f, d, detune=0.004) * _env(n, 0.12, d * 0.8) * 0.5)
    loop = np.concatenate(notes)
    # soft pad underneath to glue the notes together
    pad = _tone(131, len(loop) / SR, detune=0.003) * 0.12
    out = loop + pad
    # crossfade the ends so the loop does not click
    k = int(0.15 * SR)
    ramp = np.linspace(0, 1, k, dtype=np.float32)
    out[:k] *= ramp
    out[-k:] *= ramp[::-1]
    return (out / max(1e-6, np.abs(out).max()) * 0.5).astype(np.float32)


SFX = {
    "chirp_up": chirp_up,
    "chirp_down": chirp_down,
    "blip": blip,
    "thinking": thinking,
    "found_it": found_it,
}
