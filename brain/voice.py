"""the speaker side: piper tts, sentence at a time, interruptible.

sentence at a time matters twice. the first sentence starts playing while the rest is
still being synthesized, which cuts the time to first sound to well under the time to
synthesize the whole answer, and it gives barge in a clean seam: when the person starts
talking the current sentence stops and the queue is dropped, so the character does not
keep monologuing over them.

the audio level the mixer reports is forwarded to the body as the speech envelope, which
is what makes the head move and the bulb pulse with the voice.
"""
from __future__ import annotations

import queue
import re
import threading
import time

import numpy as np

from brain.audio import SR, AudioOut

MODEL = "models/piper/en_US-lessac-medium.onnx"
SENTENCE = re.compile(r"(?<=[.!?])\s+")


class Voice:
    def __init__(self, audio: AudioOut, model: str = MODEL):
        from piper import PiperVoice
        self.piper = PiperVoice.load(model)
        self.audio = audio
        self.pending: queue.Queue[str] = queue.Queue()
        self.cancelled = threading.Event()
        self.speaking = False
        self.synth_times: list[tuple[float, float]] = []   # (synth seconds, audio seconds)
        self.first_sound_ms: list[float] = []
        threading.Thread(target=self._work, name="tts", daemon=True).start()

    # ---- synthesis -------------------------------------------------------------------
    def synth(self, text: str) -> np.ndarray:
        t0 = time.perf_counter()
        chunks = list(self.piper.synthesize(text))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        if chunks[0].sample_rate != SR:
            # keep this honest rather than silently playing at the wrong pitch
            n = int(len(audio) * SR / chunks[0].sample_rate)
            audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                              np.arange(len(audio)), audio).astype(np.float32)
        self.synth_times.append((time.perf_counter() - t0, len(audio) / SR))
        return audio

    # ---- queue ----------------------------------------------------------------------
    def say(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.cancelled.clear()
        self._asked = time.perf_counter()
        for part in SENTENCE.split(text):
            if part.strip():
                self.pending.put(part.strip())

    def cancel(self) -> None:
        """barge in: drop everything queued and cut the sentence in flight."""
        self.cancelled.set()
        while True:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                break
        self.audio.stop("voice")
        self.speaking = False

    def busy(self) -> bool:
        return self.speaking or not self.pending.empty() or self.audio.busy("voice")

    def _work(self) -> None:
        while True:
            text = self.pending.get()
            if self.cancelled.is_set():
                continue
            audio = self.synth(text)
            if self.cancelled.is_set() or len(audio) == 0:
                continue
            first = getattr(self, "_asked", None)
            if first is not None:
                self.first_sound_ms.append(1000 * (time.perf_counter() - first))
                self._asked = None
            self.speaking = True
            self.audio.play("voice", audio, gain=0.95)
            # wait out playback so sentences do not overlap, but stay responsive to a cancel
            end = time.perf_counter() + len(audio) / SR
            while time.perf_counter() < end:
                if self.cancelled.is_set():
                    break
                time.sleep(0.02)
            self.speaking = False
