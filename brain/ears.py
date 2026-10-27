"""the microphone: voice activity detection, then transcription.

energy gated vad with a hangover rather than a wake word, because the character already
knows whether it is being looked at. visual attention is the gate, which is the whole
point of a creature with a face: you look at it and talk, you do not say its name first.

transcription is faster-whisper base.en int8 on cpu, measured at 0.66 s for 3.3 s of
speech on the dev laptop. it runs on an utterance once the speaker stops, not streaming,
because the director has nothing useful to do with a partial sentence and the cpu budget
on the four core target is better spent elsewhere.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

SR = 16000            # whisper wants 16 khz, so capture there and skip a resample
BLOCK = 512           # 32 ms
START_RMS = 0.018     # speech onset threshold
STOP_RMS = 0.010      # lower than onset, so a trailing word does not cut the utterance
START_BLOCKS = 4      # ~128 ms over threshold before we call it speech
HANG = 0.70           # seconds of quiet that end an utterance
MAX_UTTER = 12.0      # hard cap so a noisy room cannot grow an unbounded buffer
PREROLL = 0.30        # seconds kept before onset, the first phoneme is always early


@dataclass
class Utterance:
    audio: np.ndarray
    started: float
    ended: float
    rms: float

    @property
    def seconds(self) -> float:
        return len(self.audio) / SR


class Ears:
    """captures utterances continuously and hands them over whole."""

    def __init__(self, device=None):
        self.utterances: queue.Queue[Utterance] = queue.Queue()
        self.level = 0.0            # smoothed input level, for the ui and for barge in
        self.speaking = False       # someone is talking right now
        self.enabled = True         # can be muted entirely
        self.boost = 1.0            # onset threshold multiplier, raised while it speaks
        self._buf: list[np.ndarray] = []
        self._pre: list[np.ndarray] = []
        self._loud = 0
        self._last_loud = 0.0
        self._started = 0.0
        self.stream = sd.InputStream(samplerate=SR, channels=1, blocksize=BLOCK,
                                     dtype="float32", callback=self._cb, device=device)
        self.lock = threading.Lock()

    def start(self) -> None:
        self.stream.start()

    def close(self) -> None:
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

    def _cb(self, indata, frames, time_info, status) -> None:
        mono = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(mono * mono)))
        self.level = 0.7 * self.level + 0.3 * rms
        now = time.time()

        if not self.enabled:
            self._buf.clear()
            self._loud = 0
            self.speaking = False
            return

        # a short rolling pre-roll so the utterance does not start mid word
        self._pre.append(mono)
        if len(self._pre) > int(PREROLL * SR / BLOCK):
            self._pre.pop(0)

        if not self.speaking:
            self._loud = self._loud + 1 if rms > START_RMS * self.boost else 0
            if self._loud >= START_BLOCKS:
                self.speaking = True
                self._started = now
                self._last_loud = now
                with self.lock:
                    self._buf = list(self._pre)
            return

        with self.lock:
            self._buf.append(mono)
        if rms > STOP_RMS:
            self._last_loud = now
        too_long = now - self._started > MAX_UTTER
        if now - self._last_loud > HANG or too_long:
            with self.lock:
                audio = np.concatenate(self._buf) if self._buf else np.zeros(0, np.float32)
                self._buf = []
            self.speaking = False
            self._loud = 0
            if len(audio) > int(0.25 * SR):      # ignore coughs and door clicks
                self.utterances.put(Utterance(audio, self._started, now,
                                              float(np.sqrt(np.mean(audio ** 2)))))

    def next_utterance(self, timeout: float = 0.0) -> Utterance | None:
        try:
            return self.utterances.get(timeout=timeout) if timeout else self.utterances.get_nowait()
        except queue.Empty:
            return None

    def drain(self) -> None:
        while self.next_utterance() is not None:
            pass


class Transcriber:
    """faster-whisper in a worker thread, so a 0.7 s transcribe never stalls the director."""

    def __init__(self, size: str = "base.en"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(size, device="cpu", compute_type="int8")
        self.jobs: queue.Queue = queue.Queue()
        self.results: queue.Queue = queue.Queue()
        self.times: list[float] = []
        threading.Thread(target=self._work, name="stt", daemon=True).start()

    def submit(self, utt: Utterance) -> None:
        self.jobs.put(utt)

    def _work(self) -> None:
        while True:
            utt = self.jobs.get()
            t0 = time.perf_counter()
            segs, _ = self.model.transcribe(utt.audio, beam_size=1, language="en",
                                            condition_on_previous_text=False)
            text = " ".join(s.text for s in segs).strip()
            dt = time.perf_counter() - t0
            self.times.append(dt)
            self.results.put({"text": text, "utterance": utt, "stt_s": dt,
                              "rtf": dt / max(utt.seconds, 1e-3)})

    def poll(self) -> dict | None:
        try:
            return self.results.get_nowait()
        except queue.Empty:
            return None
