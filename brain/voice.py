from __future__ import annotations

import os
import queue
import re
import threading
import time

import numpy as np

from brain.audio import SR, AudioOut

MODEL = "models/piper/en_US-lessac-medium.onnx"
KOKORO_MODEL = "models/kokoro/kokoro-v1.0.onnx"
KOKORO_VOICES = "models/kokoro/voices-v1.0.bin"
KOKORO_VOICE = os.environ.get("LAMP_VOICE", "af_heart")
KOKORO_SPEED = 1.05
KOKORO_THREADS = 4
SENTENCE = re.compile(r"(?<=[.!?])\s+")


class Voice:
    def __init__(self, audio: AudioOut, model: str = MODEL):
        self.kokoro = self._load_kokoro()
        self.engine = "kokoro " + KOKORO_VOICE if self.kokoro is not None else "piper"
        self.piper = None
        if self.kokoro is None:
            from piper import PiperVoice
            self.piper = PiperVoice.load(model)
        self.synthesizing = False
        self.audio = audio
        self.pending: queue.Queue[str] = queue.Queue()
        self.cancelled = threading.Event()
        self.speaking = False
        self.synth_times: list[tuple[float, float]] = []
        self.first_sound_ms: list[float] = []
        threading.Thread(target=self._work, name="tts", daemon=True).start()

    @staticmethod
    def _load_kokoro():
        if os.environ.get("LAMP_VOICE") == "piper" or not os.path.isfile(KOKORO_MODEL):
            return None
        try:
            import onnxruntime as ort
            from kokoro_onnx import Kokoro
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = KOKORO_THREADS
            opts.inter_op_num_threads = 1
            opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
            sess = ort.InferenceSession(KOKORO_MODEL, sess_options=opts,
                                        providers=["CPUExecutionProvider"])
            k = Kokoro.from_session(sess, KOKORO_VOICES)
            k.create("Hi.", voice=KOKORO_VOICE, lang="en-us")
            return k
        except Exception as exc:
            print(f"kokoro unavailable ({type(exc).__name__}: {exc}), using piper", flush=True)
            return None

    def synth(self, text: str) -> np.ndarray:
        t0 = time.perf_counter()
        if self.kokoro is not None:
            audio, sr = self.kokoro.create(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED,
                                           lang="en-us")
            audio = np.asarray(audio, dtype=np.float32)
        else:
            chunks = list(self.piper.synthesize(text))
            if not chunks:
                return np.zeros(0, dtype=np.float32)
            audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
            sr = chunks[0].sample_rate
        if len(audio) and sr != SR:
            n = int(len(audio) * SR / sr)
            audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                              np.arange(len(audio)), audio).astype(np.float32)
        self.synth_times.append((time.perf_counter() - t0, len(audio) / SR))
        return audio

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
        self.cancelled.set()
        while True:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                break
        self.audio.stop("voice")
        self.speaking = False

    def busy(self) -> bool:
        return (self.speaking or self.synthesizing or not self.pending.empty()
                or self.audio.busy("voice"))

    def _work(self) -> None:
        while True:
            text = self.pending.get()
            if self.cancelled.is_set():
                continue
            self.synthesizing = True
            try:
                audio = self.synth(text)
                if self.cancelled.is_set() or len(audio) == 0:
                    continue
                self.speaking = True
            finally:
                self.synthesizing = False
            first = getattr(self, "_asked", None)
            if first is not None:
                self.first_sound_ms.append(1000 * (time.perf_counter() - first))
                self._asked = None
            self.audio.play("voice", audio, gain=0.95)
            end = time.perf_counter() + len(audio) / SR
            while time.perf_counter() < end:
                if self.cancelled.is_set():
                    break
                time.sleep(0.02)
            self.speaking = False
