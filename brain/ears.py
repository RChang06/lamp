from __future__ import annotations

import os
import queue
import re
import threading
import time
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import sounddevice as sd

SR = 16000
BLOCK = 512
VAD_MODEL = "models/silero_vad_16k_op15.onnx"
WHISPER_DIR = "models/whisper"
START_PROB = 0.5
STOP_PROB = 0.35
FLOOR_RMS = 0.004
START_BLOCKS = 5
START_RMS = FLOOR_RMS * 4
STOP_RMS = FLOOR_RMS
HANG = 0.70
MAX_UTTER = 12.0
PREROLL = 0.30


@dataclass
class Utterance:
    audio: np.ndarray
    started: float
    ended: float
    rms: float

    @property
    def seconds(self) -> float:
        return len(self.audio) / SR


class SileroVAD:
    def __init__(self, path: str = VAD_MODEL):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path, sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.sr = np.array(SR, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def __call__(self, block: np.ndarray) -> float:
        chunk = block.reshape(1, -1).astype(np.float32)
        if chunk.shape[1] != BLOCK:
            chunk = np.pad(chunk, ((0, 0), (0, max(0, BLOCK - chunk.shape[1]))))[:, :BLOCK]
        prob, state = self.session.run(None, {
            "input": np.concatenate((self.context, chunk), axis=1),
            "state": self.state, "sr": self.sr})
        self.state = np.asarray(state, dtype=np.float32)
        self.context = chunk[:, -64:].copy()
        return float(np.asarray(prob).item())


class Ears:
    def __init__(self, device=None):
        self.utterances: queue.Queue[Utterance] = queue.Queue()
        self.level = 0.0
        self.speaking = False
        self.enabled = True
        self.boost = 1.0
        self.prob = 0.0
        self.vad = SileroVAD()
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

        self._pre.append(mono)
        if len(self._pre) > int(PREROLL * SR / BLOCK):
            self._pre.pop(0)

        self.prob = self.vad(mono)
        start_prob = min(0.9, START_PROB + 0.15 * (self.boost - 1.0))
        if not self.speaking:
            is_speech = self.prob >= start_prob and rms > FLOOR_RMS * self.boost
            self._loud = self._loud + 1 if is_speech else 0
            if self._loud >= START_BLOCKS:
                self.speaking = True
                self._started = now
                self._last_loud = now
                with self.lock:
                    self._buf = list(self._pre)
            return

        with self.lock:
            self._buf.append(mono)
        if self.prob >= STOP_PROB and rms > FLOOR_RMS:
            self._last_loud = now
        too_long = now - self._started > MAX_UTTER
        if now - self._last_loud > HANG or too_long:
            with self.lock:
                audio = np.concatenate(self._buf) if self._buf else np.zeros(0, np.float32)
                self._buf = []
            self.speaking = False
            self._loud = 0
            if len(audio) > int(0.25 * SR):
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


MAX_TOKENS_PER_S = 8
MAX_WORDS_PER_S = 5.0
PROMPT = "Talking to a small desk lamp robot."
ASR_ENGINE = os.environ.get("LAMP_ASR", "parakeet")
PARAKEET = "nemo-parakeet-tdt-0.6b-v2"
PARAKEET_DIR = "models/parakeet-int8"
PARAKEET_FP32_DIR = "models/parakeet"


def lean_session() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return opts


def hallucinated(seg) -> bool:
    return seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0


FIXES = [(re.compile(r"\blamb\b", re.I), "lamp")]


class Transcriber:
    def __init__(self, size: str = "base.en", keep_dir: str | None = None,
                 engine: str = ASR_ENGINE):
        self.keep_dir = keep_dir
        self.engine = engine
        self.model = None
        if engine.startswith("parakeet"):
            try:
                import onnx_asr
                fp32 = engine == "parakeet-fp32"
                self.model = onnx_asr.load_model(
                    PARAKEET, PARAKEET_FP32_DIR if fp32 else PARAKEET_DIR,
                    quantization=None if fp32 else "int8", sess_options=lean_session())
            except Exception as exc:
                print(f"parakeet unavailable ({type(exc).__name__}: {exc}), using whisper",
                      flush=True)
                self.engine = "whisper"
        if self.model is None:
            from faster_whisper import WhisperModel
            self.engine = f"whisper {size}"
            self.model = WhisperModel(size, device="cpu", compute_type="int8",
                                      download_root=WHISPER_DIR)
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
            if self.engine.startswith("parakeet"):
                text = self.model.recognize(utt.audio, sample_rate=SR).strip()
            else:
                # whisper hallucinates on noise, cap the tokens
                budget = int(utt.seconds * MAX_TOKENS_PER_S) + 8
                segs, _ = self.model.transcribe(utt.audio, beam_size=1, language="en",
                                                condition_on_previous_text=False,
                                                without_timestamps=True,
                                                max_new_tokens=budget,
                                                initial_prompt=PROMPT)
                kept = [s for s in segs if not hallucinated(s)]
                text = " ".join(s.text for s in kept).strip()
            for pattern, word in FIXES:
                text = pattern.sub(word, text)
            if len(text.split()) > utt.seconds * MAX_WORDS_PER_S + 3:
                text = ""
            dt = time.perf_counter() - t0
            self.times.append(dt)
            if self.keep_dir:
                self._keep(utt, text)
            self.results.put({"text": text, "utterance": utt, "stt_s": dt,
                              "rtf": dt / max(utt.seconds, 1e-3)})

    def _keep(self, utt: Utterance, text: str) -> None:
        import json
        import wave
        os.makedirs(self.keep_dir, exist_ok=True)
        name = time.strftime("%Y%m%d-%H%M%S") + f"-{int(utt.started * 1000) % 1000:03d}.wav"
        with wave.open(os.path.join(self.keep_dir, name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((np.clip(utt.audio, -1, 1) * 32767).astype(np.int16).tobytes())
        with open(os.path.join(self.keep_dir, "transcripts.jsonl"), "a") as f:
            f.write(json.dumps({"wav": name, "text": text, "seconds": round(utt.seconds, 2)}) + "\n")

    def poll(self) -> dict | None:
        try:
            return self.results.get_nowait()
        except queue.Empty:
            return None
