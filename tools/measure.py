from __future__ import annotations

import asyncio
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, ".")

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from common import protocol as P

PORT = 8797
PY = sys.executable


def heading(text: str) -> None:
    print(f"\n{text}\n" + "-" * len(text))


def stat(name: str, values: list[float], unit: str = "ms") -> None:
    if not values:
        print(f"{name:<34} no samples")
        return
    a = np.array(values)
    print(f"{name:<34} n={len(a):<4} mean {a.mean():7.2f} p50 {np.median(a):7.2f} "
          f"p95 {np.percentile(a, 95):7.2f} max {a.max():7.2f} {unit}")


def measure_vision() -> None:
    heading("perception: face landmarker")
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision as mpv
    lm = mpv.FaceLandmarker.create_from_options(mpv.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
        running_mode=mpv.RunningMode.VIDEO,
        output_facial_transformation_matrixes=True, num_faces=1))
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    times = []
    for i in range(60):
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        t0 = time.perf_counter()
        lm.detect_for_video(img, i * 66)
        times.append(1000 * (time.perf_counter() - t0))
    lm.close()
    stat("landmarker 640x480 (no face)", times[5:])
    print("  note: a frame containing a face costs slightly more, see the live run")


def measure_tts() -> None:
    heading("speech out: piper en_US-lessac-medium")
    from piper import PiperVoice
    v = PiperVoice.load("models/piper/en_US-lessac-medium.onnx")
    lines = ["Oh, hello.",
             "There it is, the blue mug on the left.",
             "I had a look around, and I think that is the notebook you asked about."]
    first, rtf = [], []
    for text in lines:
        t0 = time.perf_counter()
        chunks = list(v.synthesize(text))
        dt = time.perf_counter() - t0
        audio = np.concatenate([c.audio_float_array for c in chunks])
        secs = len(audio) / chunks[0].sample_rate
        first.append(1000 * dt)
        rtf.append(dt / secs)
        print(f"  {dt * 1000:6.0f} ms to synthesize {secs:4.2f}s  rtf {dt / secs:.3f}  "
              f"\"{text[:38]}\"")
    stat("synthesis per reply", first)
    stat("real time factor", [r for r in rtf], unit="x")
    print("  the character plays sentence by sentence, so the first number is what the")
    print("  person waits for, not the total for a long answer")


def measure_stt() -> None:
    heading("speech in: faster-whisper base.en int8")
    from faster_whisper import WhisperModel
    from piper import PiperVoice
    v = PiperVoice.load("models/piper/en_US-lessac-medium.onnx")
    chunks = list(v.synthesize("Can you point at the blue mug on the table?"))
    audio = np.concatenate([c.audio_float_array for c in chunks])
    sr = chunks[0].sample_rate
    a16 = np.interp(np.arange(0, len(audio), sr / 16000),
                    np.arange(len(audio)), audio).astype(np.float32)
    secs = len(a16) / 16000
    t0 = time.perf_counter()
    w = WhisperModel("base.en", device="cpu", compute_type="int8")
    load = time.perf_counter() - t0
    times, texts = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        segs, _ = w.transcribe(a16, beam_size=1, language="en",
                               condition_on_previous_text=False)
        texts.append(" ".join(s.text for s in segs).strip())
        times.append(1000 * (time.perf_counter() - t0))
    print(f"  model load {load:.2f}s (once, at startup)")
    print(f"  {secs:.2f}s of speech -> {texts[0]!r}")
    stat("transcribe", times)
    stat("real time factor", [t / 1000 / secs for t in times], unit="x")


async def measure_body() -> dict:
    heading("body: protocol latency, cpu, memory, real time factor")
    env = dict(os.environ)
    proc = subprocess.Popen([PY, "-m", "body.main", "--headless", "--port", str(PORT),
                             "--seconds", "150"], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env)
    ps = psutil.Process(proc.pid)
    await asyncio.sleep(4.0)

    rtt, react = [], []
    cpu, rss = [], []
    sim_elapsed = wall_elapsed = 0.0
    try:
        await _body_loop(PORT, ps, rtt, react, cpu, rss)
    except ConnectionClosed:
        print("  body closed the connection early, reporting what was gathered")
    stat("command to acknowledged in telemetry", rtt)
    stat("command to head on target, idle", [t for t, p in react if p == "idle"])
    stat("command to head on target, alert", [t for t, p in react if p == "alert"])
    if cpu:
        print(f"{'body cpu, share of one core':<34} {cpu[0]:.1f} %")
    stat("body resident memory, process tree", rss, unit="MB")
    if "_RTF" in globals():
        sim_elapsed, wall_elapsed = globals()["_RTF"]
        print(f"{'physics real time factor':<34} {sim_elapsed / wall_elapsed:.4f} "
              f"({sim_elapsed:.1f}s simulated in {wall_elapsed:.1f}s wall)")
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    return {"rtt_ms": rtt, "react_ms": react, "cpu": cpu, "rss_mb": rss}


def tree(ps: psutil.Process) -> list[psutil.Process]:
    out = [ps]
    try:
        out += ps.children(recursive=True)
    except psutil.Error:
        pass
    return out


def cpu_seconds(ps: psutil.Process) -> float:
    total = 0.0
    for p in tree(ps):
        try:
            t = p.cpu_times()
            total += t.user + t.system
        except psutil.Error:
            pass
    return total


def rss_mb(ps: psutil.Process) -> float:
    total = 0.0
    for p in tree(ps):
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total / 1e6


async def _body_loop(PORT, ps, rtt, react, cpu, rss) -> None:
    async with connect(f"ws://127.0.0.1:{PORT}") as ws:
        await ws.recv()
        ps.cpu_percent(None)

        async def latest():
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("type") == P.MSG_STATE:
                    return msg

        s = await latest()
        sim_t0, wall_t0 = s["t"], time.perf_counter()
        cpu_t0 = cpu_seconds(ps)

        for i in range(16):
            if i == 0:
                await ws.send(P.posture("idle"))
            if i == 8:
                await ws.send(P.posture("alert"))
                await asyncio.sleep(0.4)
            az = 0.45 if i % 2 == 0 else -0.45
            t0 = time.perf_counter()
            await ws.send(P.look(az, 0.2))
            while True:
                s = await latest()
                if abs(s["aim_cmd"]["az"] - az) < 0.02:
                    rtt.append(1000 * (time.perf_counter() - t0))
                    break
            while True:
                s = await latest()
                if s["aim_err"] < 0.02:
                    react.append(1000 * (time.perf_counter() - t0))
                    break
            react[-1] = (react[-1], "alert" if i >= 8 else "idle")
            rss.append(rss_mb(ps))
            await asyncio.sleep(0.25)

        s = await latest()
        sim_elapsed = s["t"] - sim_t0
        wall_elapsed = time.perf_counter() - wall_t0
        cpu.append(100.0 * (cpu_seconds(ps) - cpu_t0) / wall_elapsed)
        globals()["_RTF"] = (sim_elapsed, wall_elapsed)


def measure_footprint() -> None:
    heading("deployment footprint")
    print(f"  python {sys.version.split()[0]}  platform {sys.platform}")
    print(f"  logical cpus {psutil.cpu_count()}  "
          f"total memory {psutil.virtual_memory().total / 1e9:.1f} GB")
    for name, path in [("face landmarker", "models/face_landmarker.task"),
                       ("piper voice", "models/piper/en_US-lessac-medium.onnx")]:
        p = pathlib.Path(path)
        if p.exists():
            print(f"  {name:<18} {p.stat().st_size / 1e6:7.1f} MB")
    venv = pathlib.Path(".venv")
    if venv.exists():
        total = sum(f.stat().st_size for f in venv.rglob("*") if f.is_file())
        print(f"  {'virtualenv':<18} {total / 1e6:7.1f} MB")
    print("  whisper base.en int8 is downloaded on first run, about 150 MB on disk")
    print("  target is 4 cores and 8 GB with no gpu, so every model above is cpu only")


async def main() -> None:
    which = sys.argv[1:] or ["vision", "tts", "stt", "body", "footprint"]
    print(f"measuring: {', '.join(which)}")
    if "vision" in which:
        measure_vision()
    if "tts" in which:
        measure_tts()
    if "stt" in which:
        measure_stt()
    if "body" in which:
        await measure_body()
    if "footprint" in which:
        measure_footprint()
    print("\ndone")


if __name__ == "__main__":
    asyncio.run(main())
