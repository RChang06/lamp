"""the character process. starts the senses, connects to the body, runs the director.

two processes on purpose. the body is a hard real time thing: it integrates physics and
holds servo references on a clock. the brain is a soft real time thing full of models that
occasionally take half a second. keeping them apart means a slow transcription cannot
stutter the motion, either one can be restarted without the other, and the body can be
driven by hand for testing. the cost is one websocket and a serialisation step, measured
in the write up.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

sys.path.insert(0, ".")

from brain.audio import AudioOut
from brain.cloud import Cloud
from brain.director import Director
from brain.ears import Ears, Transcriber
from brain.link import BodyLink
from brain.memory import SceneMemory
from brain.planner import Planner
from brain.vision import Vision
from brain.voice import Voice
from common import protocol as P

TICK = 1 / 20.0


def summary(d: Director, link: BodyLink, vision: Vision, voice: Voice, cloud: Cloud) -> None:
    print("\n---- session summary ----")
    print(f"ran {time.time() - d.t0:.0f}s, ended in state {d.state}")
    print(f"vision: {vision.frames} frames, landmarker {vision.look.latency_ms:.1f} ms/frame, "
          f"{vision.look.fps:.1f} fps")
    if voice.first_sound_ms:
        print(f"time to first sound: {min(voice.first_sound_ms):.0f} to "
              f"{max(voice.first_sound_ms):.0f} ms over {len(voice.first_sound_ms)} replies")
    if voice.synth_times:
        tot_s = sum(s for s, _ in voice.synth_times)
        tot_a = sum(a for _, a in voice.synth_times)
        print(f"tts: {tot_s:.2f}s of compute for {tot_a:.2f}s of speech "
              f"(rtf {tot_s / max(tot_a, 1e-3):.3f})")
    for kind in ("stt", "respond", "observe"):
        times = [t for k, t in d.latency if k == kind]
        if times:
            print(f"{kind}: n={len(times)} mean {sum(times) / len(times):.2f}s "
                  f"max {max(times):.2f}s")
    if cloud.calls:
        for kind in ("observe", "plan", "reply"):
            n = len([1 for k, _ in cloud.calls if k == kind])
            if n:
                print(f"cloud {kind}: n={n} mean {cloud.latency(kind):.2f}s")
    print(f"body link: {link.sent} sent, {link.dropped} dropped, connected={link.connected}")
    print(f"heard: {d.heard}")
    print(f"memory: {d.memory.describe()}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default=f"ws://127.0.0.1:{P.DEFAULT_PORT}")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--mirror", action="store_true", help="camera already mirrors the image")
    ap.add_argument("--quiet", action="store_true", help="no music bed")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long")
    ap.add_argument("--whisper", default="base.en")
    args = ap.parse_args()

    link = BodyLink(args.body)
    asyncio.create_task(link.run())

    audio = AudioOut(on_level=link.envelope)
    audio.start()
    voice = Voice(audio)
    ears = Ears()
    ears.start()
    print("loading whisper", flush=True)
    stt = Transcriber(args.whisper)
    vision = Vision(cam_index=args.cam, mirror=args.mirror)
    vision.start()

    memory = SceneMemory()
    cloud = Cloud()
    print(f"vision backend: {'cloud ' + cloud.model if cloud.available else 'none, will ask'}"
          + (f" ({cloud.reason})" if not cloud.available and cloud.reason else ""), flush=True)
    director = Director(link, vision, ears, stt, voice, audio, memory,
                        Planner(memory, cloud), cloud, quiet=args.quiet)

    # wait briefly for the body, but run without it rather than refusing to start
    for _ in range(20):
        if link.connected:
            break
        await asyncio.sleep(0.1)
    print(f"body {'connected' if link.connected else 'NOT connected, running headless'}",
          flush=True)
    director.note("awake")

    try:
        while True:
            t0 = time.perf_counter()
            await director.tick()
            if args.seconds and time.time() - director.t0 > args.seconds:
                break
            await asyncio.sleep(max(0.0, TICK - (time.perf_counter() - t0)))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        summary(director, link, vision, voice, cloud)
        vision.close()
        ears.close()
        audio.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
