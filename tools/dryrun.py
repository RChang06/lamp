from __future__ import annotations

import asyncio
import pathlib
import sys
import time
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, ".")

from websockets.asyncio.server import serve as ws_serve

from body.main import Body
from brain.director import ENGAGED, IDLE, Director
from brain.link import BodyLink
from brain.memory import SceneMemory
from brain.planner import Planner
from common import protocol as P

PORT = 8798
MEM = "run/dryrun_memory.jsonl"


@dataclass
class FakeLook:
    present: bool = False
    attending: bool = False
    az: float = 0.0
    el: float = 0.22
    yaw: float = 0.0
    pitch: float = 0.0
    size: float = 0.0
    fps: float = 15.0
    latency_ms: float = 3.1
    t: float = 0.0


class FakeVision:
    def __init__(self):
        self.look = FakeLook()
        self.frames = 0

    def latest_frame(self):
        self.frames += 1
        return np.zeros((48, 64, 3), dtype=np.uint8), time.time()

    def close(self):
        pass


@dataclass
class FakeUtterance:
    text: str
    seconds: float = 1.4
    rms: float = 0.04
    audio: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))


class FakeEars:
    def __init__(self):
        self.q: list[FakeUtterance] = []
        self.speaking = False
        self.enabled = True
        self.boost = 1.0
        self.level = 0.0

    def push(self, text: str):
        self.q.append(FakeUtterance(text))

    def next_utterance(self, timeout: float = 0.0):
        return self.q.pop(0) if self.q else None

    def drain(self):
        self.q.clear()

    def close(self):
        pass


class FakeSTT:
    def __init__(self):
        self.out: list[dict] = []

    def submit(self, utt):
        self.out.append({"text": utt.text, "utterance": utt, "stt_s": 0.31,
                         "rtf": 0.22})

    def poll(self):
        return self.out.pop(0) if self.out else None


class FakeVoice:
    def __init__(self):
        self.said: list[str] = []
        self.until = 0.0
        self.first_sound_ms: list[float] = []
        self.synth_times: list[tuple[float, float]] = []

    def say(self, text: str):
        self.said.append(text)
        dur = 0.35 + 0.045 * len(text)
        self.until = max(self.until, time.time()) + dur
        self.first_sound_ms.append(120.0)
        self.synth_times.append((0.08 * dur, dur))

    def busy(self):
        return time.time() < self.until

    def cancel(self):
        self.until = 0.0


class NullAudio:
    def play(self, *a, **k):
        pass

    def stop(self, *a, **k):
        pass

    def busy(self, *a, **k):
        return False

    def close(self):
        pass


async def main() -> None:
    pathlib.Path(MEM).unlink(missing_ok=True)
    body = Body(view=False)
    link = BodyLink(f"ws://127.0.0.1:{PORT}")
    vision, ears, stt, voice, audio = FakeVision(), FakeEars(), FakeSTT(), FakeVoice(), NullAudio()
    memory = SceneMemory(MEM)
    director = Director(link, vision, ears, stt, voice, audio, memory,
                        Planner(memory, cloud=None), cloud=None, quiet=True)

    async with ws_serve(body.serve, "127.0.0.1", PORT):
        asyncio.create_task(body.run(seconds=70))
        asyncio.create_task(link.run())
        await asyncio.sleep(0.6)
        assert link.connected, "the brain never reached the body"
        print("body connected, protocol version", link.hello.get("version"))

        marks: list[tuple[float, str]] = []

        async def script():
            await asyncio.sleep(1.0)
            print("\n-- person appears and looks at it")
            vision.look = FakeLook(present=True, attending=True, az=0.25, el=0.30, size=0.18)
            await asyncio.sleep(4.0)

            print("\n-- person asks what it can see")
            ears.push("what do you see there")
            await asyncio.sleep(2.5)
            print("-- person answers the question")
            ears.push("a blue mug")
            await asyncio.sleep(6.0)

            print("\n-- later, person asks what it saw")
            ears.push("what did you see earlier")
            await asyncio.sleep(6.0)

            print("\n-- person gives it a goal")
            ears.push("can you point at the mug")
            await asyncio.sleep(14.0)

            print("\n-- person looks away")
            vision.look = FakeLook(present=True, attending=False, az=0.6, el=0.2, size=0.18,
                                   yaw=55.0)
            await asyncio.sleep(3.0)

        runner = asyncio.create_task(script())
        while not runner.done():
            await director.tick()
            marks.append((time.time() - director.t0, director.state))
            await asyncio.sleep(0.05)
        for _ in range(40):
            await director.tick()
            await asyncio.sleep(0.05)

    states = [s for _, s in marks]
    seen = []
    for s in states:
        if not seen or seen[-1] != s:
            seen.append(s)
    print("\nstate path:", " -> ".join(seen))
    print("said:")
    for line in voice.said:
        print("   ", line)
    print("memory:", memory.describe())

    problems = []
    if ENGAGED not in seen:
        problems.append("never engaged")
    if "listening" not in seen and "thinking" not in seen:
        problems.append("never listened")
    if "speaking" not in seen:
        problems.append("never spoke")
    if seen[-1] != IDLE:
        problems.append(f"did not return to idle, ended {seen[-1]}")
    if not any("mug" in m.label for m in memory.items):
        problems.append("did not remember the mug")
    if not any("mug" in s.lower() for s in voice.said):
        problems.append("never mentioned the mug when asked")
    plans = [l for _, l in director.log if l.startswith("plan ")]
    if not plans:
        problems.append("no plan was produced for the goal")
    observes = [l for _, l in director.log if "run observe" in l]
    if len(observes) < 2:
        problems.append(f"expected an observe in the plan as well, saw {len(observes)}")

    print("\nchecks:", "all passed" if not problems else "FAILED")
    for p in problems:
        print("  -", p)
    print(f"body link: {link.sent} sent, {link.dropped} dropped")
    print(f"last telemetry: aim_err {link.aim_err}, saturated {link.saturated}, "
          f"posture {link.state.get('posture')}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
