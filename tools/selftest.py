from __future__ import annotations

import asyncio
import json
import sys
import time

sys.path.insert(0, ".")

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve as ws_serve

from body.main import Body
from common import protocol as P

PORT = 8799
RUN = 20.0


class Recorder:
    def __init__(self):
        self.states: list[dict] = []
        self.events: list[dict] = []

    async def pump(self, ws):
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == P.MSG_STATE:
                msg["wall"] = time.perf_counter()
                self.states.append(msg)
            elif msg.get("type") == P.MSG_EVENT:
                self.events.append(msg)

    def since(self, t0: float) -> list[dict]:
        return [s for s in self.states if s["wall"] >= t0]

    def settle(self, t0: float, tol: float = 0.02) -> float | None:
        rows = self.since(t0)
        for i, s in enumerate(rows):
            if s["aim_err"] <= tol and all(r["aim_err"] <= tol for r in rows[i:i + 10]):
                return s["wall"] - t0
        return None


async def main() -> None:
    body = Body(view=False)
    rec = Recorder()
    async with ws_serve(body.serve, "127.0.0.1", PORT):
        runner = asyncio.create_task(body.run(seconds=RUN))
        async with connect(f"ws://127.0.0.1:{PORT}") as ws:
            hello = json.loads(await ws.recv())
            print("hello:", hello["postures"], hello["gestures"])
            pump = asyncio.create_task(rec.pump(ws))

            await asyncio.sleep(1.0)
            idle = rec.states[-1]
            print(f"\nidle: posture={idle['posture']} aim={idle['aim']} light={idle['light']}"
                  f" settled={idle['settled']}")
            breath = [s["q"]["shoulder_pitch"] for s in rec.states[-20:]]
            print(f"breathing peak to peak on the shoulder: {max(breath) - min(breath):.4f} rad")

            await ws.send(P.posture("alert"))
            await ws.send(P.gesture("perk"))
            t0 = time.perf_counter()
            await ws.send(P.look(0.60, 0.20))
            await asyncio.sleep(2.5)
            s = rec.states[-1]
            peak = max(abs(x["aim"]["az"]) for x in rec.since(t0))
            print(f"\ngaze step to az=+0.600 el=+0.200")
            print(f"  settle to 0.02 rad: {rec.settle(t0)}")
            print(f"  final aim {s['aim']}  err {s['aim_err']}  overshoot {peak - 0.6:+.4f} rad")
            print(f"  joints {s['q']}")

            await ws.send(P.look(0.15, 0.0))
            await asyncio.sleep(1.5)
            s = rec.states[-1]
            print(f"\nsmall glance az=0.15 -> base {s['q']['base_yaw']:+.3f} "
                  f"neck {s['q']['neck_yaw']:+.3f} (base takes {100 * s['q']['base_yaw'] / 0.15:.0f}%)")

            before = rec.states[-1]["q"]["head_pitch"]
            await ws.send(P.gesture("nod"))
            await asyncio.sleep(0.35)
            mid = rec.states[-1]["q"]["head_pitch"]
            await asyncio.sleep(0.9)
            after = rec.states[-1]["q"]["head_pitch"]
            print(f"\nnod: head_pitch {before:+.3f} -> {mid:+.3f} -> {after:+.3f}  "
                  f"events {[e['ev'] for e in rec.events]}")

            await ws.send(P.posture("speak"))
            lights = []
            for i in range(60):
                level = 0.5 + 0.5 * (i % 12) / 12
                await ws.send(P.envelope(level))
                await asyncio.sleep(1 / 30)
                if rec.states:
                    lights.append(rec.states[-1]["light"])
            await ws.send(P.envelope(0.0))
            print(f"\nspeech envelope: light swung {min(lights):.2f} to {max(lights):.2f}")

            await ws.send(P.posture("search"))
            await ws.send(P.look(-1.2, -0.9))
            await asyncio.sleep(3.0)
            s = rec.states[-1]
            print(f"\nedge of workspace az=-1.200 el=-0.900")
            print(f"  achieved {s['aim']}  err {s['aim_err']}  saturated {s['saturated']}")
            print(f"  joints {s['q']}")

            await ws.send("{not json")
            await asyncio.sleep(0.3)
            await ws.send(P.gesture("does_not_exist"))
            await asyncio.sleep(0.3)
            print(f"\nafter bad input, body still reporting: {rec.states[-1]['t']}")

            pump.cancel()
        await runner

    n = len(rec.states)
    span = rec.states[-1]["wall"] - rec.states[0]["wall"]
    print(f"\ntelemetry {n} frames over {span:.1f}s = {n / span:.1f} hz "
          f"(asked for 30), sim time {rec.states[-1]['t']:.1f}s")
    print(f"warnings {rec.states[-1]['warnings']}")
    print(f"events: {[e.get('ev') for e in rec.events]}")


if __name__ == "__main__":
    asyncio.run(main())
