from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading

sys.path.insert(0, ".")

from websockets.asyncio.client import connect

from body.kinematics import POSTURES
from body.motion import GESTURES
from common import protocol as P

HELP = """
commands
  <posture>            one of: """ + ", ".join(sorted(POSTURES)) + """
  <gesture>            one of: """ + ", ".join(sorted(GESTURES)) + """
  look <az> <el>       aim in radians, az positive is its own left, el positive is up
  l <az> <el>          same thing
  light <0..1> [warm]  brightness, and optionally warmth 0 cool to 1 warm
  pulse <0..1>         slow breathing of the light
  env <0..1>           pretend it is speaking at this level
  say <text>           fake a spoken line, drives the envelope from the text length
  state                print the latest telemetry
  watch                stream telemetry until you press enter
  showcase             run the scripted tour
  help, quit
"""

SHOWCASE: list[tuple[float, str | None]] = [
    (0.0, "sleep"), (1.5, "light 0.1"),
    (2.0, None),
    (4.0, "alert"), (4.0, "perk"), (4.1, "light 0.9"),
    (5.5, "look 0.5 0.25"), (7.0, "look -0.5 0.25"), (8.5, "look 0.0 0.3"),
    (10.0, "nod"), (11.2, "shake"), (12.5, "nod2"),
    (14.0, "listen"), (14.2, "look 0.3 0.2"),
    (16.0, "speak"), (16.1, "say this is roughly what it looks like when I am talking"),
    (21.0, "search"), (21.2, "scan"), (24.8, "look 0.0 -0.5"),
    (26.0, "idle"), (26.2, "droop"), (27.5, "light 0.3"),
    (29.0, "startle"), (30.0, "alert"), (30.1, "flash"),
    (32.0, "idle"),
]


class Puppet:
    def __init__(self, url: str):
        self.url = url
        self.state: dict = {}
        self.ws = None
        self.show = False

    async def pump(self, ws) -> None:
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == P.MSG_STATE:
                self.state = msg
                if self.show:
                    q = " ".join(f"{k.split('_')[0]}{v:+.2f}" for k, v in msg["q"].items())
                    print(f"\r{msg['posture']:7s} aim {msg['aim']['az']:+.2f},"
                          f"{msg['aim']['el']:+.2f} err {msg['aim_err']:.3f} "
                          f"light {msg['light']:.2f}  {q}   ", end="", flush=True)
            elif msg.get("type") == P.MSG_EVENT and msg.get("ev") != "gesture_done":
                print(f"\n[body] {msg}")

    async def do(self, line: str) -> bool:
        parts = line.split()
        if not parts:
            return True
        cmd, rest = parts[0].lower(), parts[1:]
        if cmd in ("quit", "exit", "q"):
            return False
        if cmd in ("help", "?"):
            print(HELP)
        elif cmd in POSTURES:
            await self.ws.send(P.posture(cmd))
        elif cmd in GESTURES:
            amp = float(rest[0]) if rest else 1.0
            await self.ws.send(P.gesture(cmd, amp))
        elif cmd in ("look", "l") and len(rest) >= 2:
            await self.ws.send(P.look(float(rest[0]), float(rest[1])))
        elif cmd == "light" and rest:
            kw = {"intensity": float(rest[0])}
            if len(rest) > 1:
                kw["warm"] = float(rest[1])
            await self.ws.send(P.light(**kw))
        elif cmd == "pulse" and rest:
            await self.ws.send(P.light(pulse=float(rest[0])))
        elif cmd == "env" and rest:
            await self.ws.send(P.envelope(float(rest[0])))
        elif cmd == "say":
            await self.fake_speech(" ".join(rest) or "hello")
        elif cmd == "state":
            print(json.dumps(self.state, indent=2))
        elif cmd == "watch":
            self.show = True
            await asyncio.get_running_loop().run_in_executor(None, input)
            self.show = False
            print()
        elif cmd == "showcase":
            await self.showcase()
        else:
            print(f"? {line}   (try help)")
        return True

    async def fake_speech(self, text: str) -> None:
        import math
        import random
        await self.ws.send(P.posture("speak"))
        dur = 0.35 + 0.055 * len(text)
        steps = int(dur * 30)
        for i in range(steps):
            t = i / 30
            level = max(0.0, 0.55 + 0.45 * math.sin(2 * math.pi * 4.0 * t)) * \
                (0.3 if (int(t * 1.6) % 5 == 4) else 1.0) * random.uniform(0.85, 1.0)
            await self.ws.send(P.envelope(min(1.0, level)))
            await asyncio.sleep(1 / 30)
        await self.ws.send(P.envelope(0.0))

    async def showcase(self) -> None:
        print("showcase: watch the lamp. 33 seconds.")
        last = 0.0
        for at, cmd in SHOWCASE:
            await asyncio.sleep(max(0.0, at - last))
            last = at
            if cmd is None:
                continue
            print(f"  {at:5.1f}s  {cmd}")
            if cmd.startswith("say "):
                asyncio.create_task(self.fake_speech(cmd[4:]))
            else:
                await self.do(cmd)
        print("showcase done")

    async def run(self, showcase: bool) -> None:
        async with connect(self.url) as ws:
            self.ws = ws
            hello = json.loads(await ws.recv())
            print(f"connected to body, protocol v{hello['version']}")
            pump = asyncio.create_task(self.pump(ws))
            if showcase:
                await self.showcase()
            else:
                print(HELP)
                loop = asyncio.get_running_loop()
                while True:
                    line = await loop.run_in_executor(None, lambda: input("lamp> "))
                    if not await self.do(line):
                        break
            pump.cancel()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default=f"ws://127.0.0.1:{P.DEFAULT_PORT}")
    ap.add_argument("--showcase", action="store_true", help="run the tour and exit")
    args = ap.parse_args()
    try:
        await Puppet(args.body).run(args.showcase)
    except OSError:
        print(f"no body at {args.body}. start it first:\n  python -m body.main")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        pass
