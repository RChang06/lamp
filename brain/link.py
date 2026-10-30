from __future__ import annotations

import asyncio
import json
import time

from websockets.asyncio.client import connect

from common import protocol as P


class BodyLink:
    def __init__(self, url: str = f"ws://127.0.0.1:{P.DEFAULT_PORT}"):
        self.url = url
        self.ws = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.state: dict = {}
        self.hello: dict = {}
        self.connected = False
        self.last_env = 0.0
        self.quiet_env = False
        self.sent = 0
        self.dropped = 0
        self.rtt_ms: list[float] = []

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        while True:
            try:
                async with connect(self.url, ping_interval=20) as ws:
                    self.ws = ws
                    self.connected = True
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("type") == P.MSG_STATE:
                            self.state = msg
                        elif msg.get("type") == P.MSG_HELLO:
                            self.hello = msg
            except Exception:
                pass
            self.connected = False
            self.ws = None
            await asyncio.sleep(0.5)

    def send(self, raw: str) -> None:
        if self.ws is None or self.loop is None:
            self.dropped += 1
            return
        self.sent += 1
        self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._send(raw)))

    async def _send(self, raw: str) -> None:
        try:
            await self.ws.send(raw)
        except Exception:
            self.dropped += 1

    def look(self, az: float, el: float, weight: float = 1.0) -> None:
        self.send(P.look(az, el, weight))

    def posture(self, name: str) -> None:
        self.send(P.posture(name))

    def gesture(self, name: str, amp: float = 1.0, speed: float = 1.0) -> None:
        self.send(P.gesture(name, amp, speed))

    def light(self, **kw) -> None:
        self.send(P.light(**kw))

    def envelope(self, level: float) -> None:
        now = time.perf_counter()
        if now - self.last_env < 0.030:
            return
        if level <= 0.002 and self.quiet_env:
            return
        self.quiet_env = level <= 0.002
        self.last_env = now
        self.send(P.envelope(level))

    @property
    def saturated(self) -> list[str]:
        return self.state.get("saturated", [])

    @property
    def aim_err(self) -> float:
        return float(self.state.get("aim_err", 0.0))
