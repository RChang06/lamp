from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = 1
DEFAULT_PORT = 8765

CMD_LOOK = "look"
CMD_POSTURE = "posture"
CMD_GESTURE = "gesture"
CMD_LIGHT = "light"
CMD_ENVELOPE = "envelope"
CMD_RESET = "reset"
CMD_HELLO = "hello"

MSG_STATE = "state"
MSG_EVENT = "event"
MSG_HELLO = "hello"


@dataclass
class Command:
    cmd: str
    args: dict[str, Any] = field(default_factory=dict)
    seq: int = 0

    def dumps(self) -> str:
        return json.dumps({"cmd": self.cmd, "seq": self.seq, **self.args})


def look(az: float, el: float, weight: float = 1.0, seq: int = 0) -> str:
    return Command(CMD_LOOK, {"az": round(az, 4), "el": round(el, 4),
                              "weight": round(weight, 3)}, seq).dumps()


def posture(name: str, seq: int = 0) -> str:
    return Command(CMD_POSTURE, {"name": name}, seq).dumps()


def gesture(name: str, amp: float = 1.0, speed: float = 1.0, seq: int = 0) -> str:
    return Command(CMD_GESTURE, {"name": name, "amp": amp, "speed": speed}, seq).dumps()


def light(intensity: float | None = None, warm: float | None = None,
          pulse: float | None = None, seq: int = 0) -> str:
    args = {k: v for k, v in (("intensity", intensity), ("warm", warm), ("pulse", pulse))
            if v is not None}
    return Command(CMD_LIGHT, args, seq).dumps()


def envelope(level: float, seq: int = 0) -> str:
    return Command(CMD_ENVELOPE, {"level": round(level, 3)}, seq).dumps()


def parse(raw: str | bytes) -> dict[str, Any]:
    msg = json.loads(raw)
    if not isinstance(msg, dict):
        raise ValueError("expected a json object")
    return msg
