"""wire protocol between the character (brain) and the simulated body.

newline delimited json over a websocket. the brain sends character intent, never joint
trajectories: it says where to look and how to carry itself, and the body owns the
kinematics, the servo limits and every bit of timing. that boundary is deliberate, the
body stays runnable and testable with no perception stack attached.

angles are radians in the body frame: azimuth is ccw from the lamp's forward direction
(positive turns to its own left), elevation is positive up.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = 1
DEFAULT_PORT = 8765

# brain -> body
CMD_LOOK = "look"          # az, el, optional weight for partial attention
CMD_POSTURE = "posture"    # name of a posture preset
CMD_GESTURE = "gesture"    # name, amp, speed: additive overlay, body reports completion
CMD_LIGHT = "light"        # intensity, warm, pulse
CMD_ENVELOPE = "envelope"  # level 0..1, streamed while speaking
CMD_RESET = "reset"
CMD_HELLO = "hello"

# body -> brain
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
