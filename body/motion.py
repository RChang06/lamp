"""reference generator: posture, gestures, breathing and speech motion, rate limited.

two layers of physical limits, on purpose. this module produces a reference the real
hardware could follow, capped at the urdf velocity limits and smoothed by a critically
damped filter so there is no step input. the mujoco position servos then track that
reference under the urdf effort limits, and whatever they cannot hold shows up as
saturation in the telemetry rather than being silently clipped here.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from body.kinematics import JOINTS


def urdf_velocity_limits(path: str) -> dict[str, float]:
    """rad/s per joint, straight from the supplied urdf."""
    root = ET.parse(path).getroot()
    out = {}
    for j in root.findall("joint"):
        lim = j.find("limit")
        if lim is not None and lim.get("velocity"):
            out[j.get("name").replace("_joint", "")] = float(lim.get("velocity"))
    return out


def _hann(u: float) -> float:
    return 0.5 - 0.5 * math.cos(2 * math.pi * u)


# gesture library. each entry is (duration, offsets(u, amp) -> joint deltas, light(u, amp)).
# offsets are added on top of the aim solution, so the lamp keeps looking where it was
# looking while the gesture plays. positive head_pitch looks down.
def _nod(u, amp):
    return {"head_pitch": 0.26 * amp * math.sin(math.pi * u)}


def _nod2(u, amp):
    return {"head_pitch": 0.20 * amp * math.sin(2 * math.pi * u) * _hann(u)}


def _shake(u, amp):
    return {"neck_yaw": 0.22 * amp * math.sin(3 * math.pi * u) * _hann(u)}


def _perk(u, amp):
    s = math.sin(math.pi * u)
    return {"shoulder_pitch": 0.14 * amp * s, "head_pitch": -0.12 * amp * s}


def _droop(u, amp):
    s = math.sin(math.pi * u)
    return {"shoulder_pitch": -0.20 * amp * s, "head_pitch": 0.18 * amp * s}


def _startle(u, amp):
    s = math.sin(math.pi * u) ** 0.5
    return {"shoulder_pitch": 0.18 * amp * s, "head_pitch": -0.16 * amp * s,
            "neck_yaw": 0.05 * amp * math.sin(6 * math.pi * u)}


def _scan(u, amp):
    return {"neck_yaw": 0.62 * amp * math.sin(2 * math.pi * u)}


GESTURES = {
    "nod":     (0.70, _nod,     lambda u, a: 1.0),
    "nod2":    (1.10, _nod2,    lambda u, a: 1.0),
    "shake":   (0.90, _shake,   lambda u, a: 1.0),
    "perk":    (0.55, _perk,    lambda u, a: 1.0 + 0.55 * a * math.sin(math.pi * u)),
    "droop":   (1.30, _droop,   lambda u, a: 1.0 - 0.45 * a * math.sin(math.pi * u)),
    "startle": (0.45, _startle, lambda u, a: 1.0 + 0.8 * a * math.sin(math.pi * u)),
    "scan":    (3.20, _scan,    lambda u, a: 1.0),
    "flash":   (0.40, lambda u, a: {}, lambda u, a: 1.0 + 0.9 * a * math.sin(math.pi * u)),
}


@dataclass
class _Active:
    name: str
    amp: float
    speed: float
    t: float = 0.0


@dataclass
class Reference:
    """per joint position reference with critically damped smoothing and rate limits."""
    vel_limit: dict[str, float]
    pos: dict[str, float] = field(default_factory=dict)
    vel: dict[str, float] = field(default_factory=dict)

    def start(self, targets: dict[str, float]) -> None:
        self.pos = dict(targets)
        self.vel = {k: 0.0 for k in targets}

    def step(self, targets: dict[str, float], omega: float, dt: float) -> dict[str, float]:
        for name in JOINTS:
            err = targets[name] - self.pos[name]
            acc = omega * omega * err - 2.0 * omega * self.vel[name]
            v = self.vel[name] + acc * dt
            cap = self.vel_limit.get(name, 1.0)
            self.vel[name] = max(-cap, min(cap, v))
            self.pos[name] += self.vel[name] * dt
        return self.pos


class MotionMixer:
    """composes posture, breathing, gestures and speech motion into one joint reference."""

    def __init__(self, urdf: str):
        self.reference = Reference(urdf_velocity_limits(urdf))
        self.active: list[_Active] = []
        self.done: list[str] = []
        self.t = 0.0
        self.env = 0.0          # smoothed speech envelope
        self.light_mul = 1.0

    def fire(self, name: str, amp: float = 1.0, speed: float = 1.0) -> bool:
        if name not in GESTURES:
            return False
        self.active = [g for g in self.active if g.name != name]
        self.active.append(_Active(name, max(0.0, min(2.0, amp)), max(0.25, min(3.0, speed))))
        return True

    def feed_envelope(self, level: float, dt: float) -> None:
        # fast attack, slower release, so the body leads the syllable and settles after it
        tau = 0.035 if level > self.env else 0.11
        self.env += (level - self.env) * min(1.0, dt / tau)

    def advance(self, dt: float, breath_amp: float, breath_hz: float,
                sway: float = 0.0) -> dict[str, float]:
        """gesture plus breathing plus speech offsets for this tick, and the light factor."""
        self.t += dt
        off = {k: 0.0 for k in JOINTS}
        mul = 1.0

        # breathing: never completely still, this is most of the "alive" read. the elbow
        # moves against the shoulder so the head rises and falls instead of rocking
        b = math.sin(2 * math.pi * breath_hz * self.t)
        off["shoulder_pitch"] += breath_amp * b
        off["elbow_pitch"] -= 0.5 * breath_amp * b
        off["head_pitch"] += 0.4 * breath_amp * math.sin(2 * math.pi * breath_hz * self.t + 0.6)

        # idle sway: two slow incommensurate sines on the neck, reads as glancing around
        # rather than a metronome
        if sway > 0:
            off["neck_yaw"] += sway * (0.7 * math.sin(2 * math.pi * 0.071 * self.t)
                                       + 0.3 * math.sin(2 * math.pi * 0.173 * self.t + 1.3))

        # speech: the body moves with the voice, same signal drives the light
        if self.env > 1e-3:
            off["head_pitch"] += -0.075 * self.env
            off["shoulder_pitch"] += 0.040 * self.env
            off["neck_yaw"] += 0.018 * self.env * math.sin(2 * math.pi * 1.7 * self.t)
            mul *= 1.0 + 0.55 * self.env

        self.done.clear()
        for g in list(self.active):
            dur, fn, lightfn = GESTURES[g.name]
            g.t += dt * g.speed
            u = g.t / dur
            if u >= 1.0:
                self.active.remove(g)
                self.done.append(g.name)
                continue
            for k, v in fn(u, g.amp).items():
                off[k] += v
            mul *= lightfn(u, g.amp)

        self.light_mul = mul
        return off
