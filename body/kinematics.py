"""aim solve and posture presets for the supplied five dof lamp.

the chain decouples exactly, measured over the full joint ranges in scratch/aim_probe.py:

    aim azimuth   = base_yaw + neck_yaw
    aim elevation = shoulder_pitch + elbow_pitch - head_pitch

so pointing the head is closed form, no iterative ik and no jacobian. azimuth has one
redundant joint and elevation has two, and that slack is what carries the character:
how much the whole body turns versus just the neck, and how tall or folded it stands
while still looking at the same spot.

note the model has no roll joint, so a head tilt is not available on this body.
"""
from __future__ import annotations

from dataclasses import dataclass

JOINTS = ("base_yaw", "shoulder_pitch", "elbow_pitch", "neck_yaw", "head_pitch")


@dataclass(frozen=True)
class Posture:
    """how the lamp carries itself while aiming wherever the brain asked."""
    name: str
    shoulder: float      # nominal shoulder pitch, sets height and lean
    elbow: float         # nominal elbow pitch, folds the arm
    base_share: float    # fraction of the azimuth taken by the heavy base
    omega: float         # smoothing bandwidth, rad/s, higher is snappier
    breath_amp: float    # idle breathing amplitude, rad
    breath_hz: float
    light: float         # resting light intensity 0..1
    neck_free: float     # azimuth the neck covers alone before the base joins in, rad
    neck_bias: float = 0.0   # standing offset on the neck, stands in for a tilt
    rest_head: float | None = None   # if set, ignore the aim elevation and park the head here
    sway: float = 0.0        # slow idle neck sway amplitude, rad


# shoulder is up positive, elbow folds negative, see the reach envelope in aim_probe
POSTURES = {
    # asleep: leaned back, upper arm folded down, shade facing the desk, barely lit, very slow.
    # parked at the urdf soft limits (shoulder 0.95, elbow -1.70, head 0.60), the tightest fold
    # a real servo should hold, about 33% of shoulder torque. it does not track the aim.
    "sleep":  Posture("sleep",  shoulder=0.95, elbow=-1.70, base_share=1.0, omega=2.5,
                      breath_amp=0.012, breath_hz=0.16, light=0.10, neck_free=0.0,
                      rest_head=0.60),
    # idle: the classic desk lamp stance from the supplied reference picture, lower arm
    # leaning back, upper arm reaching forward, weight over the base, scanning slowly
    "idle":   Posture("idle",   shoulder=0.35, elbow=-1.20, base_share=0.80, omega=4.0,
                      breath_amp=0.050, breath_hz=0.22, light=0.30, neck_free=0.40,
                      sway=0.18),
    # alert: tall, leaning in slightly, bright
    "alert":  Posture("alert",  shoulder=0.30,  elbow=-0.70, base_share=0.55, omega=9.0,
                      breath_amp=0.006, breath_hz=0.30, light=0.85, neck_free=0.30),
    # listening: leaned in toward the speaker (head ~0.18 m closer than alert), neck turned
    # 20 deg as if offering an ear, held nearly still. shoulder + elbow stays at -0.80 so a
    # face slightly above the lamp is reachable without the solver unfolding the arm
    "listen": Posture("listen", shoulder=-0.25, elbow=-0.55, base_share=0.45, omega=6.5,
                      breath_amp=0.004, breath_hz=0.26, light=0.60, neck_free=0.35, neck_bias=0.35),
    # speaking: halfway between idle and alert, relaxed but engaged, the audio envelope
    # adds the motion
    "speak":  Posture("speak",  shoulder=0.33,  elbow=-0.95, base_share=0.50, omega=8.0,
                      breath_amp=0.005, breath_hz=0.28, light=0.75, neck_free=0.30),
    # searching the scene for the goal, tall so the camera clears the desk
    "search": Posture("search", shoulder=0.45,  elbow=-0.55, base_share=0.35, omega=7.0,
                      breath_amp=0.005, breath_hz=0.24, light=0.95, neck_free=0.20),
}


class AimSolver:
    """turns an aim request plus a posture into joint targets, inside the urdf limits."""

    def __init__(self, model):
        self.lim = {}
        for name in JOINTS:
            lo, hi = model.actuator(name).ctrlrange
            self.lim[name] = (float(lo), float(hi))

    def clamp(self, name: str, value: float) -> float:
        lo, hi = self.lim[name]
        return lo if value < lo else hi if value > hi else value

    def solve(self, az: float, el: float, p: Posture) -> tuple[dict[str, float], float, float]:
        """joint targets for this aim, plus the azimuth and elevation actually achievable."""
        if p.rest_head is not None:
            # parked posture: whole body faces az, neck centred, arm and head fixed
            base = self.clamp("base_yaw", az)
            sh, eb, head = p.shoulder, p.elbow, self.clamp("head_pitch", p.rest_head)
            return ({"base_yaw": base, "shoulder_pitch": sh, "elbow_pitch": eb,
                     "neck_yaw": 0.0, "head_pitch": head}, base, sh + eb - head)

        # azimuth: the light neck covers the first neck_free radians by itself, which is
        # what a glance looks like, and only past that does the heavy base start to turn.
        sign = 1.0 if az >= 0 else -1.0
        excess = max(0.0, abs(az) - p.neck_free)
        base = self.clamp("base_yaw", sign * excess * p.base_share)
        neck = self.clamp("neck_yaw", az - base + p.neck_bias)
        base = self.clamp("base_yaw", az - (neck - p.neck_bias))
        got_az = base + neck - p.neck_bias

        # elevation: hold the posture's arm shape and pitch the head, then spill any
        # overflow into the elbow and shoulder so the aim survives extreme targets
        sh, eb = p.shoulder, p.elbow
        head = self.clamp("head_pitch", sh + eb - el)
        owed = el - (sh + eb - head)
        if abs(owed) > 1e-6:
            new_eb = self.clamp("elbow_pitch", eb + owed)
            owed -= new_eb - eb
            eb = new_eb
        if abs(owed) > 1e-6:
            new_sh = self.clamp("shoulder_pitch", sh + owed)
            owed -= new_sh - sh
            sh = new_sh
        got_el = sh + eb - head

        return ({"base_yaw": base, "shoulder_pitch": sh, "elbow_pitch": eb,
                 "neck_yaw": neck, "head_pitch": head}, got_az, got_el)
