from __future__ import annotations

from dataclasses import dataclass

JOINTS = ("base_yaw", "shoulder_pitch", "elbow_pitch", "neck_yaw", "head_pitch")
EL_REF = 0.22
SH_FLOOR = -0.45
SH_SOFT_MAX = 0.95
EB_SOFT_MIN = -1.70
BEND_SH = 2.0


@dataclass(frozen=True)
class Posture:
    name: str
    shoulder: float
    elbow: float
    base_share: float
    omega: float
    breath_amp: float
    breath_hz: float
    light: float
    neck_free: float
    neck_bias: float = 0.0
    rest_head: float | None = None
    sway: float = 0.0
    el_share: float = 0.0
    el_bias: float = 0.0


POSTURES = {
    "sleep":  Posture("sleep",  shoulder=0.95, elbow=-1.70, base_share=1.0, omega=2.5,
                      breath_amp=0.012, breath_hz=0.16, light=0.10, neck_free=0.0,
                      rest_head=0.60),
    "idle":   Posture("idle",   shoulder=0.35, elbow=-1.20, base_share=0.80, omega=4.0,
                      breath_amp=0.050, breath_hz=0.22, light=0.30, neck_free=0.40,
                      sway=0.18,
                      el_share=0.70),
    "alert":  Posture("alert",  shoulder=0.55,  elbow=-1.20, base_share=0.55, omega=9.0,
                      breath_amp=0.006, breath_hz=0.30, light=0.85, neck_free=0.30,
                      el_share=0.70),
    "listen": Posture("listen", shoulder=-0.25, elbow=-0.55, base_share=0.45, omega=6.5,
                      breath_amp=0.004, breath_hz=0.26, light=0.60, neck_free=0.35, neck_bias=0.35,
                      el_share=0.60),
    "speak":  Posture("speak",  shoulder=0.33,  elbow=-0.95, base_share=0.50, omega=8.0,
                      breath_amp=0.005, breath_hz=0.28, light=0.75, neck_free=0.30,
                      el_share=0.70),
    "think":  Posture("think",  shoulder=0.70,  elbow=-1.30, base_share=0.50, omega=5.0,
                      breath_amp=0.020, breath_hz=0.20, light=0.55, neck_free=0.35,
                      neck_bias=-0.30, el_share=0.40, el_bias=0.25),
    "search": Posture("search", shoulder=0.45,  elbow=-0.55, base_share=0.35, omega=7.0,
                      breath_amp=0.005, breath_hz=0.24, light=0.95, neck_free=0.20,
                      el_share=0.70),
}


class AimSolver:
    def __init__(self, model):
        self.lim = {}
        for name in JOINTS:
            lo, hi = model.actuator(name).ctrlrange
            self.lim[name] = (float(lo), float(hi))

    def clamp(self, name: str, value: float) -> float:
        lo, hi = self.lim[name]
        return lo if value < lo else hi if value > hi else value

    def solve(self, az: float, el: float, p: Posture) -> tuple[dict[str, float], float, float]:
        if p.rest_head is not None:
            base = self.clamp("base_yaw", az)
            sh, eb, head = p.shoulder, p.elbow, self.clamp("head_pitch", p.rest_head)
            return ({"base_yaw": base, "shoulder_pitch": sh, "elbow_pitch": eb,
                     "neck_yaw": 0.0, "head_pitch": head}, base, sh + eb - head)

        sign = 1.0 if az >= 0 else -1.0
        excess = max(0.0, abs(az) - p.neck_free)
        base = self.clamp("base_yaw", sign * excess * p.base_share)
        neck = self.clamp("neck_yaw", az - base + p.neck_bias)
        base = self.clamp("base_yaw", az - (neck - p.neck_bias))
        got_az = base + neck - p.neck_bias

        el += p.el_bias
        sh, eb = p.shoulder, p.elbow
        if p.el_share > 0:
            # knee bend: shoulder leans back while the elbow folds forward
            delta = p.el_share * (el - EL_REF)
            sh = min(max(sh - BEND_SH * delta, SH_FLOOR), SH_SOFT_MAX)
            eb = min(max(eb + (BEND_SH + 1.0) * delta, EB_SOFT_MIN), self.lim["elbow_pitch"][1])
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
