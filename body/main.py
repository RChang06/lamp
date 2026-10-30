from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time

import mujoco
import numpy as np
from websockets.asyncio.server import serve as ws_serve
from websockets.exceptions import ConnectionClosed

sys.path.insert(0, ".")

from body.kinematics import JOINTS, POSTURES, AimSolver
from body.motion import GESTURES, MotionMixer
from common import protocol as P
from scene import build_model

URDF = "robot/dummy_lamp_5dof.urdf"
WARM = np.array([1.00, 0.82, 0.55])
COOL = np.array([0.85, 0.90, 1.00])


class Body:
    def __init__(self, view: bool = True, ctrl_hz: float = 50.0, tele_hz: float = 30.0):
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.solver = AimSolver(self.model)
        self.mixer = MotionMixer(URDF)
        self.ctrl_dt = 1.0 / ctrl_hz
        self.tele_dt = 1.0 / tele_hz
        self.clients: set = set()
        self.want_view = view
        self.viewer = None

        self.posture = POSTURES["idle"]
        self.az = 0.0
        self.el = -0.25
        self.rest_az = 0.0
        self.weight = 1.0
        self.light_set = self.posture.light
        self.warmth = 0.85
        self.pulse = 0.0
        self.last_env_t = time.perf_counter()
        self.events: list[dict] = []

        self.act = {name: self.model.actuator(name).id for name in JOINTS}
        self.qadr = {name: self.model.joint(name + "_joint").qposadr[0] for name in JOINTS}
        self.effort = {}
        for name in JOINTS:
            jid = self.model.joint(name + "_joint").id
            self.effort[name] = float(self.model.jnt_actfrcrange[jid][1])
        self.dofadr = {n: self.model.joint(n + "_joint").dofadr[0] for n in JOINTS}
        self.kp = {n: float(self.model.actuator_gainprm[self.act[n]][0]) for n in JOINTS}
        self.light_id = self.model.light("lamp_light").id
        self.bulb_id = self.model.material("bulb").id

        targets, got_az, got_el = self.solver.solve(self.az, self.el, self.posture)
        self.cmd_aim = (got_az, got_el)
        self.meas_aim = (got_az, got_el)
        self.aim_err = 0.0
        self.track_err = 0.0
        self.level = self.light_set
        for name, q in targets.items():
            self.data.qpos[self.qadr[name]] = q
            self.data.ctrl[self.act[name]] = q
        self.mixer.reference.start(targets)
        mujoco.mj_forward(self.model, self.data)

    def apply(self, msg: dict) -> None:
        cmd = msg.get("cmd")
        if cmd == P.CMD_LOOK:
            self.weight = float(msg.get("weight", 1.0))
            self.az = float(msg.get("az", self.az))
            self.el = float(msg.get("el", self.el))
        elif cmd == P.CMD_POSTURE:
            p = POSTURES.get(msg.get("name", ""))
            if p is not None:
                if p.rest_head is not None and self.posture.rest_head is None:
                    self.rest_az = self.meas_aim[0]
                self.posture = p
                self.light_set = p.light
        elif cmd == P.CMD_GESTURE:
            if not self.mixer.fire(msg.get("name", ""), float(msg.get("amp", 1.0)),
                                   float(msg.get("speed", 1.0))):
                self.events.append({"ev": "unknown_gesture", "name": msg.get("name")})
        elif cmd == P.CMD_LIGHT:
            if "intensity" in msg:
                self.light_set = max(0.0, min(1.0, float(msg["intensity"])))
            if "warm" in msg:
                self.warmth = max(0.0, min(1.0, float(msg["warm"])))
            if "pulse" in msg:
                self.pulse = max(0.0, min(1.0, float(msg["pulse"])))
        elif cmd == P.CMD_ENVELOPE:
            now = time.perf_counter()
            self.mixer.feed_envelope(max(0.0, min(1.0, float(msg.get("level", 0.0)))),
                                     max(1e-3, now - self.last_env_t))
            self.last_env_t = now

    def control(self) -> None:
        p = self.posture
        aim_az = self.az * self.weight
        aim_el = self.el * self.weight + (1.0 - self.weight) * -0.15
        if p.rest_head is not None:
            aim_az = self.rest_az
        targets, got_az, got_el = self.solver.solve(aim_az, aim_el, p)
        self.cmd_aim = (got_az, got_el)

        # breathing and gestures go on after the filter so they keep their amplitude
        off = self.mixer.advance(self.ctrl_dt, p.breath_amp, p.breath_hz, p.sway)
        ref = self.mixer.reference.step(targets, p.omega, self.ctrl_dt)

        # gravity feedforward, holds the pose without an integrator
        bias = self.data.qfrc_bias
        for name in JOINTS:
            ff = bias[self.dofadr[name]] / self.kp[name]
            self.data.ctrl[self.act[name]] = self.solver.clamp(name, ref[name] + off[name] + ff)

        mul = self.mixer.light_mul
        if self.pulse > 0:
            mul *= 1.0 + self.pulse * 0.35 * math.sin(2 * math.pi * 0.7 * self.mixer.t)
        self.level = max(0.0, min(1.6, self.light_set * mul))
        rgb = (self.warmth * WARM + (1.0 - self.warmth) * COOL) * self.level
        self.model.light_diffuse[self.light_id] = rgb
        self.model.mat_emission[self.bulb_id] = 0.15 + 0.85 * min(1.0, self.level)

        q = self.data.qpos
        off_az = off["base_yaw"] + off["neck_yaw"]
        off_el = off["shoulder_pitch"] + off["elbow_pitch"] - off["head_pitch"]
        meas_az = q[self.qadr["base_yaw"]] + q[self.qadr["neck_yaw"]] - p.neck_bias - off_az
        meas_el = (q[self.qadr["shoulder_pitch"]] + q[self.qadr["elbow_pitch"]]
                   - q[self.qadr["head_pitch"]] - off_el)
        ref_az = ref["base_yaw"] + ref["neck_yaw"] - p.neck_bias
        ref_el = ref["shoulder_pitch"] + ref["elbow_pitch"] - ref["head_pitch"]
        self.meas_aim = (meas_az, meas_el)
        self.aim_err = math.hypot(meas_az - got_az, meas_el - got_el)
        self.track_err = math.hypot(meas_az - ref_az, meas_el - ref_el)

        for name in self.mixer.done:
            self.events.append({"ev": "gesture_done", "name": name})

    def telemetry(self) -> dict:
        q = {n: float(self.data.qpos[self.qadr[n]]) for n in JOINTS}
        sat = [n for n in JOINTS
               if abs(float(self.data.actuator_force[self.act[n]])) > 0.95 * self.effort[n]]
        return {
            "type": P.MSG_STATE,
            "t": round(self.data.time, 4),
            "posture": self.posture.name,
            "q": {k: round(v, 4) for k, v in q.items()},
            "aim": {"az": round(self.meas_aim[0], 4), "el": round(self.meas_aim[1], 4)},
            "aim_cmd": {"az": round(self.cmd_aim[0], 4), "el": round(self.cmd_aim[1], 4)},
            "aim_err": round(self.aim_err, 4),
            "track_err": round(self.track_err, 4),
            "saturated": sat,
            "light": round(self.level, 3),
            "gestures": [g.name for g in self.mixer.active],
            "settled": bool(self.aim_err < 0.02),
            "warnings": [int(w.number) for w in self.data.warning if w.number],
        }

    async def serve(self, ws) -> None:
        self.clients.add(ws)
        await ws.send(json.dumps({"type": P.MSG_HELLO, "version": P.PROTOCOL_VERSION,
                                  "joints": list(JOINTS), "postures": sorted(POSTURES),
                                  "gestures": sorted(GESTURES)}))
        try:
            async for raw in ws:
                try:
                    self.apply(P.parse(raw))
                except Exception as exc:
                    await ws.send(json.dumps({"type": P.MSG_EVENT, "ev": "bad_command",
                                              "error": str(exc)}))
        except ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)

    async def broadcast(self, payload: dict) -> None:
        if not self.clients:
            return
        raw = json.dumps(payload)
        await asyncio.gather(*(c.send(raw) for c in list(self.clients)), return_exceptions=True)

    async def run(self, seconds: float | None = None) -> None:
        if self.want_view:
            from mujoco import viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.model, self.data,
                                                   show_left_ui=False, show_right_ui=False)
        t0 = time.perf_counter()
        next_ctrl = next_tele = next_view = 0.0
        while True:
            wall = time.perf_counter() - t0
            if seconds is not None and wall > seconds:
                return
            if wall >= next_ctrl:
                self.control()
                next_ctrl += self.ctrl_dt
            while self.data.time < wall:
                mujoco.mj_step(self.model, self.data)
            if wall >= next_tele:
                await self.broadcast(self.telemetry())
                for ev in self.events:
                    await self.broadcast({"type": P.MSG_EVENT, **ev})
                self.events.clear()
                next_tele += self.tele_dt
            if self.viewer is not None and wall >= next_view:
                if not self.viewer.is_running():
                    return
                self.viewer.sync()
                next_view += 1 / 60
            await asyncio.sleep(max(0.0, min(next_ctrl, next_tele) - (time.perf_counter() - t0)))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=P.DEFAULT_PORT)
    ap.add_argument("--headless", action="store_true", help="no viewer window")
    ap.add_argument("--seconds", type=float, default=None, help="exit after this long")
    args = ap.parse_args()
    body = Body(view=not args.headless)
    async with ws_serve(body.serve, "127.0.0.1", args.port):
        print(f"body listening on ws://127.0.0.1:{args.port}", flush=True)
        await body.run(args.seconds)


if __name__ == "__main__":
    asyncio.run(main())
