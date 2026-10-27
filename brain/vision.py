"""the camera: one owner, two consumers.

the attention tracker runs the face landmarker at 15 hz, which is plenty for gaze and
half the cpu of running it at camera rate, and it is the only thing that decides whether
someone is attending. scene memory borrows the same latest frame when it wants to look at
an object, so the camera is opened exactly once.

engagement is deliberately asymmetric: fast to notice, slow to give up. symmetric
thresholds flicker on the boundary and flicker reads as broken rather than alive.
"""
from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL = "models/face_landmarker.task"

# a typical laptop webcam is around 65 degrees horizontal
HALF_FOV_H = math.radians(32.0)
HALF_FOV_V = math.radians(32.0) * 0.5625      # 16:9 sensor

ENGAGE_DWELL = 0.40      # seconds of a frontal face before we call it attention
DISENGAGE_DWELL = 2.00   # seconds without one before we let go
FRONTAL_YAW = 28.0       # degrees of head yaw still counted as looking at us
EL_BIAS = 0.22           # the lamp sits below a seated face, so look up a little


@dataclass
class Look:
    """what the camera currently believes about the person in front of it."""
    present: bool = False
    attending: bool = False
    az: float = 0.0
    el: float = EL_BIAS
    yaw: float = 0.0
    pitch: float = 0.0
    size: float = 0.0        # face width as a fraction of the frame, a distance proxy
    fps: float = 0.0
    latency_ms: float = 0.0
    t: float = field(default_factory=time.time)


class Vision:
    def __init__(self, cam_index: int = 0, rate: float = 15.0, mirror: bool = False,
                 width: int = 640, height: int = 480):
        self.cam_index = cam_index
        self.period = 1.0 / rate
        self.mirror = mirror
        self.size = (width, height)
        self.look = Look()
        self.frame: np.ndarray | None = None
        self.frame_t = 0.0
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.thread: threading.Thread | None = None
        self.frames = 0
        self.detect_ms: list[float] = []

    def start(self) -> None:
        self.thread = threading.Thread(target=self._loop, name="vision", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stop_flag.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def latest_frame(self) -> tuple[np.ndarray | None, float]:
        """a copy of the most recent frame, for scene memory to look at."""
        with self.lock:
            if self.frame is None:
                return None, 0.0
            return self.frame.copy(), self.frame_t

    def _loop(self) -> None:
        # the constant exists on every platform, so test the platform, not the constant
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
        cap = cv2.VideoCapture(self.cam_index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL),
            running_mode=vision.RunningMode.VIDEO,
            output_facial_transformation_matrixes=True,
            num_faces=1)
        landmarker = vision.FaceLandmarker.create_from_options(opts)

        t_start = time.time()
        last_frontal = 0.0        # last time a frontal face was seen
        frontal_since = None      # when the current run of frontal frames began
        attending = False
        fps_mark, fps_count = time.perf_counter(), 0

        while not self.stop_flag.is_set():
            tick = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self.lock:
                self.frame = frame
                self.frame_t = time.time()
            self.frames += 1
            fps_count += 1

            t0 = time.perf_counter()
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = landmarker.detect_for_video(img, int((time.time() - t_start) * 1000))
            self.detect_ms.append(1000 * (time.perf_counter() - t0))
            del self.detect_ms[:-200]

            now = time.time()
            present = bool(res.face_landmarks)
            az, el, yaw, pitch, face_w = 0.0, EL_BIAS, 0.0, 0.0, 0.0
            if present:
                pts = res.face_landmarks[0]
                xs = [p.x for p in pts]
                ys = [p.y for p in pts]
                cx, cy = float(np.mean(xs)), float(np.mean(ys))
                face_w = float(max(xs) - min(xs))
                # the raw frame is not mirrored, so a person moving to their own right
                # moves toward image left, which is the lamp's own left, hence +az there
                az = (0.5 - cx) * 2 * HALF_FOV_H
                el = (0.5 - cy) * 2 * HALF_FOV_V + EL_BIAS
                if res.facial_transformation_matrixes:
                    R = np.array(res.facial_transformation_matrixes[0])[:3, :3]
                    yaw = math.degrees(math.atan2(R[0, 2], R[2, 2]))
                    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -R[1, 2]))))
            frontal = present and abs(yaw) <= FRONTAL_YAW
            if frontal:
                if frontal_since is None:
                    frontal_since = now
                last_frontal = now
            elif now - last_frontal > 0.25:
                # a quarter second of tolerance: a blink or a dropped detection is
                # not the person looking away
                frontal_since = None

            # asymmetric engagement, see the module docstring
            if not attending and frontal_since is not None \
                    and now - frontal_since >= ENGAGE_DWELL:
                attending = True
            elif attending and now - last_frontal > DISENGAGE_DWELL:
                attending = False

            if time.perf_counter() - fps_mark >= 1.0:
                fps = fps_count / (time.perf_counter() - fps_mark)
                fps_mark, fps_count = time.perf_counter(), 0
            else:
                fps = self.look.fps

            self.look = Look(present=present, attending=attending, az=az, el=el, yaw=yaw,
                             pitch=pitch, size=face_w, fps=fps,
                             latency_ms=float(np.mean(self.detect_ms[-30:])), t=now)

            sleep = self.period - (time.perf_counter() - tick)
            if sleep > 0:
                time.sleep(sleep)

        cap.release()
        landmarker.close()
