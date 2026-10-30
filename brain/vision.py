from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from mediapipe.tasks.python import BaseOptions, vision
import uniface
from uniface import BYTETracker, MobileGaze, RetinaFace
from uniface.constants import GazeWeights, RetinaFaceWeights

POSE_MODEL = "models/pose_landmarker_lite.task"
UNIFACE_DIR = "models/uniface"
DETECTOR = RetinaFaceWeights.MNET_V2
GAZE = GazeWeights.MOBILEONE_S0

HALF_FOV_H = math.radians(32.0)
HALF_FOV_V = math.radians(32.0) * 0.5625

# quick to notice, slow to let go, so it never flickers
ENGAGE_DWELL = 0.40
DISENGAGE_DWELL = 2.00
GONE_GRACE = 0.80
POSE_EVERY = 3
GAZE_EVERY = 2

GAZE_CONE = 22.0
GAZE_PITCH_CENTRE = -6.0
GAZE_SMOOTH = 0.45
FACE_CONF = 0.55
BODY_VIS = 0.5
FRONTAL_YAW = GAZE_CONE

EL_BIAS = 0.22
ORT_THREADS = 2


# onnxruntime defaults to one busy waiting thread per core
def lean(model) -> None:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = ORT_THREADS
    opts.inter_op_num_threads = 1
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    model.session = ort.InferenceSession(model.session._model_path, sess_options=opts,
                                         providers=["CPUExecutionProvider"])


@dataclass
class Look:
    present: bool = False
    attending: bool = False
    looking: bool = False
    face: bool = False
    body: bool = False
    az: float = 0.0
    el: float = EL_BIAS
    yaw: float = 0.0
    pitch: float = 0.0
    size: float = 0.0
    track_id: int = -1
    bbox: tuple[float, float, float, float] | None = None
    body_xy: tuple[float, float] | None = None
    fps: float = 0.0
    latency_ms: float = 0.0
    t: float = field(default_factory=time.time)


class Vision:
    def __init__(self, cam_index: int = 0, rate: float = 10.0, mirror: bool = False,
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
        with self.lock:
            if self.frame is None:
                return None, 0.0
            return self.frame.copy(), self.frame_t

    def _body(self, pose, frame, ts_ms: int) -> tuple[bool, float, float]:
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = pose.detect_for_video(img, ts_ms)
        if not res.pose_landmarks:
            return False, 0.5, 0.5
        lm = res.pose_landmarks[0]
        seen = [p for p in (lm[0], lm[11], lm[12]) if (p.visibility or 0) > BODY_VIS]
        if not seen:
            return False, 0.5, 0.5
        head = lm[0] if (lm[0].visibility or 0) > BODY_VIS else seen[0]
        return True, float(head.x), float(head.y)

    def _loop(self) -> None:
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
        cap = cv2.VideoCapture(self.cam_index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])

        uniface.set_cache_dir(UNIFACE_DIR)
        detector = RetinaFace(model_name=DETECTOR, input_size=(320, 320),
                              confidence_threshold=FACE_CONF)
        tracker = BYTETracker(track_thresh=FACE_CONF, track_buffer=10)
        gaze_model = MobileGaze(model_name=GAZE)
        lean(detector)
        lean(gaze_model)
        pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=vision.RunningMode.VIDEO, num_poses=1))

        t_start = time.time()
        last_seen = 0.0
        last_looking = 0.0
        looking_since = None
        attending = False
        body, body_xy = False, (0.5, 0.5)
        g_yaw = g_pitch = None
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
            h, w = frame.shape[:2]
            now = time.time()

            t0 = time.perf_counter()
            faces = detector.detect(frame, max_num=3)
            dets = np.array([[*f.bbox[:4], f.confidence] for f in faces], np.float32) \
                if faces else np.empty((0, 5), np.float32)
            tracks = tracker.update(dets)

            face, track_id, size, bbox = False, -1, 0.0, None
            cx, cy = 0.5, 0.5
            looking = False
            if len(tracks):
                x1, y1, x2, y2, tid = max(tracks, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                if x2 - x1 > 8 and y2 - y1 > 8:
                    face, track_id = True, int(tid)
                    size = (x2 - x1) / w
                    bbox = (x1 / w, y1 / h, x2 / w, y2 / h)
                    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                    if g_yaw is None or self.frames % GAZE_EVERY == 0:
                        g = gaze_model.estimate(frame[y1:y2, x1:x2])
                        yaw_d, pitch_d = math.degrees(g.yaw), math.degrees(g.pitch)
                        if g_yaw is None:
                            g_yaw, g_pitch = yaw_d, pitch_d
                        else:
                            g_yaw += GAZE_SMOOTH * (yaw_d - g_yaw)
                            g_pitch += GAZE_SMOOTH * (pitch_d - g_pitch)
                    looking = math.hypot(g_yaw, g_pitch - GAZE_PITCH_CENTRE) <= GAZE_CONE
            if not face:
                g_yaw = g_pitch = None

            if self.frames % POSE_EVERY == 0 or not face:
                body, bx, by = self._body(pose, frame, int((now - t_start) * 1000))
                body_xy = (bx, by)
            self.detect_ms.append(1000 * (time.perf_counter() - t0))
            del self.detect_ms[:-200]

            if face or body:
                last_seen = now
            present = now - last_seen <= GONE_GRACE
            if not face and body:
                cx, cy = body_xy

            if looking:
                if looking_since is None:
                    looking_since = now
                last_looking = now
            elif now - last_looking > 0.25:
                looking_since = None
            if not attending and looking_since is not None \
                    and now - looking_since >= ENGAGE_DWELL:
                attending = True
            elif attending and now - last_looking > DISENGAGE_DWELL:
                attending = False

            az = (0.5 - cx) * 2 * HALF_FOV_H if present else 0.0
            el = (0.5 - cy) * 2 * HALF_FOV_V + EL_BIAS if present else EL_BIAS

            if time.perf_counter() - fps_mark >= 1.0:
                fps = fps_count / (time.perf_counter() - fps_mark)
                fps_mark, fps_count = time.perf_counter(), 0
            else:
                fps = self.look.fps

            self.look = Look(present=present, attending=attending, looking=looking,
                             face=face, body=body, az=az, el=el,
                             yaw=g_yaw or 0.0, pitch=g_pitch or 0.0, size=size,
                             track_id=track_id, bbox=bbox,
                             body_xy=body_xy if body else None, fps=fps,
                             latency_ms=float(np.mean(self.detect_ms[-30:])), t=now)

            sleep = self.period - (time.perf_counter() - tick)
            if sleep > 0:
                time.sleep(sleep)
        cap.release()
