from __future__ import annotations

import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")

from brain.cloud import Cloud
from brain.memory import SceneMemory
from brain.planner import Planner
from common import actions as A


def synthetic_desk() -> bytes:
    img = np.full((480, 640, 3), 210, dtype=np.uint8)
    cv2.rectangle(img, (0, 330), (640, 480), (150, 170, 190), -1)
    cv2.rectangle(img, (120, 250), (210, 350), (190, 110, 40), -1)
    cv2.ellipse(img, (165, 250), (45, 16), 0, 0, 360, (220, 150, 80), -1)
    cv2.ellipse(img, (222, 295), (26, 30), 0, 250, 440, (190, 110, 40), 10)
    cv2.rectangle(img, (300, 300), (470, 350), (60, 200, 230), -1)
    cv2.rectangle(img, (300, 300), (470, 350), (40, 150, 180), 3)
    cv2.rectangle(img, (520, 270), (580, 350), (120, 120, 130), -1)
    for dx, dy in ((0, -40), (-25, -25), (25, -25), (-12, -55), (12, -55)):
        cv2.ellipse(img, (550 + dx, 270 + dy), (22, 11), 0, 0, 360, (60, 140, 60), -1)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    cv2.imwrite("scratch/synthetic_desk.jpg", img)
    return buf.tobytes()


def main() -> None:
    cloud = Cloud()
    print(f"model {cloud.model}  available {cloud.available}")
    if not cloud.available:
        print(f"no credentials: {cloud.reason}")
        print("this is a supported mode, the character asks instead. nothing to check.")
        return

    jpeg = synthetic_desk()
    print(f"synthetic scene: {len(jpeg) / 1000:.0f} kB jpeg, saved to "
          "scratch/synthetic_desk.jpg\n")

    t0 = time.perf_counter()
    report = cloud.observe(jpeg, known="nothing yet")
    dt = time.perf_counter() - t0
    if report is None:
        print(f"observe FAILED after {dt:.2f}s: {cloud.reason}")
        return
    print(f"observe: {dt:.2f}s")
    print(f"  caption: {report.caption}")
    for o in report.objects:
        print(f"  object : {o.label}" + (f" ({o.detail})" if o.detail else ""))

    mem = SceneMemory("run/cloud_check_memory.jsonl")
    from brain.memory import Observation
    for o in report.objects[:4]:
        mem.note(Observation(label=o.label, detail=o.detail, az=0.1, el=-0.3))

    goal = "can you shine your light on the mug and tell me what is next to it"
    t0 = time.perf_counter()
    plan = Planner(mem, cloud).plan(goal, "mug", report.caption, jpeg)
    dt = time.perf_counter() - t0
    print(f"\nplan: {dt:.2f}s, source {Planner(mem, cloud).last_source or 'cloud'}")
    for step in plan:
        print(f"  {step}")

    kinds = [s.kind for s in plan]
    print(f"\nvalidated {len(plan)} steps, vocabulary respected: "
          f"{all(k in A.SCHEMA for k in kinds)}")
    print(f"observes before done: {'observe' in kinds and 'done' in kinds}")
    print("\nnote: a plan is refused whole if any step is invalid, and an observe is "
          "inserted if a plan would finish without looking again")


if __name__ == "__main__":
    main()
