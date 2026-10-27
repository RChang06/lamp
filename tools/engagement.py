"""engagement reliability, measured with a person in front of the camera.

the one number in the submission that cannot be synthesized. it runs trials, tells you
when to look and when to look away, and records whether the character noticed, how long
it took, and whether it ever engaged while nobody was attending.

  uv run python tools/engagement.py --trials 10

report the false engagement count honestly: it is the number that decides whether the
thing is pleasant to sit next to.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from brain.vision import DISENGAGE_DWELL, ENGAGE_DWELL, Vision


def wait_for(v: Vision, want: bool, limit: float) -> float | None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < limit:
        if v.look.attending == want:
            return time.perf_counter() - t0
        time.sleep(0.01)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--mirror", action="store_true")
    a = ap.parse_args()

    v = Vision(cam_index=a.cam, mirror=a.mirror)
    v.start()
    time.sleep(2.0)
    print(f"camera up: {v.look.fps:.0f} fps, landmarker {v.look.latency_ms:.1f} ms/frame")
    print(f"thresholds: engage after {ENGAGE_DWELL:.2f}s frontal, "
          f"disengage after {DISENGAGE_DWELL:.2f}s away\n")

    on, off = [], []
    missed_on = missed_off = false_on = 0

    for i in range(1, a.trials + 1):
        print(f"trial {i}/{a.trials}")
        print("  look AWAY from the lamp", flush=True)
        time.sleep(2.5)
        if v.look.attending:
            wait_for(v, False, 3.0)
        # watch for a spurious engagement while nobody is attending
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 2.0:
            if v.look.attending:
                false_on += 1
                print("  ! engaged while you were away")
                break
            time.sleep(0.02)

        print("  now LOOK AT it", flush=True)
        dt = wait_for(v, True, 5.0)
        if dt is None:
            missed_on += 1
            print("  ! never engaged")
        else:
            on.append(dt)
            print(f"  engaged in {dt:.2f}s")

        time.sleep(1.0)
        print("  look AWAY again", flush=True)
        dt = wait_for(v, False, 6.0)
        if dt is None:
            missed_off += 1
            print("  ! never let go")
        else:
            off.append(dt)
            print(f"  released in {dt:.2f}s")
        print()

    v.close()

    print("---- engagement reliability ----")
    n = a.trials
    print(f"trials                     {n}")
    print(f"engaged when looked at     {len(on)}/{n}  ({100 * len(on) / n:.0f}%)")
    print(f"released when looked away  {len(off)}/{n}  ({100 * len(off) / n:.0f}%)")
    print(f"false engagements          {false_on} over {2.0 * n:.0f}s of looking away")
    for name, vals in (("engage latency", on), ("release latency", off)):
        if vals:
            arr = np.array(vals)
            print(f"{name:<26} mean {arr.mean():.2f}s  p50 {np.median(arr):.2f}s  "
                  f"max {arr.max():.2f}s")
    print("\nthe engage latency includes the dwell by design, so it cannot go below "
          f"{ENGAGE_DWELL:.2f}s")
    print("the release latency likewise cannot go below "
          f"{DISENGAGE_DWELL:.2f}s, which is the point: slow to let go")


if __name__ == "__main__":
    main()
