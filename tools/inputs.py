from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")

from brain.ears import START_RMS, STOP_RMS, Ears, Transcriber
from brain.vision import GAZE_CONE, GAZE_PITCH_CENTRE, Vision

LOG = "run/inputs_log.jsonl"
HISTORY = 12.0
FPS = 30.0

GREEN, AMBER, RED, GREY, WHITE = (90, 200, 110), (60, 170, 240), (80, 80, 220), (110, 110, 110), (235, 235, 235)


def signals(look, ears) -> dict[str, bool]:
    return {
        "face": look.face,
        "body": look.body,
        "present": look.present,
        "looking": look.looking,
        "attending": look.attending,
        "distracted": look.present and not look.attending,
        "gone": not look.present,
        "speaking": ears.speaking,
        "addressed": ears.speaking and look.attending,
        "nearby talk": ears.speaking and not look.attending,
    }


COLOURS = {"face": GREEN, "body": GREEN, "present": GREEN, "looking": GREEN, "attending": GREEN, "distracted": AMBER,
           "gone": RED, "speaking": GREEN, "addressed": GREEN, "nearby talk": AMBER}


def draw(frame, look, ears, sig, hist, heard, rows_t):
    h, w = frame.shape[:2]
    panel = np.zeros((h, 300, 3), np.uint8)
    panel[:] = (24, 22, 21)

    y = 28
    cv2.putText(panel, "inputs", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    y += 28
    for name, on in sig.items():
        col = COLOURS[name] if on else GREY
        cv2.circle(panel, (22, y - 5), 7, col, -1 if on else 1)
        cv2.putText(panel, name, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col if on else GREY, 1)
        y += 24

    y += 6
    off = ((look.yaw ** 2 + (look.pitch - GAZE_PITCH_CENTRE) ** 2) ** 0.5) if look.face else 0.0
    for label, val in (("gaze yaw/pitch", f"{look.yaw:+5.1f} / {look.pitch:+5.1f} deg"),
                       ("off cone centre", f"{off:4.1f} deg (<= {GAZE_CONE:.0f})"),
                       ("track id", f"{look.track_id}"),
                       ("aim az/el", f"{look.az:+.2f} / {look.el:+.2f} rad"),
                       ("face size", f"{look.size:.2f} of frame"),
                       ("detect", f"{look.latency_ms:.1f} ms @ {look.fps:.0f} fps")):
        cv2.putText(panel, f"{label}: {val}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)
        y += 19

    y += 10
    cv2.putText(panel, "mic level", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
    y += 8
    full = 0.06
    bx, bw = 12, 276
    cv2.rectangle(panel, (bx, y), (bx + bw, y + 14), GREY, 1)
    lvl = min(1.0, ears.level / full)
    cv2.rectangle(panel, (bx, y), (bx + int(bw * lvl), y + 14), GREEN if ears.speaking else AMBER, -1)
    for thr, col in ((START_RMS, WHITE), (STOP_RMS, GREY)):
        x = bx + int(bw * min(1.0, thr / full))
        cv2.line(panel, (x, y - 3), (x, y + 17), col, 1)
    y += 34
    cv2.putText(panel, "heard:", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
    for line in heard[-3:]:
        y += 18
        cv2.putText(panel, line[:40], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREEN, 1)

    if look.present:
        cx = int((0.5 - look.az / (2 * 0.5585)) * w)
        cv2.drawMarker(frame, (cx, h // 2), GREEN if look.attending else AMBER,
                       cv2.MARKER_CROSS, 22, 2)

    view = np.hstack([cv2.flip(frame, 1), panel])

    strip_h = 14 * len(sig) + 10
    strip = np.zeros((strip_h, view.shape[1], 3), np.uint8)
    strip[:] = (18, 17, 16)
    now = time.time()
    sw = view.shape[1] - 110
    for r, name in enumerate(sig):
        yy = 6 + r * 14
        cv2.putText(strip, name, (6, yy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, GREY, 1)
        for t, s in hist:
            if s[name]:
                x = 105 + int(sw * (1 - (now - t) / HISTORY))
                cv2.line(strip, (x, yy + 2), (x, yy + 11), COLOURS[name], 2)
    return np.vstack([view, strip])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--no-stt", action="store_true")
    args = ap.parse_args()

    os.makedirs("run", exist_ok=True)
    vision = Vision(cam_index=args.cam)
    ears = Ears()
    stt = None if args.no_stt else Transcriber()
    vision.start()
    ears.start()
    print("input monitor running, q or esc to quit, log in", LOG, flush=True)

    hist: collections.deque = collections.deque()
    heard: list[str] = []
    prev: dict[str, bool] = {}
    t0 = time.time()
    with open(LOG, "a") as log:
        log.write(json.dumps({"t": 0.0, "session": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        while True:
            tick = time.perf_counter()
            frame, _ = vision.latest_frame()
            look = vision.look
            sig = signals(look, ears)
            now = time.time()
            hist.append((now, sig))
            while hist and now - hist[0][0] > HISTORY:
                hist.popleft()

            changed = {k: v for k, v in sig.items() if prev.get(k) != v}
            if changed and prev:
                rec = {"t": round(now - t0, 2), **changed, "yaw": round(look.yaw, 1), "pitch": round(look.pitch, 1),
                       "level": round(ears.level, 4)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
            prev = sig

            utt = ears.next_utterance()
            if utt is not None:
                if stt is not None:
                    stt.submit(utt)
                else:
                    heard.append(f"[{utt.seconds:.1f}s of speech]")
            if stt is not None:
                res = stt.poll()
                if res is not None:
                    heard.append(f"{res['text']!r} ({res['stt_s']:.2f}s)")
                    log.write(json.dumps({"t": round(time.time() - t0, 2), "heard": res["text"],
                                          "stt_s": round(res["stt_s"], 2),
                                          "seconds": round(res["utterance"].seconds, 2)}) + "\n")
                    log.flush()

            if frame is not None:
                cv2.imshow("lamp inputs", draw(frame, look, ears, sig, hist, heard, None))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty("lamp inputs", cv2.WND_PROP_VISIBLE) < 1 and frame is not None:
                break
            time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - tick)))

    ears.close()
    vision.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
