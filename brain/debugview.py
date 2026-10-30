from __future__ import annotations

import time

import cv2
import numpy as np

from brain import director as D

GREEN, AMBER, RED, GREY, WHITE = (90, 200, 110), (60, 170, 240), (80, 80, 220), (110, 110, 110), (235, 235, 235)
STATE_COLOUR = {D.ASLEEP: (160, 120, 90), D.IDLE: GREY, D.ENGAGED: GREEN,
                D.LISTENING: (230, 190, 80), D.THINKING: (200, 140, 230),
                D.SPEAKING: (80, 200, 230)}
WINDOW = "lamp brain"


class DebugView:
    def __init__(self, director, vision, ears, link, every: float = 1 / 12):
        self.d, self.vision, self.ears, self.link = director, vision, ears, link
        self.every = every
        self.last = 0.0
        self.closed = False

    def draw(self) -> None:
        now = time.time()
        if self.closed or now - self.last < self.every:
            return
        self.last = now
        frame, _ = self.vision.latest_frame()
        if frame is None:
            frame = np.zeros((480, 640, 3), np.uint8)
        d, look, ears = self.d, self.vision.look, self.ears
        h, w = frame.shape[:2]
        view = cv2.flip(frame, 1)

        box = getattr(look, "bbox", None)
        if box is not None:
            x1, y1, x2, y2 = box
            p1 = (int((1 - x2) * w), int(y1 * h))
            p2 = (int((1 - x1) * w), int(y2 * h))
            looking = getattr(look, "looking", look.attending)
            bc = GREEN if looking else AMBER
            cv2.rectangle(view, p1, p2, bc, 2)
            cv2.putText(view, f"id {look.track_id}  {'looking' if looking else 'away'}",
                        (p1[0], max(58, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bc, 2)
            cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            length = (p2[0] - p1[0]) * 1.2
            gx = int(cx + length * np.sin(np.radians(look.yaw)))
            gy = int(cy - length * np.sin(np.radians(look.pitch)))
            cv2.arrowedLine(view, (cx, cy), (gx, gy), (255, 200, 80), 2, tipLength=0.25)
        body_xy = getattr(look, "body_xy", None)
        if body_xy is not None:
            bx, by = int((1 - body_xy[0]) * w), int(body_xy[1] * h)
            cv2.circle(view, (bx, by), 9, (200, 120, 255), 2)
            cv2.putText(view, "body", (bx + 12, by + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 120, 255), 1)

        col = STATE_COLOUR.get(d.state, WHITE)
        cv2.rectangle(view, (0, 0), (w, 44), (20, 20, 20), -1)
        cv2.putText(view, d.state.upper(), (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
        cv2.putText(view, f"{now - d.since:5.1f}s", (w - 110, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, WHITE, 1)

        panel = np.zeros((h, 330, 3), np.uint8)
        panel[:] = (24, 22, 21)
        y = 26

        def row(label, on, colour=GREEN, extra=""):
            nonlocal y
            c = colour if on else GREY
            cv2.circle(panel, (20, y - 5), 7, c, -1 if on else 1)
            cv2.putText(panel, label + extra, (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
            y += 22

        def text(line, colour=WHITE, scale=0.45):
            nonlocal y
            cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1)
            y += 19

        text("what it sees", GREY)
        row("face", getattr(look, "face", look.present))
        row("body", getattr(look, "body", look.present))
        row("present", look.present)
        row("looking", getattr(look, "looking", look.attending))
        row("attending", look.attending)
        row("distracted", look.present and not getattr(look, "looking", look.attending), AMBER)
        row("speaking", ears.speaking)
        row("addressed", ears.speaking and d.speech_addressed)

        y += 6
        text("timers", GREY)
        away = now - d.last_looking
        gone = 0.0 if d.gone_since is None else now - d.gone_since
        convo = d.in_conversation(now)
        limit = D.CONVO_DISENGAGE if convo else D.DISTRACT_DISENGAGE
        text(f"away for  {min(away, 99):5.1f}s  (ignore <{D.GLANCE_IGNORE}, give up >{limit})",
             AMBER if away > D.GLANCE_IGNORE else WHITE)
        text(f"gone for  {gone:5.1f}s  (sleep after {D.DOZE_AFTER:.0f})", RED if gone else WHITE)
        text(f"conversation {'yes' if convo else 'no'}, light dipped {'yes' if d.soft_distracted else 'no'}")
        text(f"gaze {look.yaw:+.0f}/{look.pitch:+.0f} deg, {look.fps:.0f} fps, {look.latency_ms:.0f} ms")
        body = self.link.state.get("posture", "?") if self.link.connected else "NOT CONNECTED"
        text(f"body posture: {body}", GREEN if self.link.connected else RED)

        y += 4
        bw = 300
        cv2.rectangle(panel, (12, y), (12 + bw, y + 10), GREY, 1)
        cv2.rectangle(panel, (12, y), (12 + int(bw * min(1.0, ears.level / 0.06)), y + 10),
                      GREEN if ears.speaking else AMBER, -1)
        y += 26
        text(f"speech prob {getattr(ears, 'prob', 0.0):.2f} (starts at 0.5)",
             GREEN if getattr(ears, 'prob', 0.0) >= 0.5 else WHITE)

        text("last notes", GREY)
        for t, line in d.log[-7:]:
            text(f"{t:6.1f} {line}"[:46], WHITE, 0.4)

        out = np.hstack([view, panel])
        cv2.imshow(WINDOW, out)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            self.close()
        elif cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            self.close()

    def close(self) -> None:
        self.closed = True
        cv2.destroyWindow(WINDOW)
