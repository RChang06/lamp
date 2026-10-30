from __future__ import annotations

import asyncio
import random
import re
import time

import cv2

from brain import planner as PL
from brain.audio import SFX, AudioOut, music_bed
from brain.memory import Observation, SceneMemory
from common import actions as A

ASLEEP, IDLE, ENGAGED, LISTENING, THINKING, SPEAKING = (
    "asleep", "idle", "engaged", "listening", "thinking", "speaking")

FAREWELLS = ["See you.", "Goodnight.", "I will be here."]
IDENTITY = "I am a lamp, mostly. I watch, I listen, and I try to be useful."
NO_MEMORY = "I do not think I have seen that yet."
QUESTION = re.compile(r"^(what|where|who|why|how|did|do|does|can|could|is|are|was)|\?",
                      re.I)
LISTEN_TIMEOUT = 6.0
IDLE_DRIFT = 11.0
DOZE_AFTER = 30.0
WAKE_DWELL = 0.5
WAKE_STRETCH = 1.6

GLANCE_IGNORE = 0.5
DISTRACT_DISENGAGE = 2.0
CONVO_DISENGAGE = 4.0
CONVO_WINDOW = 8.0
REGREET_QUIET = 5.0
REGREET_FULL = 60.0
LOST_HOLD = 2.0
TALK_TO_EMPTY = 1.5
THINK_SOUND_AFTER = 0.8
THINK_TIMEOUT = 15.0
HISTORY_TURNS = 4
TURN_GESTURES = {"nod", "shake", "perk", "droop", "flash", "startle"}
LOOK_DEADBAND = 0.03
FOLLOW_GAIN = 1.6
FOLLOW_EL_CENTRE = 0.22


class Director:
    def __init__(self, link, vision, ears, stt, voice, audio: AudioOut,
                 memory: SceneMemory, plan: PL.Planner, cloud=None, quiet: bool = False):
        self.link = link
        self.vision = vision
        self.ears = ears
        self.stt = stt
        self.voice = voice
        self.audio = audio
        self.memory = memory
        self.planner = plan
        self.cloud = cloud
        self.quiet = quiet

        self.state = IDLE
        self.since = time.time()
        self.job: asyncio.Task | None = None
        self.last_drift = 0.0
        self.drift_az = 0.0
        self.music_on = False
        self.last_face = time.time()
        self.face_since: float | None = None
        self.woke_at = 0.0
        self.wake_armed = True
        self.last_looking = 0.0
        self.soft_distracted = False
        self.engaged_until = -1e9
        self.last_talk = -1e9
        self.gone_since: float | None = None
        self.sent_look: tuple[float, float, float] | None = None
        self.speech_addressed = False
        self.was_speaking = False
        self.log: list[tuple[float, str]] = []
        self.latency: list[tuple[str, float]] = []
        self.heard: list[str] = []
        self.history: list[tuple[str, str]] = []
        self.think_sound = False
        self.t0 = time.time()

    def note(self, line: str) -> None:
        self.log.append((time.time() - self.t0, line))
        print(f"[{time.time() - self.t0:6.1f}s] {line}", flush=True)

    def go(self, state: str) -> None:
        if state == self.state:
            return
        self.note(f"{self.state} -> {state}")
        if self.state == THINKING:
            self.link.light(pulse=0.0)
        self.state = state
        self.since = time.time()
        self.think_sound = False

    @property
    def busy(self) -> bool:
        return self.job is not None and not self.job.done()

    def start(self, coro) -> None:
        if self.busy:
            return
        self.job = asyncio.create_task(coro)

    def sfx(self, name: str) -> None:
        if name in SFX:
            self.audio.play(f"sfx_{name}", SFX[name](), gain=0.8)

    def music(self, on: bool) -> None:
        if on and not self.music_on and not self.quiet:
            self.audio.play("music", music_bed(), gain=0.22, loop=True)
            self.music_on = True
        elif not on and self.music_on:
            self.audio.stop("music")
            self.music_on = False

    def say(self, text: str) -> None:
        if not text:
            return
        self.note(f'say: "{text}"')
        self.voice.say(text)
        self.last_talk = time.time()
        self.go(SPEAKING)

    def in_conversation(self, now: float) -> bool:
        return now - self.last_talk < CONVO_WINDOW

    def look_at_person(self, look, weight: float) -> None:
        if self.sent_look is not None:
            az, el, w = self.sent_look
            if abs(az - look.az) < LOOK_DEADBAND and abs(el - look.el) < LOOK_DEADBAND \
                    and abs(w - weight) < 0.05:
                return
        az = max(-1.3, min(1.3, look.az * FOLLOW_GAIN))
        el = FOLLOW_EL_CENTRE + (look.el - FOLLOW_EL_CENTRE) * FOLLOW_GAIN
        self.link.look(az, el, weight)
        self.sent_look = (look.az, look.el, weight)

    def engage(self) -> None:
        away = time.time() - self.engaged_until
        self.link.posture("alert")
        self.music(True)
        self.go(ENGAGED)
        self.soft_distracted = False
        if away < REGREET_QUIET:
            self.note(f"back after {away:.1f}s, carrying on")
        elif away < REGREET_FULL:
            self.note(f"back after {away:.0f}s, a nod")
            self.sfx("blip")
            self.link.gesture("nod", amp=0.6)
        else:
            self.note("hello, without words")
            self.sfx("chirp_up")
            self.link.gesture("perk")
            self.link.light(intensity=1.0)

    def disengage(self, why: str = "looked away") -> None:
        self.note(f"disengage: {why}")
        self.sfx("chirp_down")
        self.link.gesture("droop", amp=0.8)
        self.link.posture("idle")
        self.music(False)
        self.voice.cancel()
        self.engaged_until = time.time()
        self.soft_distracted = False
        self.go(IDLE)

    def think(self) -> None:
        self.link.posture("think")
        self.link.light(pulse=0.45)
        self.go(THINKING)

    def doze(self, armed: bool = True) -> None:
        self.link.gesture("droop", amp=1.0)
        self.link.posture("sleep")
        self.music(False)
        self.voice.cancel()
        self.wake_armed = armed
        self.go(ASLEEP)

    def wake(self, why: str) -> None:
        self.note(f"waking: {why}")
        self.sfx("blip")
        self.link.posture("idle")
        self.link.gesture("perk", amp=1.2)
        self.ears.drain()
        self.woke_at = time.time()
        self.last_face = time.time()
        self.go(IDLE)

    async def tick(self) -> None:
        look = self.vision.look
        now = time.time()
        if look.present:
            self.last_face = now
            if self.face_since is None:
                self.face_since = now
        else:
            self.face_since = None

        if self.state == ASLEEP:
            if not look.present:
                self.wake_armed = True
            elif self.wake_armed and now - self.face_since >= WAKE_DWELL:
                self.wake("face")
            if self.ears.speaking and self.wake_armed:
                self.wake("voice")
            return

        looking = getattr(look, "looking", look.attending)
        if looking:
            self.last_looking = now
        away_for = now - self.last_looking
        if look.present:
            self.gone_since = None
        elif self.gone_since is None:
            self.gone_since = now
        gone_for = 0.0 if self.gone_since is None else now - self.gone_since

        if look.present:
            weight = 0.55 if self.state == IDLE else 1.0
            self.look_at_person(look, weight)
        elif gone_for > LOST_HOLD and self.state == IDLE and now - self.last_drift > IDLE_DRIFT:
            self.drift_az = random.uniform(-0.5, 0.5)
            self.link.look(self.drift_az, -0.2, 0.8)
            self.sent_look = None
            self.last_drift = now

        if self.ears.speaking and not self.was_speaking:
            # speech only counts if they were looking or already talking to it
            self.speech_addressed = looking or self.in_conversation(now)
            if not self.speech_addressed and look.present and self.state != SPEAKING:
                self.note("talk nearby, not to me, a curious glance")
                self.link.gesture("perk", amp=0.35)
        self.was_speaking = self.ears.speaking

        self.ears.boost = 3.0 if self.voice.busy() else 1.0

        result = self.stt.poll()
        if result is not None:
            text = result["text"]
            self.latency.append(("stt", result["stt_s"]))
            if text:
                self.last_talk = time.time()
                self.heard.append(text)
                self.note(f'heard: "{text}" ({result["stt_s"]:.2f}s, rtf {result["rtf"]:.2f})')
                self.start(self.respond(text))
            else:
                self.note("heard nothing usable")
                self.link.posture("alert" if look.attending else "idle")
                self.go(ENGAGED if look.attending else IDLE)

        if self.state == THINKING:
            if not self.think_sound and now - self.since > THINK_SOUND_AFTER:
                self.sfx("thinking")
                self.think_sound = True
            if now - self.since > THINK_TIMEOUT and not self.busy:
                self.note("thought too long, back to listening")
                self.link.posture("alert")
                self.go(ENGAGED)

        if self.busy:
            return

        if self.state == IDLE:
            if self.ears.next_utterance() is not None:
                self.note("speech while idle, ignored")
            if look.attending and away_for < GLANCE_IGNORE and now - self.woke_at > WAKE_STRETCH:
                self.engage()
            elif not look.present and now - self.last_face > DOZE_AFTER:
                self.note(f"nobody for {DOZE_AFTER:.0f}s, dozing off")
                self.doze()
            return

        limit = CONVO_DISENGAGE if self.in_conversation(now) else DISTRACT_DISENGAGE
        if self.state == SPEAKING and gone_for > TALK_TO_EMPTY:
            self.disengage(f"gone {gone_for:.1f}s while talking, trailing off")
            return
        if away_for > limit and self.state != SPEAKING:
            self.disengage(f"looked away {away_for:.1f}s")
            return
        if away_for > GLANCE_IGNORE and not self.soft_distracted:
            self.soft_distracted = True
            self.note("glanced away, light dips")
            self.link.light(intensity=0.55)
        elif away_for <= GLANCE_IGNORE and self.soft_distracted:
            self.soft_distracted = False
            self.note("looked back, small lift")
            self.link.light(intensity=0.85)
            self.link.gesture("perk", amp=0.4)

        if self.state == ENGAGED:
            if self.ears.speaking and self.speech_addressed:
                self.link.posture("listen")
                self.go(LISTENING)
            else:
                utt = self.ears.next_utterance()
                addressed = (self.speech_addressed or away_for < GLANCE_IGNORE
                             or self.in_conversation(now))
                if utt is not None and addressed:
                    self.stt.submit(utt)
                    self.think()
                elif utt is not None:
                    self.note(f"dropped {utt.seconds:.1f}s of talk not aimed at me")

        elif self.state == LISTENING:
            utt = self.ears.next_utterance()
            if utt is not None:
                self.note(f"utterance {utt.seconds:.1f}s rms {utt.rms:.3f}, transcribing")
                self.link.gesture("flash", amp=0.3)
                self.stt.submit(utt)
                self.think()
            elif not self.ears.speaking and now - self.since > LISTEN_TIMEOUT:
                self.link.posture("alert")
                self.go(ENGAGED)

        elif self.state == SPEAKING:
            if self.ears.speaking and self.speech_addressed and now - self.since > 0.6:
                self.note("barge in, stopping speech")
                self.voice.cancel()
                self.link.posture("listen")
                self.go(LISTENING)
            elif not self.voice.busy():
                self.link.posture("alert" if look.present else "idle")
                self.go(ENGAGED if look.present else IDLE)
            else:
                self.link.posture("speak")

    async def respond(self, text: str) -> None:
        t0 = time.perf_counter()
        intent, args = PL.route(text)
        self.note(f"intent {intent} {args if args else ''}")

        if intent != PL.Intent.BYE and self.cloud is not None and self.cloud.available:
            if await self.respond_cloud(text):
                self.latency.append(("respond", time.perf_counter() - t0))
                return

        if intent == PL.Intent.BYE:
            self.say(random.choice(FAREWELLS))
            self.music(False)
            await self.wait_for_voice()
            self.doze(armed=False)
        elif intent == PL.Intent.WHOAMI:
            self.link.gesture("nod")
            self.say(IDENTITY)
        elif intent == PL.Intent.THANKS:
            self.sfx("blip")
            self.link.gesture("nod", amp=0.7)
            self.say("Any time.")
        elif intent == PL.Intent.GREET:
            self.link.gesture("nod", amp=0.8)
            self.say("Hello again.")
        elif intent == PL.Intent.LIGHT:
            self.link.light(intensity=args["intensity"])
            self.say("How is that?")
        elif intent == PL.Intent.RECALL:
            await self.recall(text)
        elif intent == PL.Intent.LOOK_NOW:
            await self.run_plan(A.validate([
                {"action": "observe", "note": text},
                {"action": "done", "summary": "described the scene"}]))
        elif intent == PL.Intent.GOAL:
            scene = ""
            plan = self.planner.plan(text, args.get("object", ""), scene, self.frame_jpeg())
            self.note(f"plan ({self.planner.last_source}): " + " | ".join(str(a) for a in plan))
            await self.run_plan(plan)
        else:
            self.link.gesture("nod", amp=0.5)
            self.say("I am not sure what to do with that, but I am listening.")

        self.latency.append(("respond", time.perf_counter() - t0))

    async def respond_cloud(self, text: str) -> bool:
        jpeg = self.frame_jpeg()
        aim = self.link.state.get("aim", {"az": 0.0, "el": 0.0})
        history = "\n".join(f"person: {p}\nlamp: {l}" for p, l in self.history[-HISTORY_TURNS:])
        t0 = time.perf_counter()
        turn = await asyncio.get_running_loop().run_in_executor(
            None, self.cloud.turn, text, jpeg, self.memory.describe(), history,
            A.prompt_reference())
        if turn is None:
            self.note(f"cloud turn failed ({self.cloud.reason}), answering locally")
            return False
        self.latency.append(("turn", time.perf_counter() - t0))

        for obj in turn.seen[:4]:
            self.memory.note(Observation(label=obj.label, detail=obj.detail,
                                         az=aim["az"], el=aim["el"], source="vision"))
        self.note(f"turn {time.perf_counter() - t0:.2f}s: saw "
                  f"{[o.label for o in turn.seen[:4]]}, gesture {turn.gesture or '-'}, "
                  f"{len(turn.actions)} actions")
        self.history.append((text, turn.say))

        plan = None
        if turn.actions:
            try:
                plan = PL.Planner.require_observe(
                    A.validate(self.cloud.steps_to_dicts(turn.actions)))
            except A.PlanError as exc:
                self.note(f"refused the model's plan: {exc}")

        if turn.gesture in TURN_GESTURES:
            self.link.gesture(turn.gesture, amp=0.8)
        if turn.say:
            self.say(turn.say)
            if plan or turn.goodnight:
                await self.wait_for_voice()
        if plan:
            self.note("plan: " + " | ".join(str(a) for a in plan))
            await self.run_plan(plan)
        if turn.goodnight:
            self.music(False)
            self.doze(armed=False)
        return True

    async def recall(self, question: str) -> None:
        from brain.memory import tokens
        if not tokens(question) and self.memory.items:
            recent = self.memory.recent(3)
            self.link.gesture("nod", amp=0.6)
            self.say("I have seen " + ", and ".join(o.phrase() for o in recent) + ".")
            return
        obs, score = self.memory.find(question)
        self.note(f"recall score {score:.2f} -> {obs.label if obs else None}")
        if (obs is None or score < 0.25) and self.memory.items:
            recent = self.memory.recent(3)
            self.link.gesture("nod", amp=0.6)
            self.say("I have seen " + ", and ".join(o.phrase() for o in recent) + ".")
            return
        if obs is None or score < 0.25:
            self.link.gesture("shake", amp=0.7)
            self.say(NO_MEMORY)
            return
        self.link.look(obs.az, obs.el)
        self.sfx("found_it")
        ago = time.time() - obs.t
        when = "just now" if ago < 90 else f"about {int(ago / 60)} minutes ago"
        self.say(f"Yes, {obs.phrase()}. I saw it {when}.")

    def frame_jpeg(self) -> bytes | None:
        frame, _ = self.vision.latest_frame()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buf.tobytes() if ok else None

    async def run_plan(self, plan: list[A.Action]) -> None:
        for step in plan:
            self.note(f"  run {step}")
            k, a = step.kind, step.args
            if k == "say":
                self.say(a["text"])
                await self.wait_for_voice()
            elif k == "look_at":
                await self.look_at(a["target"])
            elif k == "observe":
                await self.observe(a.get("note", ""))
            elif k == "scan":
                self.link.posture("search")
                self.link.gesture("scan", amp=a.get("amp", 1.0))
                await asyncio.sleep(3.2)
            elif k == "light":
                self.link.light(intensity=a.get("intensity", 0.8), warm=a.get("warm", 0.85))
            elif k == "gesture":
                self.link.gesture(a["name"], amp=a.get("amp", 1.0))
                await asyncio.sleep(0.5)
            elif k == "sound":
                self.sfx(a["name"])
                await asyncio.sleep(0.3)
            elif k == "music":
                self.music(bool(a.get("on", True)))
            elif k == "wait":
                await asyncio.sleep(a.get("seconds", 0.5))
            elif k == "done":
                self.note(f"  done: {a.get('summary', '')}")
        self.link.posture("alert" if self.vision.look.attending else "idle")
        self.go(ENGAGED if self.vision.look.attending else IDLE)

    async def look_at(self, target: str) -> None:
        if target in ("person", "face", "you"):
            look = self.vision.look
            self.link.look(look.az, look.el)
        elif target in ("scene", "around", "table", "desk"):
            self.link.look(0.0, -0.45)
        else:
            obs, score = self.memory.find(target)
            if obs is not None and score > 0.3:
                self.link.look(obs.az, obs.el)
            else:
                self.link.look(random.uniform(-0.6, 0.6), -0.40)
        await asyncio.sleep(1.0)

    async def observe(self, note: str) -> None:
        self.link.light(intensity=1.0)
        self.sfx("thinking")
        jpeg = self.frame_jpeg()
        if jpeg is None:
            self.say("My camera is not giving me anything.")
            return
        aim = self.link.state.get("aim", {"az": 0.0, "el": 0.0})
        report = None
        if self.cloud is not None and self.cloud.available:
            t0 = time.perf_counter()
            report = await asyncio.get_running_loop().run_in_executor(
                None, self.cloud.observe, jpeg, self.memory.describe(), note)
            self.latency.append(("observe", time.perf_counter() - t0))
        if report is None and note:
            known, score = self.memory.find(note)
            if known is not None and score > 0.3:
                self.note(f"no vision backend, cannot verify {known.label} visually")
                self.say("That is where I last saw it, though I cannot check from here.")
                await self.wait_for_voice()
                return
        if report is None:
            self.note("no vision backend, asking instead")
            self.ears.drain()
            self.say("I am looking, but I cannot tell what that is. What is it?")
            await self.wait_for_voice()
            heard = (await self.wait_for_speech(6.0)).strip(" .")
            if heard and not QUESTION.search(heard) and len(heard.split()) <= 6:
                obs = self.memory.note(Observation(label=heard, az=aim["az"], el=aim["el"],
                                                   source="speech"))
                self.sfx("found_it")
                self.link.gesture("nod")
                self.say(f"{obs.label}. I will remember that.")
                await self.wait_for_voice()
            elif heard:
                self.note(f"declined to learn {heard!r}, not a label")
                self.start(self.respond(heard))
            return
        for obj in report.objects[:4]:
            self.memory.note(Observation(label=obj.label, detail=obj.detail,
                                         az=aim["az"], el=aim["el"], source="vision"))
        self.note(f"observed: {report.caption} | kept {len(report.objects[:4])} objects")
        self.sfx("found_it")
        self.say(report.answer or report.caption)
        await self.wait_for_voice()

    async def wait_for_voice(self, limit: float = 14.0) -> None:
        end = time.time() + limit
        await asyncio.sleep(0.15)
        while self.voice.busy() and time.time() < end:
            await asyncio.sleep(0.05)

    async def wait_for_speech(self, limit: float) -> str:
        self.link.posture("listen")
        end = time.time() + limit
        while time.time() < end:
            utt = self.ears.next_utterance()
            if utt is not None:
                self.stt.submit(utt)
                while time.time() < end + 4:
                    res = self.stt.poll()
                    if res is not None:
                        return res["text"]
                    await asyncio.sleep(0.05)
            await asyncio.sleep(0.05)
        return ""
