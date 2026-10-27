"""the director: four states, and the edges between them are where the character lives.

    idle      lowered, dim, breathing slowly, drifting
    engaged   up, bright, tracking the face
    listening held still, light steady, neck offered toward the speaker
    speaking  moving and glowing with its own voice

recalling is not a state. looking something up is part of composing a reply, not a way of
being, and making it a state only produces a mode the character can get stuck in.

the acknowledgement lives on the idle to engaged edge rather than in a state, so it fires
once per arrival. the thresholds are asymmetric on purpose, 0.4 s to notice and 2.0 s to
let go, because equal thresholds flicker at the boundary and flicker reads as broken.
"""
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

IDLE, ENGAGED, LISTENING, SPEAKING = "idle", "engaged", "listening", "speaking"

GREETINGS = ["Oh, hello.", "Hey there.", "There you are.", "Well, hello."]
FAREWELLS = ["See you.", "Goodnight.", "I will be here."]
IDENTITY = "I am a lamp, mostly. I watch, I listen, and I try to be useful."
NO_MEMORY = "I do not think I have seen that yet."
QUESTION = re.compile(r"^(what|where|who|why|how|did|do|does|can|could|is|are|was)|\?",
                      re.I)
LISTEN_TIMEOUT = 6.0        # give up waiting for speech and go back to just being present
IDLE_DRIFT = 11.0           # seconds between idle glances, so it is never a statue


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
        self.quiet = quiet           # suppress the music bed, for a quiet room or a test

        self.state = IDLE
        self.since = time.time()
        self.job: asyncio.Task | None = None
        self.last_drift = 0.0
        self.drift_az = 0.0
        self.music_on = False
        self.log: list[tuple[float, str]] = []
        self.latency: list[tuple[str, float]] = []
        self.heard: list[str] = []
        self.t0 = time.time()

    # ---- bookkeeping -----------------------------------------------------------------
    def note(self, line: str) -> None:
        self.log.append((time.time() - self.t0, line))
        print(f"[{time.time() - self.t0:6.1f}s] {line}", flush=True)

    def go(self, state: str) -> None:
        if state == self.state:
            return
        self.note(f"{self.state} -> {state}")
        self.state = state
        self.since = time.time()

    @property
    def busy(self) -> bool:
        return self.job is not None and not self.job.done()

    def start(self, coro) -> None:
        if self.busy:
            return
        self.job = asyncio.create_task(coro)

    # ---- sound -----------------------------------------------------------------------
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
        self.go(SPEAKING)

    # ---- the edges -------------------------------------------------------------------
    def engage(self) -> None:
        self.sfx("chirp_up")
        self.link.posture("alert")
        self.link.gesture("perk")
        self.music(True)
        self.go(ENGAGED)
        self.say(random.choice(GREETINGS))

    def disengage(self) -> None:
        self.sfx("chirp_down")
        self.link.gesture("droop", amp=0.8)
        self.link.posture("idle")
        self.music(False)
        self.voice.cancel()
        self.go(IDLE)

    # ---- the loop --------------------------------------------------------------------
    async def tick(self) -> None:
        look = self.vision.look
        now = time.time()

        # gaze: follow the face whenever there is one, whatever state we are in
        if look.present:
            weight = 1.0 if look.attending else 0.55
            self.link.look(look.az, look.el, weight)
        elif self.state == IDLE and now - self.last_drift > IDLE_DRIFT:
            # a slow glance somewhere else, the cheapest possible sign of being alive
            self.drift_az = random.uniform(-0.5, 0.5)
            self.link.look(self.drift_az, -0.2, 0.8)
            self.last_drift = now

        # the microphone stays open while the lamp talks, so it can be interrupted, but
        # the onset threshold goes up so it does not trigger on its own voice coming back
        # through the speaker. proper echo cancellation is the real fix, see the write up.
        self.ears.boost = 3.0 if self.voice.busy() else 1.0

        # transcripts arriving from the worker
        result = self.stt.poll()
        if result is not None:
            text = result["text"]
            self.latency.append(("stt", result["stt_s"]))
            if text:
                self.heard.append(text)
                self.note(f'heard: "{text}" ({result["stt_s"]:.2f}s, rtf {result["rtf"]:.2f})')
                self.start(self.respond(text))
            else:
                self.go(ENGAGED if look.attending else IDLE)

        if self.busy:
            return                        # a plan is running, it owns the character

        if self.state == IDLE:
            if look.attending:
                self.engage()

        elif self.state == ENGAGED:
            if not look.attending and now - self.since > 0.3:
                self.disengage()
            elif self.ears.speaking:
                self.link.posture("listen")
                self.go(LISTENING)
            else:
                utt = self.ears.next_utterance()
                if utt is not None:
                    self.stt.submit(utt)
                    self.go(LISTENING)

        elif self.state == LISTENING:
            utt = self.ears.next_utterance()
            if utt is not None:
                self.note(f"utterance {utt.seconds:.1f}s rms {utt.rms:.3f}, transcribing")
                self.link.gesture("flash", amp=0.3)
                self.stt.submit(utt)
            elif not self.ears.speaking and now - self.since > LISTEN_TIMEOUT:
                self.link.posture("alert")
                self.go(ENGAGED)

        elif self.state == SPEAKING:
            if self.ears.speaking and now - self.since > 0.6:
                # barge in: it stops talking the moment the person starts
                self.note("barge in, stopping speech")
                self.voice.cancel()
                self.link.posture("listen")
                self.go(LISTENING)
            elif not self.voice.busy():
                self.link.posture("alert" if look.attending else "idle")
                self.go(ENGAGED if look.attending else IDLE)
            else:
                self.link.posture("speak")

    # ---- responding ------------------------------------------------------------------
    async def respond(self, text: str) -> None:
        t0 = time.perf_counter()
        intent, args = PL.route(text)
        self.note(f"intent {intent} {args if args else ''}")

        if intent == PL.Intent.BYE:
            self.say(random.choice(FAREWELLS))
            self.music(False)
            await self.wait_for_voice()
            self.link.posture("sleep")
            self.go(IDLE)
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
            reply = ""
            if self.cloud is not None and self.cloud.available:
                reply = self.cloud.reply(text, self.memory.describe())
            self.link.gesture("nod", amp=0.5)
            self.say(reply or "I am not sure what to do with that, but I am listening.")

        self.latency.append(("respond", time.perf_counter() - t0))

    async def recall(self, question: str) -> None:
        from brain.memory import tokens
        if not tokens(question) and self.memory.items:
            # nothing specific was asked, so say what it has
            recent = self.memory.recent(3)
            self.link.gesture("nod", amp=0.6)
            self.say("I have seen " + ", and ".join(o.phrase() for o in recent) + ".")
            return
        obs, score = self.memory.find(question)
        self.note(f"recall score {score:.2f} -> {obs.label if obs else None}")
        if (obs is None or score < 0.25) and self.memory.items:
            # a weak match is not the same as an empty memory. offering what it has is
            # both more useful and more in character than a flat denial.
            recent = self.memory.recent(3)
            self.link.gesture("nod", amp=0.6)
            self.say("I have seen " + ", and ".join(o.phrase() for o in recent) + ".")
            return
        if obs is None or score < 0.25:
            self.link.gesture("shake", amp=0.7)
            self.say(NO_MEMORY)
            return
        # look back where it was when it saw the thing, then answer
        self.link.look(obs.az, obs.el)
        self.sfx("found_it")
        ago = time.time() - obs.t
        when = "just now" if ago < 90 else f"about {int(ago / 60)} minutes ago"
        self.say(f"Yes, {obs.phrase()}. I saw it {when}.")

    # ---- running a plan --------------------------------------------------------------
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
        """look at the frame and remember what is in it, or admit it cannot tell."""
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
            # the no model path: ask, and learn from the answer
            self.note("no vision backend, asking instead")
            self.ears.drain()      # before asking: an answer given over the question counts
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
                # it heard something, but a question is not the name of a thing
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

    # ---- small waits -----------------------------------------------------------------
    async def wait_for_voice(self, limit: float = 14.0) -> None:
        end = time.time() + limit
        await asyncio.sleep(0.15)
        while self.voice.busy() and time.time() < end:
            await asyncio.sleep(0.05)

    async def wait_for_speech(self, limit: float) -> str:
        """used when the character has asked a direct question and wants the answer."""
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
