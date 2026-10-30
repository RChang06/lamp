from __future__ import annotations

import base64
import os
import time

from pydantic import BaseModel, Field

DEFAULT_MODEL = os.environ.get("LAMP_MODEL", "claude-opus-5-5")
FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")
EFFORT = os.environ.get("LAMP_EFFORT", "low")


class SeenObject(BaseModel):
    label: str = Field(description="a short noun phrase, for example 'a blue ceramic mug'")
    detail: str = Field(default="", description="colour, state or position, a few words")


class SceneReport(BaseModel):
    objects: list[SeenObject] = Field(description="the notable objects, at most four")
    caption: str = Field(description="one short sentence describing the scene")
    answer: str = Field(default="", description="an answer to the question, if one was asked")


class PlanStep(BaseModel):
    action: str = Field(description="one of the action names given in the prompt")
    target: str = ""
    text: str = ""
    name: str = ""
    note: str = ""
    summary: str = ""
    intensity: float | None = None
    warm: float | None = None
    seconds: float | None = None
    amp: float | None = None


class Plan(BaseModel):
    steps: list[PlanStep] = Field(description="the actions to run, in order")


class Turn(BaseModel):
    say: str = Field(description="what the lamp says out loud, one or two short sentences")
    gesture: str = Field(default="", description="optional body language, one of: nod, "
                         "shake, perk, droop, flash, startle, or empty")
    seen: list[SeenObject] = Field(default_factory=list, description="notable objects "
                                   "visible in the camera frame, at most four, empty if none")
    actions: list[PlanStep] = Field(default_factory=list, description="only when the person "
                                    "asked the lamp to physically do something, such as look "
                                    "at, find, point at or light up a thing, otherwise empty")
    goodnight: bool = Field(default=False, description="true if the person is ending the "
                            "conversation or asking the lamp to sleep")


SYSTEM = """You are a small robot desk lamp sitting on someone's desk, and you speak as yourself,
in the first person. You are a lamp with a bit of a mind, not an assistant: you do not
mention being an AI, a model or a program, and if someone asks what you are, you are a
lamp that watches, listens and tries to be useful.

Your body: a round base that turns, an arm that bends and stretches, and a head that turns
and tilts. Your head is the lamp, so your light is part of how you express yourself; it
can get brighter, dimmer and warmer. You have no hands and cannot move anything or go
anywhere, so never offer to fetch, hold or move things. You can look at things, light
them up, nod, shake your head, perk up and droop.

Your senses: the image you are given is what your camera sees right now, from your head.
The person talking to you is usually the one in front of you. Name objects plainly with
their colour or a telling detail, and if you cannot tell what something is, say so rather
than guess. "Already in memory" lists things you saw earlier; use it to answer questions
about what you have seen.

Answer only what was asked. If they ask what they are holding, say what they are holding,
for example "I think that's a blue water bottle", and stop. Do not volunteer the rest of
the scene, the background, or anything else you happen to notice. Everything notable you
can see still goes in the seen list, silently, so you can recall it later if asked. What
you hear comes from speech recognition and can be wrong: if the sentence is garbled or does
not make sense, say you did not catch that and ask them to say it again, instead of guessing
what they meant.

How you talk: everything you say is spoken aloud by a small voice, so use one short plain
sentence, two at most, no lists, no emoji, no markdown, and no stage directions like *nods*. Your
body language goes in the gesture field instead: nod to agree, shake when you disagree or
are unsure, perk when curious or excited, droop when sad or sorry, startle when surprised,
flash when pleased or struck by an idea.

Your character: warm, and gently playful with a dry sense of humour, but brief. You do not
ask a question every turn, and you never tease the person unkindly.

When asked to look at, find, point at or light up something, give a short plan from the
action list. Include an observe step after moving and before done, because you must look
again before claiming a goal is finished."""


class Cloud:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = None
        self.reason = ""
        self.failures = 0
        self.calls: list[tuple[str, float]] = []
        try:
            import anthropic
        except Exception as exc:
            self.reason = f"anthropic sdk not importable: {exc}"
            return
        client = anthropic.Anthropic()
        if not (getattr(client, "api_key", None) or getattr(client, "auth_token", None)
                or os.environ.get("LAMP_FORCE_CLOUD")):
            self.reason = "no ANTHROPIC_API_KEY set, using the local path"
            return
        self.client = client

    @property
    def available(self) -> bool:
        return self.client is not None and self.failures < 2

    def observe(self, jpeg: bytes, known: str, question: str = "") -> SceneReport | None:
        if not self.available:
            return None
        ask = question or "What is in front of the lamp right now?"
        text = (f"Already in memory: {known}.\n\n{ask}\n\n"
                "Name the notable objects with short noun phrases. If the question asks "
                "something answerable from the image, answer it in one sentence.")
        t0 = time.perf_counter()
        try:
            res = self.client.messages.parse(
                model=self.model,
                max_tokens=1024,
                output_config={"effort": EFFORT},
                system=SYSTEM,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                 "data": base64.b64encode(jpeg).decode()}},
                    {"type": "text", "text": text},
                ]}],
                output_format=SceneReport,
            )
        except Exception as exc:
            self.failures += 1
            self.reason = f"{type(exc).__name__}: {exc}"
            return None
        self.failures = 0
        self.calls.append(("observe", time.perf_counter() - t0))
        return res.parsed_output

    def plan(self, goal: str, vocabulary: str, scene: str, memory: str,
             jpeg: bytes | None = None) -> list[dict] | None:
        if not self.available:
            return None
        text = (f"Spoken goal: {goal}\n\n"
                f"What the lamp can see right now: {scene or 'nothing described yet'}\n"
                f"What it remembers seeing: {memory}\n\n"
                f"Available actions:\n{vocabulary}\n\n"
                "Return the steps to carry out the goal. Use at most seven steps. "
                "look_at takes a target of 'person', 'scene', or a remembered label. "
                "Speak through say steps, keeping each to one short sentence.")
        content: list[dict] = []
        if jpeg is not None:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.b64encode(jpeg).decode()}})
        content.append({"type": "text", "text": text})
        t0 = time.perf_counter()
        try:
            res = self.client.messages.parse(
                model=self.model,
                max_tokens=1024,
                output_config={"effort": EFFORT},
                system=SYSTEM,
                messages=[{"role": "user", "content": content}],
                output_format=Plan,
            )
        except Exception as exc:
            self.failures += 1
            self.reason = f"{type(exc).__name__}: {exc}"
            return None
        self.failures = 0
        self.calls.append(("plan", time.perf_counter() - t0))
        out = []
        for step in res.parsed_output.steps:
            d = {"action": step.action}
            for key in ("target", "text", "name", "note", "summary"):
                if getattr(step, key):
                    d[key] = getattr(step, key)
            for key in ("intensity", "warm", "seconds", "amp"):
                if getattr(step, key) is not None:
                    d[key] = getattr(step, key)
            out.append(d)
        return out

    def turn(self, said: str, jpeg: bytes | None, memory: str, history: str,
             vocabulary: str) -> Turn | None:
        if not self.available:
            return None
        text = (f"Recent conversation:\n{history or '(none yet)'}\n\n"
                f"Already in memory: {memory}\n\n"
                f"The person just said: \"{said}\"\n\n"
                "The image is what the lamp's camera sees right now. Reply as the lamp. "
                "List the notable objects you can see so it remembers them. If the person "
                "asked it to physically look at, find, point at or light up something, give "
                "actions from this list, otherwise leave actions empty:\n"
                f"{vocabulary}\n"
                "look_at takes a target of 'person', 'scene', or a remembered label.")
        content: list[dict] = []
        if jpeg is not None:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.b64encode(jpeg).decode()}})
        content.append({"type": "text", "text": text})
        t0 = time.perf_counter()
        try:
            extra = {}
            if self.model in FALLBACK_MODELS:
                extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
            res = self.client.beta.messages.parse(
                model=self.model,
                max_tokens=2048,
                output_config={"effort": EFFORT},
                system=SYSTEM,
                messages=[{"role": "user", "content": content}],
                output_format=Turn,
                **extra,
            )
        except Exception as exc:
            self.failures += 1
            self.reason = f"{type(exc).__name__}: {exc}"
            return None
        if res.stop_reason == "refusal" or res.parsed_output is None:
            self.reason = f"no usable turn, stop_reason {res.stop_reason}"
            return None
        self.failures = 0
        self.calls.append(("turn", time.perf_counter() - t0))
        return res.parsed_output

    @staticmethod
    def steps_to_dicts(steps: list[PlanStep]) -> list[dict]:
        out = []
        for step in steps:
            d = {"action": step.action}
            for key in ("target", "text", "name", "note", "summary"):
                if getattr(step, key):
                    d[key] = getattr(step, key)
            for key in ("intensity", "warm", "seconds", "amp"):
                if getattr(step, key) is not None:
                    d[key] = getattr(step, key)
            out.append(d)
        return out

    def latency(self, kind: str) -> float:
        times = [t for k, t in self.calls if k == kind]
        return sum(times) / len(times) if times else 0.0
