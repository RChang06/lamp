"""the one remote step: look at a frame, and turn a spoken goal into a short action plan.

everything that has to be fast is local. face tracking, the engagement decision, speech
in, speech out and all motion stay on the machine, so the character keeps reacting at
camera rate with the network unplugged. what goes out is the part that genuinely needs a
large model: naming what is in one still frame, and choosing a sequence of lamp actions
for a spoken goal.

what is sent: one jpeg of the camera frame at the moment of an observe action, the text of
the current request, and the short list of labels already in scene memory. no audio ever
leaves the machine, no continuous video, and nothing is sent at all unless an observe or a
goal is in flight. with no api key present the character falls back to asking the person
what it is looking at, which needs no network and no model.

the model returns a validated structure, and the plan is then checked again against the
action vocabulary before anything moves. a wrong or hostile answer can only ever be a
refused plan.
"""
from __future__ import annotations

import base64
import os
import time

from pydantic import BaseModel, Field

DEFAULT_MODEL = os.environ.get("LAMP_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("LAMP_EFFORT", "low")       # a plan is short, latency is the cost


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


SYSTEM = """You are the mind of a small desk lamp robot with five joints: it can turn its
base, raise and fold its arm, and turn and tilt its head. Its head is a lamp, so its light
is part of how it expresses itself. It speaks in short, warm, plain sentences, never more
than two, and never describes its own joints or internal state.

You will be asked either to describe what the camera sees, or to turn a spoken goal into a
sequence of actions. Keep plans short. When a goal asks you to find, check or point at
something, include an observe step after moving and before the final done step, because
the lamp must actually look again before it claims a goal is finished."""


class Cloud:
    """wraps the remote calls. absent credentials is a normal state, not an error."""

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
        # the sdk constructs happily with no credentials and only fails at request time,
        # so resolve one now. finding this out in the middle of a demonstration is worse
        # than knowing at startup which mode the character is in.
        if not (getattr(client, "api_key", None) or getattr(client, "auth_token", None)
                or os.environ.get("LAMP_FORCE_CLOUD")):
            self.reason = "no ANTHROPIC_API_KEY set, using the local path"
            return
        self.client = client

    @property
    def available(self) -> bool:
        # two consecutive failures and it stays local for the rest of the session rather
        # than pausing on every observe to time out again
        return self.client is not None and self.failures < 2

    # ---- vision -----------------------------------------------------------------------
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

    # ---- planning ---------------------------------------------------------------------
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

    # ---- conversation ----------------------------------------------------------------
    def reply(self, said: str, memory: str) -> str:
        """one short spoken reply. used only for talk that is not a goal or a recall."""
        if not self.available:
            return ""
        t0 = time.perf_counter()
        try:
            res = self.client.messages.create(
                model=self.model,
                max_tokens=160,
                output_config={"effort": EFFORT},
                system=SYSTEM,
                messages=[{"role": "user", "content":
                           f"It remembers seeing: {memory}. "
                           f"The person said: {said}. "
                           "Reply as the lamp, one short sentence."}],
            )
        except Exception as exc:
            self.failures += 1
            self.reason = f"{type(exc).__name__}: {exc}"
            return ""
        self.calls.append(("reply", time.perf_counter() - t0))
        return "".join(b.text for b in res.content if getattr(b, "type", "") == "text").strip()

    def latency(self, kind: str) -> float:
        times = [t for k, t in self.calls if k == kind]
        return sum(times) / len(times) if times else 0.0
