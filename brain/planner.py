"""routing what was said, and turning a goal into actions without the network if need be.

two layers, and the cheap one answers most of the demonstration. intent routing is a
handful of patterns over the transcript: greetings, questions about something remembered,
light requests, goodbyes. only an actual goal, something that needs the scene looked at
and a sequence chosen, goes out to the model.

the local planner is not a stub. with no api key the character still engages, talks,
remembers and carries out the goals it recognises. what it loses is the ability to name
objects it has never been told about, and there it does the honest thing: it looks at the
thing, says it cannot tell what it is, and asks. the answer goes into memory from speech
instead of from vision, and every later question about that object works identically.
"""
from __future__ import annotations

import re

from brain.memory import SceneMemory
from common import actions as A

GREET = re.compile(r"\b(hi|hey|hello|morning|evening|there you are|you awake)\b", re.I)
BYE = re.compile(r"\b(bye|goodnight|good night|see you|that's all|thats all|go to sleep)\b", re.I)
THANKS = re.compile(r"\b(thanks|thank you|cheers|nice one)\b", re.I)
WHOAMI = re.compile(r"\b(who are you|what are you|your name)\b", re.I)
RECALL = re.compile(r"\b(what did you see|what do you remember|remember (seeing|what)|"
                    r"what was|where (is|was|did)|did you see)\b", re.I)
LOOKNOW = re.compile(r"\b(what (do you see|can you see|is that|is this)|look at (this|that)|"
                     r"have a look|check this out)\b", re.I)
BRIGHTER = re.compile(r"\b(brighter|more light|turn it up|light me up)\b", re.I)
DIMMER = re.compile(r"\b(dimmer|darker|too bright|turn it down)\b", re.I)
GOAL = re.compile(r"\b(point at|shine on|light up the|find|look for|check (if|whether)|"
                  r"can you (find|point|shine|check)|show me)\b", re.I)

# a goal usually names its object after the verb
GOAL_OBJECT = re.compile(r"\b(?:point at|shine on|light up(?: the)?|find|look for|show me)\s+"
                         r"(?:the\s+|my\s+|a\s+|an\s+)?([a-z0-9 ]{2,30})", re.I)


class Intent:
    GREET = "greet"
    BYE = "bye"
    THANKS = "thanks"
    WHOAMI = "whoami"
    RECALL = "recall"
    LOOK_NOW = "look_now"
    LIGHT = "light"
    GOAL = "goal"
    CHITCHAT = "chitchat"


def route(text: str) -> tuple[str, dict]:
    t = text.strip()
    if not t:
        return Intent.CHITCHAT, {}
    if BYE.search(t):
        return Intent.BYE, {}
    if WHOAMI.search(t):
        return Intent.WHOAMI, {}
    if RECALL.search(t):
        return Intent.RECALL, {}
    if BRIGHTER.search(t):
        return Intent.LIGHT, {"intensity": 1.0}
    if DIMMER.search(t):
        return Intent.LIGHT, {"intensity": 0.35}
    if GOAL.search(t):
        m = GOAL_OBJECT.search(t)
        return Intent.GOAL, {"object": (m.group(1).strip() if m else "")}
    if LOOKNOW.search(t):
        return Intent.LOOK_NOW, {}
    if GREET.search(t):
        return Intent.GREET, {}
    if THANKS.search(t):
        return Intent.THANKS, {}
    return Intent.CHITCHAT, {}


class LocalPlanner:
    """plans the goals it recognises, with no model and no network."""

    def __init__(self, memory: SceneMemory):
        self.memory = memory

    def plan(self, goal: str, obj: str) -> list[dict]:
        known, score = self.memory.find(obj or goal)
        if known is not None and score > 0.35:
            # it has seen this before, so it can go straight there and check
            return [
                {"action": "look_at", "target": known.label},
                {"action": "light", "intensity": 1.0, "warm": 0.9},
                {"action": "observe", "note": obj or known.label},
                {"action": "say", "text": f"There it is, {known.label}."},
                {"action": "done", "summary": f"pointed out {known.label}"},
            ]
        # it has not, so look around, then admit it and ask, which is how it learns
        return [
            {"action": "scan", "amp": 0.9},
            {"action": "observe", "note": obj},
            {"action": "say", "text": (f"I had a look for {obj}, but I am not sure I can "
                                       f"tell what I am seeing. What is it?" if obj else
                                       "I had a look, but I am not sure what I am seeing. "
                                       "What is it?")},
            {"action": "done", "summary": "asked for help identifying the target"},
        ]


class Planner:
    """prefers the model, falls back to the local planner, validates either way."""

    def __init__(self, memory: SceneMemory, cloud=None):
        self.memory = memory
        self.local = LocalPlanner(memory)
        self.cloud = cloud
        self.last_source = "none"

    def plan(self, goal: str, obj: str, scene: str = "", jpeg: bytes | None = None):
        if self.cloud is not None and self.cloud.available:
            raw = self.cloud.plan(goal, A.prompt_reference(), scene, self.memory.describe(), jpeg)
            if raw:
                try:
                    plan = A.validate(raw)
                    self.last_source = "cloud"
                    return self.require_observe(plan)
                except A.PlanError:
                    pass          # a refused plan falls through to the local one
        self.last_source = "local"
        return self.require_observe(A.validate(self.local.plan(goal, obj)))

    @staticmethod
    def require_observe(plan: list[A.Action]) -> list[A.Action]:
        """the challenge asks the character to look again before completing a goal, so a
        plan that would finish without looking gets an observe inserted rather than being
        allowed to claim success on stale information."""
        kinds = [a.kind for a in plan]
        if "done" in kinds and "observe" not in kinds:
            at = kinds.index("done")
            plan.insert(at, A.Action("observe", {"note": "checking before finishing"}))
        return plan
