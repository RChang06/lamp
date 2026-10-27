"""the action vocabulary: the only things a plan is allowed to say.

this is the model-to-action boundary the challenge asks to be defined. a language model
never emits joint angles, timings or audio. it emits a short list of these actions, each
one validated here before the director will run it, and the body decides how any of them
actually look. that keeps three properties worth having: a bad or hostile plan cannot
drive the arm somewhere unsafe, the plan is readable in a log and reproducible in a test,
and swapping the planner for a different model, or for no model at all, changes nothing
downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# name -> (required args, optional args with defaults)
SCHEMA: dict[str, tuple[tuple[str, ...], dict[str, Any]]] = {
    "look_at":  (("target",), {}),              # a remembered label, or "person", or "scene"
    "observe":  ((), {"note": ""}),             # take a frame, describe it, remember it
    "scan":     ((), {"amp": 1.0}),             # sweep the head to take the scene in
    "light":    ((), {"intensity": 0.8, "warm": 0.85}),
    "gesture":  (("name",), {"amp": 1.0}),
    "say":      (("text",), {}),
    "sound":    (("name",), {}),
    "music":    ((), {"on": True}),
    "wait":     ((), {"seconds": 0.5}),
    "done":     ((), {"summary": ""}),
}

MAX_STEPS = 10
MAX_SAY_CHARS = 240


@dataclass
class Action:
    kind: str
    args: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.kind == "say":
            return f'say "{self.args["text"][:48]}"'
        inner = " ".join(f"{k}={v}" for k, v in self.args.items())
        return f"{self.kind}{(' ' + inner) if inner else ''}"


class PlanError(ValueError):
    pass


def validate(raw: list[dict]) -> list[Action]:
    """turn whatever the planner produced into actions, or refuse the whole plan.

    refusing the whole plan rather than skipping bad steps is on purpose: a plan with a
    step dropped out of the middle can be worse than no plan, for instance losing the
    observe that was supposed to check the result before saying it was done.
    """
    if not isinstance(raw, list):
        raise PlanError("a plan must be a list of actions")
    if len(raw) > MAX_STEPS:
        raise PlanError(f"plan has {len(raw)} steps, the limit is {MAX_STEPS}")
    out: list[Action] = []
    for i, step in enumerate(raw):
        if not isinstance(step, dict):
            raise PlanError(f"step {i} is not an object")
        kind = step.get("action") or step.get("kind")
        if kind not in SCHEMA:
            raise PlanError(f"step {i}: unknown action {kind!r}")
        required, optional = SCHEMA[kind]
        args = {k: v for k, v in step.items() if k not in ("action", "kind")}
        for key in required:
            if key not in args:
                raise PlanError(f"step {i}: {kind} needs {key}")
        merged = dict(optional)
        merged.update({k: v for k, v in args.items() if k in required or k in optional})
        if kind == "say":
            merged["text"] = str(merged["text"])[:MAX_SAY_CHARS]
        if kind == "wait":
            merged["seconds"] = max(0.0, min(3.0, float(merged["seconds"])))
        out.append(Action(kind, merged))
    return out


def prompt_reference() -> str:
    """the vocabulary as a line for a planner prompt, generated from the schema itself so
    the prompt cannot drift away from what the validator accepts."""
    lines = []
    for kind, (required, optional) in SCHEMA.items():
        keys = list(required) + [f"{k}?" for k in optional]
        lines.append(f"  {kind}({', '.join(keys)})")
    return "\n".join(lines)
