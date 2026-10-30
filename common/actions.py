from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA: dict[str, tuple[tuple[str, ...], dict[str, Any]]] = {
    "look_at":  (("target",), {}),
    "observe":  ((), {"note": ""}),
    "scan":     ((), {"amp": 1.0}),
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
    lines = []
    for kind, (required, optional) in SCHEMA.items():
        keys = list(required) + [f"{k}?" for k in optional]
        lines.append(f"  {kind}({', '.join(keys)})")
    return "\n".join(lines)
