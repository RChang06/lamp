"""what the character saw, and how it answers about it later.

deliberately a jsonl file and a token overlap search rather than embeddings and a vector
store. the whole memory of a demonstration is a few dozen short records, so a vector
index would be slower to load than the data it indexes and would add a model to the
deployment for no measurable gain. the retrieval that matters here is "which of the six
things i have seen is this question about", and overlap with a recency prior answers that
at this scale. the interface is narrow enough that swapping in embeddings later touches
one method.

records are appended to disk as they are made, so the memory survives a restart and the
demonstration can be inspected afterwards, which is also the evidence trail.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import time
from dataclasses import asdict, dataclass, field

STOP = {"the", "a", "an", "is", "was", "were", "are", "do", "does", "did", "what", "where",
        "when", "which", "that", "this", "it", "of", "on", "in", "at", "to", "my", "your",
        "i", "you", "me", "and", "or", "there", "here", "see", "saw", "tell", "about",
        "can", "could", "would", "please", "again", "remember", "did", "have", "has",
        "earlier", "before", "today", "ago", "then", "anything", "else", "yet", "just"}


def tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1}


@dataclass
class Observation:
    label: str                  # the short name, "a blue mug"
    detail: str = ""            # anything else worth keeping, colour, position, state
    az: float = 0.0             # where the character was looking when it saw this
    el: float = 0.0
    source: str = "vision"      # vision, speech, or inferred
    frame: str = ""             # path to the saved frame, if one was kept
    t: float = field(default_factory=time.time)

    def phrase(self) -> str:
        return f"{self.label}{(', ' + self.detail) if self.detail else ''}"


class SceneMemory:
    def __init__(self, path: str = "run/memory.jsonl"):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.items: list[Observation] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    self.items.append(Observation(**json.loads(line)))
                except Exception:
                    continue

    def note(self, obs: Observation) -> Observation:
        # if we have seen this label already, update it rather than piling up duplicates
        for old in self.items:
            if old.label.lower() == obs.label.lower():
                old.detail = obs.detail or old.detail
                old.az, old.el, old.t = obs.az, obs.el, obs.t
                self._append(old)
                return old
        self.items.append(obs)
        self._append(obs)
        return obs

    def _append(self, obs: Observation) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(obs)) + "\n")

    def recent(self, n: int = 5) -> list[Observation]:
        return sorted(self.items, key=lambda o: o.t, reverse=True)[:n]

    def find(self, query: str) -> tuple[Observation | None, float]:
        """best match for a spoken question, with its score, so the caller can decline."""
        q = tokens(query)
        if not q or not self.items:
            return None, 0.0
        now = time.time()
        best, best_score = None, 0.0
        for obs in self.items:
            have = tokens(obs.phrase())
            if not have:
                continue
            overlap = len(q & have) / math.sqrt(len(q) * len(have))
            # a mild recency prior, ten minutes old is worth about nine tenths of now
            recency = 1.0 / (1.0 + (now - obs.t) / 6000.0)
            score = overlap * recency
            if score > best_score:
                best, best_score = obs, score
        return best, best_score

    def describe(self, n: int = 6) -> str:
        """what the character has seen, as a line for a prompt or a spoken summary."""
        if not self.items:
            return "nothing yet"
        return "; ".join(o.phrase() for o in self.recent(n))
