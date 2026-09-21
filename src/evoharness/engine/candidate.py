"""Candidate + the small shared value types (EvalResult, PromptSection, Pool)."""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

_ids = itertools.count()


@dataclass
class EvalResult:
    score: float                       # higher is better (tasks negate internally if needed)
    metrics: dict = field(default_factory=dict)
    error: str | None = None
    seconds: float = 0.0


@dataclass
class PromptSection:
    title: str
    text: str

    def render(self) -> str:
        return f"## {self.title}\n{self.text}"


@dataclass
class Candidate:
    code: str
    parent_id: str | None = None
    id: str = field(default_factory=lambda: f"c{next(_ids):04d}")
    scores: dict[str, float] = field(default_factory=dict)  # split -> score (higher better)
    cost: dict[str, float] = field(default_factory=dict)    # usd, tokens (cumulative at birth)
    meta: dict = field(default_factory=dict)                # gen, island, error, seconds, novelty

    def to_dict(self) -> dict:
        return asdict(self)

    def score(self, split: str = "train") -> float:
        return self.scores.get(split, float("-inf"))


class Pool:
    """Accepted candidates. Structure (islands, incumbents) lives in cand.meta, owned by Search."""

    def __init__(self):
        self.all: list[Candidate] = []
        self.by_id: dict[str, Candidate] = {}

    def add(self, cand: Candidate) -> None:
        self.all.append(cand)
        self.by_id[cand.id] = cand

    def parent_of(self, cand: Candidate) -> Candidate | None:
        return self.by_id.get(cand.parent_id) if cand.parent_id else None

    def best(self, split: str = "train") -> Candidate | None:
        live = [c for c in self.all if not c.meta.get("pruned")]
        return max(live, key=lambda c: c.score(split), default=None)
