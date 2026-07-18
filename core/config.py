"""Flat run config: 5 switches + models + budgets + objective. One dict, JSON-serializable."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SWITCHES: dict[str, tuple[str, ...]] = {
    "feedback": ("score_only", "reflections"),
    "gate": ("public_only", "holdout"),
    "search": ("greedy", "islands"),
    "knowledge": ("off", "wiki_fs"),
    "roles": ("single_strong", "split_roles"),
}
OBJECTIVES = ("quality", "quality_per_dollar", "time_capped")
TASKS = ("binpacking", "circles", "tsp")


@dataclass
class Config:
    task: str = "binpacking"
    seed: int = 0
    switches: dict[str, str] = field(
        default_factory=lambda: {k: opts[0] for k, opts in SWITCHES.items()}
    )
    models: dict[str, str] = field(
        default_factory=lambda: {"strong": "glm-5.2", "cheap": "glm-4.7"}
    )
    budget: dict = field(
        default_factory=lambda: {"max_usd": 2.0, "max_calls": 120, "max_seconds": 1800}
    )
    objective: str = "quality"
    temperatures: dict = field(
        default_factory=lambda: {"writer": 0.9, "reflect": 0.4, "judge": 0.2}
    )
    generations: int = 1000  # upper bound; budget stops the run first
    wiki_mode: str = "inject"  # S4=wiki_fs sub-mode: "inject" (top-k pages) | "tool" (read_wiki calls)

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}, expected one of {TASKS}")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"unknown objective {self.objective!r}, expected one of {OBJECTIVES}")
        for axis, opts in SWITCHES.items():
            val = self.switches.setdefault(axis, opts[0])
            if val not in opts:
                raise ValueError(f"switch {axis}={val!r}, expected one of {opts}")

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return asdict(self)
