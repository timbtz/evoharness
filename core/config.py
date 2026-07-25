"""Flat run config: 5 switches + models + budgets + objective. One dict, JSON-serializable."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SWITCHES: dict[str, tuple[str, ...]] = {
    "feedback": ("score_only", "reflections", "memory"),
    "gate": ("public_only", "holdout"),
    "search": ("greedy", "islands", "staged"),
    "knowledge": ("off", "wiki_fs", "web"),
    "roles": ("single_strong", "split_roles"),
}
OBJECTIVES = ("quality", "quality_per_dollar", "time_capped")
TASKS = ("binpacking", "circles", "tsp", "matmul_c", "cvrp", "stellar_p2")


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
    resume_from: str | None = None  # run_id (or "file:<path>"): start from that run's
    # best accepted candidate / that file's code
    refiner: str | None = None  # claude model id: every refiner_every writer candidates
    # a headless Claude session sees the whole window (ideas, feedback, code) + the wiki
    # and synthesizes ONE v2 (evaluated + gated like any candidate; id suffix "f")
    refiner_every: int = 5  # refiner cadence: one session per N writer candidates
    stall_stop: int | None = None  # stop after N consecutive candidates without a new
    # combined (train+val) best — the DAG campaign's branch-termination rule
    review_every: int | None = None  # override MemoryWiki.REVIEW_EVERY for this run
    analyst: str | None = None  # claude model id: independent in-run analysis session
    # every analyst_every candidates; writes new-ideas/analyst-*.md to the wiki
    analyst_every: int = 10
    analyst_web: bool = False  # give the analyst WebSearch/WebFetch: it can pull
    # approaches beyond the wiki (papers, write-ups) into its proposals
    analyst_inject: int = 0  # >0: the analyst may WRITE up to N candidate modules
    # to <run_dir>/INJECT/ per session; the loop evaluates + gates them like
    # writer candidates at the next generation (ids c<gen>i<k>)
    merge_from: str | None = None  # run_id (or "file:<path>"): that run's best program is
    # shown every generation as "Approach B" with a merge directive (branch-merge runs)

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}, expected one of {TASKS}")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"unknown objective {self.objective!r}, expected one of {OBJECTIVES}")
        for axis, opts in SWITCHES.items():
            val = self.switches.setdefault(axis, opts[0])
            if val not in opts:
                raise ValueError(f"switch {axis}={val!r}, expected one of {opts}")
        if self.switches["knowledge"] == "web" and self.switches["feedback"] != "memory":
            raise ValueError("knowledge=web requires feedback=memory "
                             "(research pages are delivered via the memory wiki)")

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
