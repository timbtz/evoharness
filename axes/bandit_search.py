"""S3 Search option: bandit_fork (lineage bandit + fork-on-stall).

AIDE²'s evolved policy (the one thing its outer loop kept): treat each lineage as
a bandit arm, allocate pulls between them, and when the best lineage STALLS, FORK
— copy the global best into a fresh lineage under a different strategy and fund it
as a new arm. The two biggest jumps in our own campaign (+0.036, +0.028) were
exactly fork-on-stall moves executed by the refiner by luck; this makes them policy.

Our Staged axis is the degenerate case of this: one lineage, stall_stop, then a
human picks the next pin. bandit_fork keeps several lineages alive and forks
automatically.

This is a minimal, robust instantiation: ε-greedy lineage allocation (SeaEVO's
ε=0.2 — exploit the global-best's lineage, else explore a random one) plus
fork-on-stall (a redesign directive when the global best is barren for STALL
generations; a runnable redesign is adopted as a new lineage root even if it
scores worse, mirroring Staged.adopt). UCB-over-lineages and reward attribution
are deliberate omissions for v1 — knobs to add once the mechanism proves out.

Default-off and additive; the running stellar run (search=staged) is unaffected.
"""
from __future__ import annotations

import random

from core.candidate import Candidate, Pool, PromptSection
from axes.search import Staged  # reuse the proven redesign/operator prompt text


class BanditFork:
    EPS = 0.2        # explore probability (SeaEVO)
    STALL = 12       # barren generations over the global best before a fork
    OPERATORS = Staged.OPERATORS
    REDESIGN = Staged.REDESIGN

    def __init__(self, key, seed: int = 0):
        self.key = key
        self.best = float("-inf")
        self.global_stall = 0
        self.cur_branch = 0
        self.next_branch = 0
        self.fork_due = False

    def _global_best(self, pool: Pool) -> Candidate | None:
        live = [c for c in pool.all if not c.meta.get("pruned")]
        return max(live, key=lambda c: (self.key(c, pool, "train"), c.meta.get("gen", 0)),
                   default=None)

    def select_parents(self, pool: Pool, rng: random.Random) -> list[Candidate]:
        gb = self._global_best(pool)
        gb_key = self.key(gb, pool, "train") if gb else float("-inf")
        if gb_key > self.best:
            self.best, self.global_stall = gb_key, 0
        else:
            self.global_stall += 1
        self.fork_due = self.global_stall >= self.STALL

        branches = sorted({c.meta.get("branch", 0) for c in pool.all}) or [0]
        if self.fork_due and gb is not None:
            self.cur_branch = gb.meta.get("branch", 0)   # redesign FROM the global best
        elif rng.random() < self.EPS or len(branches) <= 1:
            self.cur_branch = rng.choice(branches)        # explore a random lineage
        else:
            self.cur_branch = gb.meta.get("branch", 0)    # exploit the best lineage
        members = [c for c in pool.all
                   if c.meta.get("branch", 0) == self.cur_branch and not c.meta.get("pruned")] \
            or pool.all
        return [max(members, key=lambda c: (self.key(c, pool, "train"), c.meta.get("gen", 0)))]

    def prompt_sections(self) -> list[PromptSection]:
        if self.fork_due:
            return [PromptSection("Directive: fork — start a fresh lineage with a full redesign",
                                  self.REDESIGN.format(stall=self.global_stall, best=self.best))]
        if self.global_stall and self.global_stall % 5 == 0:
            op = self.OPERATORS[(self.global_stall // 5) % len(self.OPERATORS)]
            return [PromptSection("Directive", op)]
        return []

    def adopt(self, cand: Candidate) -> bool:
        """A fork-generation redesign becomes the root of a NEW lineage regardless
        of score, so the search gets a fresh STALL budget to prove a different
        approach. The global best is never lost: pool.best() ignores branches."""
        if not self.fork_due or cand.meta.get("error"):
            return False
        self.next_branch += 1
        cand.meta.update(branch=self.next_branch, branch_root=True)
        self.fork_due = False
        self.global_stall = 0
        return True

    def insert(self, pool: Pool, cand: Candidate) -> None:
        cand.meta.setdefault("branch", self.cur_branch)
        pool.add(cand)
