"""S3 Search: greedy (1+1) | islands.

Greedy ← LLaMEA's plus-strategy elitism (MIT). Islands ← FunSearch programs_database
(Apache-2.0) compressed: 4 flat islands, EoH-style keep-first-per-unique-score, rank-weighted
parent sampling, ring migration of the global best every 5 generations. No resets, no clusters.
"""
from __future__ import annotations

import random
from dataclasses import replace

from core.candidate import Candidate, Pool


class Greedy11:
    """(1+1): one incumbent; the gate already ensured any inserted child beats its parent."""

    def __init__(self, key, seed: int = 0):
        self.key = key

    def select_parents(self, pool: Pool, rng: random.Random) -> list[Candidate]:
        best = max(pool.all, key=lambda c: self.key(c, pool, "train"))
        return [best]

    def insert(self, pool: Pool, cand: Candidate) -> None:
        pool.add(cand)


class Islands:
    K, CAP, MIGRATE_EVERY = 4, 8, 5

    def __init__(self, key, seed: int = 0):
        self.key = key
        self.t = 0  # generation counter (one select_parents call per generation)

    def _members(self, pool: Pool, island: int) -> list[Candidate]:
        return [
            c for c in pool.all
            if not c.meta.get("pruned") and c.meta.get("island") in (island, "all")
        ]

    def select_parents(self, pool: Pool, rng: random.Random) -> list[Candidate]:
        self.t += 1
        island = self.t % self.K
        members = sorted(
            self._members(pool, island),
            key=lambda c: self.key(c, pool, "train"), reverse=True,
        )
        # EoH rank weights: probs ~ 1/(rank+1+len)
        weights = [1.0 / (rank + 1 + len(members)) for rank in range(len(members))]
        parent = rng.choices(members, weights=weights, k=1)[0]
        parent.meta.setdefault("island", island)  # pin the shared seed on first use
        return [parent]

    def insert(self, pool: Pool, cand: Candidate) -> None:
        parent = pool.parent_of(cand)
        island = parent.meta.get("island", self.t % self.K) if parent else "all"
        island = island if island != "all" else self.t % self.K
        cand.meta["island"] = island
        pool.add(cand)
        # prune: keep first-per-unique-score (EoH), best CAP by objective
        members = sorted(
            self._members(pool, island),
            key=lambda c: self.key(c, pool, "train"), reverse=True,
        )
        seen: set[float] = set()
        kept = 0
        for c in members:
            s = round(c.score("train"), 5)
            if s in seen or kept >= self.CAP:
                c.meta["pruned"] = True
            else:
                seen.add(s)
                kept += 1
        if self.t % self.MIGRATE_EVERY == 0:
            self._migrate(pool)

    def _migrate(self, pool: Pool) -> None:
        best = max(
            (c for c in pool.all if not c.meta.get("pruned")),
            key=lambda c: self.key(c, pool, "train"),
        )
        target = (best.meta.get("island", 0) + 1) % self.K if best.meta.get("island") != "all" else 0
        if any(c.score("train") == best.score("train") for c in self._members(pool, target)):
            return  # already represented there
        migrant = replace(
            best, id=f"{best.id}m{self.t}", parent_id=best.id,
            scores=dict(best.scores), cost=dict(best.cost),
            meta={"island": target, "migrant": True, "gen": self.t},
        )
        pool.add(migrant)
