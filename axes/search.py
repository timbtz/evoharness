"""S3 Search: greedy (1+1) | islands | staged.

Greedy ← LLaMEA's plus-strategy elitism (MIT). Islands ← FunSearch programs_database
(Apache-2.0) compressed: 4 flat islands, EoH-style keep-first-per-unique-score, rank-weighted
parent sampling, ring migration of the global best every 5 generations. No resets, no clusters.
Staged ← stagnation-driven exploration (user design, 2026-07-20): exploit until STALL barren
generations, then adopt a from-scratch redesign as the new branch root even if it scores worse;
operator directives (simplify/combine/wildcard) every OP_EVERY generations of a dry spell.
"""
from __future__ import annotations

import random
from dataclasses import replace

from core.candidate import Candidate, Pool, PromptSection


class Greedy11:
    """(1+1): one incumbent; the gate ensured any inserted child beats or ties its
    parent. Ties follow the NEWEST tied candidate so the incumbent drifts across
    score plateaus instead of freezing on the seed."""

    def __init__(self, key, seed: int = 0):
        self.key = key

    def select_parents(self, pool: Pool, rng: random.Random) -> list[Candidate]:
        best = max(pool.all,
                   key=lambda c: (self.key(c, pool, "train"), c.meta.get("gen", 0)))
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
        # the "all"-island seed keeps every island non-empty; the fallback guards
        # against pathological pool states ever crashing parent selection
        members = self._members(pool, island) or [pool.best("train")]
        members = sorted(
            members,
            key=lambda c: (self.key(c, pool, "train"), c.meta.get("gen", 0)),
            reverse=True,
        )
        # EoH rank weights: probs ~ 1/(rank+1+len)
        weights = [1.0 / (rank + 1 + len(members)) for rank in range(len(members))]
        return [rng.choices(members, weights=weights, k=1)[0]]

    def insert(self, pool: Pool, cand: Candidate) -> None:
        parent = pool.parent_of(cand)
        if parent is None:
            cand.meta["island"] = "all"  # root seed: shared member of every island
            pool.add(cand)
            return
        island = parent.meta.get("island", self.t % self.K)
        cand.meta["island"] = island if island != "all" else self.t % self.K
        island = cand.meta["island"]
        pool.add(cand)
        # prune: keep newest-per-unique-score (EoH keep-one; newest so accepted
        # ties advance the frontier), best CAP by objective
        members = sorted(
            self._members(pool, island),
            key=lambda c: (self.key(c, pool, "train"), c.meta.get("gen", 0)),
            reverse=True,
        )
        seen: set[float] = set()
        kept = 0
        for c in members:
            if c.meta.get("island") == "all":
                continue  # the shared root seed is never pruned nor counted
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


class Staged:
    """Stagnation-driven pipeline: climb from the incumbent; after STALL generations
    without a new GLOBAL best, demand a full redesign and adopt it as the new branch
    root even when it scores worse — it gets its own STALL budget to prove itself.
    During a dry spell an operator directive rotates in every OP_EVERY generations.
    The global best is never lost: pool.best() ignores branches (elitism at run end),
    and every candidate carries its branch id in meta for the ledger."""

    STALL, OP_EVERY = 50, 25
    OPERATORS = (
        "Simplify: make the program substantially shorter and clearer without losing "
        "score. Delete mechanisms that don't pay for themselves.",
        "Combine: merge the strongest mechanisms seen across earlier attempts into one "
        "coherent design instead of tweaking the incumbent.",
        "Wildcard: pick one promising idea from the literature you know (a different "
        "construction, neighborhood, acceptance rule, or data structure) and integrate "
        "it boldly, even if it restructures the program.",
    )
    REDESIGN = (
        "This line of improvements has plateaued: {stall} generations without beating "
        "the best score {best:.4f}. Do a FULL redesign now: choose a fundamentally "
        "different algorithmic approach and write it from scratch. Do not preserve the "
        "incumbent's structure; reuse isolated pieces only where they clearly serve the "
        "new design. A redesign is kept as the new working base even if it starts out "
        "scoring worse, so prioritize long-term potential over safe tweaks."
    )

    def __init__(self, key, seed: int = 0):
        self.key = key
        self.branch, self.stall, self.best = 0, 0, float("-inf")
        self.redesigning = False

    def select_parents(self, pool: Pool, rng: random.Random) -> list[Candidate]:
        gb = max(self.key(c, pool, "train") for c in pool.all)
        if gb > self.best:
            self.best, self.stall = gb, 0
        else:
            self.stall += 1
        self.redesigning = self.stall >= self.STALL  # stays on until a redesign runs
        members = [c for c in pool.all if c.meta.get("branch", 0) == self.branch] or pool.all
        return [max(members, key=lambda c: (self.key(c, pool, "train"), c.meta.get("gen", 0)))]

    def prompt_sections(self) -> list[PromptSection]:
        if self.redesigning:
            return [PromptSection("Directive: full redesign",
                                  self.REDESIGN.format(stall=self.stall, best=self.best))]
        if self.stall and self.stall % self.OP_EVERY == 0:
            op = self.OPERATORS[(self.stall // self.OP_EVERY) % len(self.OPERATORS)]
            return [PromptSection("Directive", op)]
        return []

    def adopt(self, cand: Candidate) -> bool:
        """Gate bypass for redesign generations: a runnable redesign becomes the new
        branch root regardless of score."""
        if not self.redesigning or cand.meta.get("error"):
            return False
        self.branch += 1
        self.stall = 0
        self.redesigning = False
        cand.meta.update(branch=self.branch, branch_root=True)
        return True

    def insert(self, pool: Pool, cand: Candidate) -> None:
        cand.meta.setdefault("branch", self.branch)
        pool.add(cand)


