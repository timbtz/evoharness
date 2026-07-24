"""S2 Gate: public_only | holdout. Correctness (ran without error) is a precondition, not a switch."""
from __future__ import annotations

from core.candidate import Candidate, Pool


class PublicOnly:
    """Accept if the train-split objective improves on the parent."""

    split = "train"

    def __init__(self, task, key):
        self.task, self.key = task, key

    def _ensure_score(self, cand: Candidate) -> None:
        pass  # train is always evaluated by the loop

    def accept(self, cand: Candidate, pool: Pool) -> bool:
        if cand.meta.get("error"):
            return False
        self._ensure_score(cand)
        parent = pool.parent_of(cand)
        if parent is None:
            return True
        self._ensure_score(parent)
        child, par = self.key(cand, pool, self.split), self.key(parent, pool, self.split)
        eps = getattr(self.task, "noise", {}).get(self.split, 0.0)
        if eps and 0 < abs(child - par) <= eps:
            # margin within eval noise (anytime tasks): re-run, decide on the median
            scores = [cand.scores[self.split]] + [
                self.task.evaluate(cand.code, self.split).score for _ in range(2)]
            cand.scores[self.split] = sorted(scores)[1]
            child = self.key(cand, pool, self.split)
        if child != par:
            return child > par
        # exact tie: accept genuinely different code (novelty > 0) so the search
        # can drift across score plateaus instead of freezing on the incumbent —
        # but a tie on the gated split must not smuggle in a train regression
        # (s37: val-tie band admitted train -0.59; §6.3)
        if self.split != "train":
            tr_eps = getattr(self.task, "noise", {}).get("train", 0.0)
            if cand.score("train") < parent.score("train") - tr_eps:
                return False
        return cand.meta.get("novelty", 0) > 0


class HoldoutGate(PublicOnly):
    """Survival decided on the val split; train is only used for parent selection.
    Anti-overfitting gate (ShinkaEvolve-style public/private separation, adapted)."""

    split = "val"

    def _ensure_score(self, cand: Candidate) -> None:
        if "val" not in cand.scores and not cand.meta.get("error"):
            res = self.task.evaluate(cand.code, "val")
            cand.scores["val"] = res.score
