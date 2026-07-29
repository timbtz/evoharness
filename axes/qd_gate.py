"""S2 Gate option: quality_diversity (MAP-Elites acceptance).

The default gates (public_only / holdout) accept a candidate only if it beats the
*global* best (or its parent) on one scalar. That is the measured novelty-collapse
mechanism: a 0.63 design that is structurally different from the 0.640 champion is
just "worse" and dies, so the search re-parameterizes one cell for 300 candidates.

Quality-diversity replaces the global rule with a *cell* rule: a candidate is kept
if it is the best in its BEHAVIORAL CELL, not if it beats the world. Cells are
defined by a structural descriptor (program complexity x code-novelty), so
genuinely different programs survive even when they score slightly lower. The
fitness maximized *within* a cell is still the gated-split objective, so quality
never collapses — it just stops being the only thing that matters.

This is the acceptance-criterion axis the 2026-07-27 audit + AIDE²/SeaEVO/LLaMEA
evidence point at. It is default-off and additive: existing configs (gate=holdout)
are byte-identical and the running stellar run is unaffected. Descriptor axes are
stdlib-only (ast + difflib) so the module can never fail to import.
"""
from __future__ import annotations

import ast
import difflib

from core.candidate import Candidate, Pool


def _complexity(code: str) -> int:
    """AST node count — a coarse Kolmogorov-style structural feature (LLaMEA)."""
    try:
        return sum(1 for _ in ast.walk(ast.parse(code)))
    except SyntaxError:
        return 0


def _complexity_bin(code: str) -> int:
    n = _complexity(code)
    if n <= 15:
        return 0
    if n <= 30:
        return 1
    if n <= 60:
        return 2
    return 3


def _novelty(cand: Candidate, refs: list[Candidate]) -> float:
    """1 - similarity to the NEAREST accepted program (0 = identical, 1 = unrelated)."""
    if not refs:
        return 1.0
    return max(1.0 - difflib.SequenceMatcher(None, cand.code, r.code).ratio() for r in refs)


def _novelty_bin(nov: float) -> int:
    return min(int(nov * 10), 9)  # 0..9


class QualityDiversity:
    """MAP-Elites gate: keep the best scorer per (complexity, novelty) cell.

    Behaves like HoldoutGate on the split (survival decided on val, the held-out
    anti-overfitting signal) but replaces "beat the parent/global best" with "be
    the best in your cell". Errors and -inf are rejected; everything else is
    admitted into its cell if the cell is empty or if it beats the cell champion.
    """

    split = "val"

    def __init__(self, task, key):
        self.task, self.key = task, key
        self.cells: dict[tuple[int, int], Candidate] = {}  # descriptor -> champion

    def _descriptor(self, cand: Candidate, pool: Pool) -> tuple[int, int]:
        refs = [c for c in pool.all if not c.meta.get("pruned")]
        return (_complexity_bin(cand.code), _novelty_bin(_novelty(cand, refs)))

    def _ensure_score(self, cand: Candidate) -> None:
        if self.split not in cand.scores and not cand.meta.get("error"):
            res = self.task.evaluate(cand.code, self.split)
            cand.scores[self.split] = res.score

    def accept(self, cand: Candidate, pool: Pool) -> bool:
        if cand.meta.get("error"):
            return False
        self._ensure_score(cand)
        if cand.score(self.split) == float("-inf"):
            return False
        desc = self._descriptor(cand, pool)
        champ = self.cells.get(desc)
        if champ is None or self.key(cand, pool, self.split) > self.key(champ, pool, self.split):
            self.cells[desc] = cand
            return True
        return False
