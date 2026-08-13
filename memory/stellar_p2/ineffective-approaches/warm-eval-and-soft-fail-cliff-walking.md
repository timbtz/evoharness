# Sequential Warm-Ladders and Graded Soft-Fail Cliff Walking
Structuring the contraction search as a sequential warm-ladder of ≤1e-3 max-coeff micro-steps to exploit a ~25% wall-clock discount, or using graded soft-fail sentinels to bisect into the convergence cliff, strictly fails to beat the batched static grid.

## How it was tried
- `p3-on` c0003, c0005, c0016 (ACC, train 0.5830): Implemented sequential warm-ladders and graded soft-fail cliff bisection (via `_soft_key`) alongside the static grid. Safely tied the parent floor exactly.
- `p3-on` c0007 (ERR, score -inf): Replaced the Phase 4 bisection probe with a same-grid micro-ladder. Crashed with `SyntaxError: unmatched ')'`.
- `p3-on` c0010 (REJ, train 0.5839, val -0.0107): Replaced Phase 1 with a sequential warm-ladder of fine micro-steps. Tied parent train but yielded a pathological boundary that collapsed in validation.
- `p3-on` c0015 (REJ, train 0.5798): Replaced the static grid with a warm-ladder of micro-steps. Regressed slightly below the parent floor.
- `p3-on` c0019 (REJ, train 0.5827): Replaced the static grid with a depth-continuation warm-ladder and bisect. Regressed slightly below the floor and selected a pathological boundary.
- `g2` c0003 (REJ, train 0.5820): Attempted mid-point bisection over `(base, s2b)` depth parameters. Regressed slightly.
- `g2` c0006 (ERR, score -inf): Attempted an "honest cliff probe" pushing past feasibility walls to a soft-fail regime, then using `fm.margin_step` to calibrate. Crashed due to a syntax error in list extension.
- `g2` c0012 (ACC, train 0.5830): Deep-then-bisect warm ladder evaluating grid points at `b1=-7.0..-8.5e-3` and interpolating between the deepest converged point and the first soft-fail. Tied parent floor.

## Why it failed
Writers predicted that a sequential warm-ladder of ≤1e-3 micro-steps would trigger a ~25% solver warm-path discount, allowing 2-4 extra evals within the budget. The code constructed exactly these chains. However, the static contraction grid evaluates all points in a single batched `eval_many` call, which is already the most budget-efficient mechanism in Python. Micro-ladders consume Python loop overhead without unlocking enough wall-clock savings to yield a net gain in evaluation capacity. Furthermore, graded soft-fail sentinels routinely misidentify pathological boundaries as "near-converged", leading to negative validation scores. Interpolating linearly between safe and unsafe Fourier matrices to "bisect" the cliff strictly preserves infeasible deformations.

## Verdict
exhausted — Stop replacing batched static grids with sequential warm-ladders. Stop attempting to bisect into the convergence cliff using graded soft-fail sentinels.
