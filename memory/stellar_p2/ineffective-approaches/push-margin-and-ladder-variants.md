# push run: Margin Projection and Warm-Ladder Variants on Isolated B3
Applying `fm.margin_step` orthogonal projections or sequential warm-ladders onto the proven B3 contraction grid fails to improve scores, strictly regressing or capping at the same train floor as the parent.

## How it was tried
- `push` c0001 (REJ, train 0.5748): Added a constant-margin projected m=2 perturbation orthogonal to the aspect gradient on top of the winner. Caused a severe regression (val -0.011).
- `push` c0003r1 (REJ, train 0.5718): Replaced the static grid with a sequential warm-ladder bisection into the convergence cliff using graded soft-fail scores. Eval starvation and broken fallback logic caused a severe regression.
- `push` c0005 (REJ, train 0.5803): Attempted to use `fm.aspect()` for free pre-screening and `fm.margin_step` as a post-processing placement on the winner. Regressed slightly.
- `push` c0006r1 (REJ, train 0.5860, val -0.0130): Added a warm-ladder step-climbing phase using graded soft_fails on top of the proven Phase 1 grid. Increased train but selected a pathological boundary that collapsed in validation.
- `push` c0007 (ACC, train 0.5830): Attempted to combine free aspect placement with a constant-margin projected contraction sweep. The projection failed, silently evaluating only the incumbent grid.
- `push` c0009 (REJ, train 0.5815): Replaced the contraction ladder with randomized Pareto exploration mixed with `fm.margin_step`. Regressed.
- `push` c0015 (REJ, train 0.5731): Replaced the contraction sweep with margin-aware placement and an orthogonal projection search. Regressed severely.

## Why it failed
Writers predicted that the new `fm.margin_step` and `fm.margin_grad` tools would allow "free" aspect margin placement, saving evals, and that orthogonal structural moves would raise L without paying the aspect ratio tolerance. The code applied these exact mechanisms. However, the landscape is strictly saturated. Any structural move orthogonal to the aspect gradient disrupts the delicate QI balance and selects a pathological boundary (negative validation score) or misses the true optimum entirely. The bisection and sequential warm-ladders are structurally isomorphic to previously refuted iterative SPSA/coordinate-descent mechanisms: they starve the eval budget and fail to guarantee non-regression.

## Verdict
refuted — Stop using `fm.margin_step`/`fm.margin_grad` to build orthogonal structural moves. The projection destroys QI balance and consistently selects pathological boundaries. Do not replace the single batched static grid with sequential bisection or warm-ladders; they burn the budget without yielding Pareto-superior points.
