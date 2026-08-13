# Solver Gradient Ascent (`fm.metric_grad` / `fm.grad_step`)
The first solver-gradient campaign did not establish that solver gradients fail. It tested a proxy step rule in a weak basin, while the LLM campaign did not call the solver-gradient tools at all.

## How it was tried
- `g2` c0001 (REJ, train -0.0128, val -0.0113): Attempted `k=20` (41 evals) + adaptive cap halving on the grid winner. Selected a pathological boundary that collapsed in validation.
- `g2` c0002 (ACC, train 0.5830): Attempted `k=6` (13 evals) + 4 cap halving steps. Safely tied the parent floor but provided no structural gain.
- `g2` c0004 (REJ, train 0.5818): Trimmed grid to 10 evals to fund `k=3` (7 evals) multi-round ascent. Regressed due to shrinking the proven contraction grid.
- `g2` c0007 (REJ, train 0.5807): Attempted `k=8` (17 evals) + 3 cap steps. Regressed slightly, consuming evals for no gain.

## Why it failed
Writers predicted that `metric_grad` would unlock structural L improvements orthogonal to the contraction grid. The code called `metric_grad` and evaluated `grad_step` outputs exactly as documented. However, the projection constraints and the 3e-5 trust region cap restrict `grad_step` to sub-decimal micro-perturbations. This mechanism cannot generate the coordinated R/Z-split m-differential scaling required to safely push the aspect ratio wall. It simply returns the incumbent or slightly disrupts QI geometry, yielding negative validation scores.

## Scope correction (2026-08-12)
The standalone loop started from an official 0.6321 boundary, not the actual
0.6408 champion, and optimized projected raw L while accepting on composite
honest score. Its rejected steps often raised L while losing honest score. The
LLM gradient arms called `margin_step`, but produced zero `metric_grad` or
`grad_step` calls. This evidence is basin- and implementation-scoped, not a
global exhaustion result.

## Verdict
open — Retest composite honest-score ascent, with novelty continuation and
frequent re-linearization, on the actual champion and on structurally distinct
starts. Do not infer basin-generation ability from a local polisher.
