# g2 window (c0030–c0049): Isolated State Traps, Syntax Errors, and Phase 4 Regression Bug
In run `g2` (isolated state capping ~0.584 train), attempts to replace the expensive, ineffective `solver-gradient` loop (Phase 4) or restructure the pipeline triggered a catastrophic -1.087 train regression bug or timed out. Successful repairs merely tied the 0.584 parent floor.

## How it was tried
- `g2` c0030 (ACC, train 0.5843): Added a novel-axis-generation phase evaluating 12 dynamic `nfp=3` NAE seeds. Safely tied the parent floor as NAE seeds lacked baseline L.
- `g2` c0031 (REJ, train 0.5644): Replaced pipeline with a massive nevergrad CMA-ES optimization over Fourier coefficients. Starved the budget, evaluating few feasible points.
- `g2` c0032 (REJ, train 0.5646): Added a batched bank-blend recombination portfolio + targeted gradient polish. Regressed due to pathological shape distortion from blending.
- `g2` c0033 (ACC, train 0.5843): Replaced previous phases with a `metric_grad` polish loop at 3e-5 trust region. Tied the floor, providing no structural gain.
- `g2` c0034 (REJ, train 0.5715): Added an n-profile-torsion transform (sinusoidal n-modulation) and dynamic warm-ladder search. Regressed by disrupting QI geometry.
- `g2` c0035 (REJ, train -1.0886): Attempted to tighten the grid by dropping redundant phases and removing fallback logic. Selected a pathological boundary, crashing validation.
- `g2` c0036 (ERR): Added a batched contract-and-blend recombination phase. Crashed with `SyntaxError`.
- `g2` c0037 (REJ, train 0.5783): Attempted a warm-ladder sequential deepening strategy. Regressed due to weaker search diversity compared to the batched grid.
- `g2` c0038, c0041 (REJ, train -1.0872): Attempted to replace the gradient loop with Pareto-aware composition sweeps. Selected a pathological boundary crashing validation.
- `g2` c0039, c0048 (ERR): Attempted to remove the gradient loop. Crashed with `SyntaxError: unmatched ')'`.
- `g2` c0040 (REJ, train -1.0885): Attempted a dedicated low-fidelity (LF) verification budget. Selected a pathological boundary.
- `g2` c0042, c0043, c0045, c0046, c0047, c0049 (REJ, train -1.0872): Attempts to harden None-guards, add constant-margin orthogonal walks, or leave the parent code exactly intact and add a trailing sweep. All catastrophically regressed to -1.087 train.
- `g2` c0044 (REJ, train -1.0872): Cleanly removed the gradient phase. Still regressed to -1.087 train.

## Why it failed
Writers predicted that iterative optimizers, bank-seed blends, or replacing the proven grid with warm-ladders/orthogonal projections would find missed Pareto improvements. The code applied these exact mechanisms.
However, dynamically generated NAE seeds and bank-seed blends fundamentally lack the baseline physics structure required by VMEC, capping them immediately. Furthermore, any structural modification to the parent pipeline (c0033) — specifically removing or altering Phase 4 (solver-gradient) — triggered a catastrophic validation crash (-1.087 train), proving the parent state contains a hidden dependency on Phase 4's cache state or variable assignments that strictly requires complete isolation from untested modifications. Orthogonal perturbations and n-axis torsion only disrupt QI geometry.

## Verdict
exhausted — Stop attempting to restructure the c0033 pipeline, replace the gradient loop, or blend bank seeds. The binding constraint is the isolated baseline matrix; no local trick can overcome it.
