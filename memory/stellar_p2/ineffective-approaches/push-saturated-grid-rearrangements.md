# push run: Saturated Grid Rearrangements and Stage Compositions on Isolated B3
Rearranging the proven static two-stage contraction grid parameters or tacking on a third d-term distortion fails to break the B3 basin plateau, yielding zero train score gains on the isolated run.

## How it was tried
- `push` c0002 (ACC, train 0.5830): Restored the exact proven parent sweep and added an orthogonal-depth stage: a lightweight `cr`/`cz` asymmetric-axis sweep. Safely tied the incumbent.
- `push` c0004 (ACC, train 0.5830): Added a second global axisymmetric distortion d-term to the contraction formula. Safely tied the incumbent.
- `push` c0008 (ACC, train 0.5830): Replaced the static Phase 1 grid with a fully sequential warm-ladder that adapts the `base`/`curv` depth at every step by stepping along the Pareto frontier. Tied the incumbent.
- `push` c0010 (ACC, train 0.5830): Expanded the bank-seed fallback (Phase 4) from 2 to 5 candidates, added a Phase 3.5 mid-depth sweep, and enlarged Phase 3. Tied the incumbent.
- `push` c0011 (ACC, train 0.5830): Replaced the static grid with a sequential warm-ladder that adapts the contraction direction at each step by testing 3 perturbations of the current incumbent. Tied the incumbent.
- `push` c0013 (REJ, train 0.5818): Replaced the adaptive ladder with the proven static two-stage batched grid plus `fm.margin_step` calls. Regressed slightly.
- `push` c0014 (REJ, train 0.5804): Restored the static batched two-stage contraction grid with free aspect pre-screening. Regressed slightly.
- `push` c0016r1 (REJ, train 0.5806): Restored the single-batch joint stage-1 grid, adding a constant-margin orthogonal structural probe via `fm.margin_step`. Regressed slightly.
- `push` c0018 (REJ, train 0.5813): Replaced the fragmented ladder with the exact single-batch two-stage joint sweep. Regressed slightly.
- `push` c0019 (REJ, train 0.5645): Replaced the entire structure with a single streamlined batched joint grid. Regressed severely.
- `push` c0020 (REJ, train 0.5731): Replaced the structure with a streamlined two-batch contraction sweep. Regressed severely.

## Why it failed
The writers hypothesized that tweaking the parameters of the contraction formula, changing the evaluation batch sizes, or adding axisymmetric distortions would uncover a better aspect/QI tradeoff. The code applied these changes, but the accepted mechanisms merely guaranteed non-regression by falling back to the incumbent, while the rejected mechanisms introduced bugs or cut critical coverage, missing the proven optimum. The local search space for the R/Z-split m-differential contraction on the isolated B3 boundary is definitively saturated.

## Verdict
exhausted — Stop proposing micro-sweeps of stage parameters, `cr/cz` asymmetric grids, or three-stage structural distortions on this saturated basin. The exact two-stage static grid is the strict optimum.
