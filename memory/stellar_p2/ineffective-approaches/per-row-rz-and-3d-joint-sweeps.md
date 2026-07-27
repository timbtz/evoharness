# Per-Row R/Z Splits, Angular Twists, R0 Rescales, and Constant Stage Splits
Introducing fundamentally different contraction axes—such as per-row `cr/cz` gradients, poloidal-angular twists, uniform R0 rescales, or trying to rebalance b1/b2 splits at a fixed total depth—fails to beat the incumbent and either plateaus, regresses, or times out.

## How it was tried
- `stellar_p2-s105-26196944` c0038 (REJ, train 0.6263, val 0.6375): Gave each poloidal mode `m` its own `cr/cz` ratio interpolated as `cr*(1+alpha*(m-1)/mpol)`. Regressed.
- `stellar_p2-s105-26196944` c0043 (REJ, train 0.6219, val 0.6333): Varied the `cr/cz` R/Z asymmetry split ratio continuously on the second stage only, predicting it shifts high-mode R coefficients distinctly from Z. Regressed, showing that splitting `cr` across stages destroys the global aspect coordination.
- `stellar_p2-s105-26196944` c0033 (REJ, train 0.6256, val 0.6367): Varied `b1` and `b2` pairs at fixed total depth (`Σ(b1+b2) ≈ -8.0e-3 to -8.5e-3`).
- `stellar_p2-s105-26196944` c0039 (ERR, timeout 720s): Attempted a 3D joint grid varying `b1`, `c1`, and `b2` in a two-round design. Timed out.
- `stellar_p2-s105-26196944` c0040 (REJ, train 0.6258, val 0.6372): Applied a poloidal-angular twist `(θ→θ+δ·sin(nfp·φ))` as a truncated linear mode-mixing transform. Regressed.

## Why it failed
Writers predicted that decoupling the R/Z aspect/elongation tradeoff per-row, rebalancing the cross-terms at constant depth, or applying mode-mixing twists would uncover a Pareto-superior aspect/QI tradeoff. However:
1. **Per-row/stage-split R/Z gradients** destroy the global, coordinated aspect-relief mechanism that makes uniform `(cr,cz)=(0.5,0.7)` successful.
2. **Constant-depth b1/b2 splits** effectively linearize the distribution of the contraction, yielding no net Parelo-superior gain over the already optimal multiplicative composition.
3. **Angular twists** arbitrarily redistribute spectral power without targeting aspect ratio, degrading QI balance for no L gain.
4. **Two-round 3D joint designs** structurally betray the budget. The accepted winners in this window used a single batched `eval_many` of ~5 evals; the multi-round timeouts consumed 550–720s.

## Verdict
exhausted — Stop inventing structurally distinct local perturbation axes (twists, per-row gradients, stage splits, 3D sweeps) on the saturated B3 basin. The search space for local perturbations is fully mapped.
