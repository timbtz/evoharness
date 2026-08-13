# stellar_p2-p3-on: Isolated State Traps, Margin Steps, and Orthogonal Projections
Without the authentic B3-lhhhhappy3 escape boundary, this run is structurally capped at ~0.583 train. Attempts to break out using `fm.margin_step`, constant-margin orthogonal walks, graded soft-fail cliff walking, cross-bank blends, nfp=2 NAE probes, split-rate m-differentials, toroidal-axis contractions, ternary depth searches, and multi-bank portfolios all failed, regressing, timing out, yielding pathological boundaries, or hitting parse errors.

## How it was tried
- `p3-on` c0001 (REJ): Third-stage contraction + orthogonal `fm.margin_step` projections. Regressed.
- `p3-on` c0004 (REJ): Adaptive continuous coordinate-descent using free `fm.aspect`. Yielded pathological boundary.
- `p3-on` c0006 (REJ): Recombination pool of convex blends between B3 and bank seeds. Regressed.
- `p3-on` c0008/c0012/c0018/c0020 (ACC, train 0.5830): `fm.margin_step` post-processing, bank-seed nfp=2 probes, warm-ladders, and Pareto sweeps. Tied parent floor.
- `p3-on` c0009 (ERR), c0019 (REJ): Replaced static grid with adaptive walks and soft-fail bisection. Crashed or regressed to pathological states.
- `p3-on` c0011/c0014 (REJ): Constant-margin orthogonal walks and exact c0034 grid parameter rebalancing. Regressed.
- `p3-on` c0017 (REJ): Multi-seed contraction portfolio and per-toroidal-mode selective contraction. Regressed severely.
- `p3-on` c0021 (REJ, val -0.0106): Introduced split-rate m-differential `cr` profile (cr rises with m). Yielded pathological boundary.
- `p3-on` c0022 (ACC, train 0.5830): `fm.aspect()` pre-filter on the saturated grid. Tied parent floor.
- `p3-on` c0023 (REJ, val -0.0130): Toroidal-axis differential contraction. Yielded pathological boundary.
- `p3-on` c0024 (REJ, val -0.0104): Stripped NAE probes and focused tightly on 4-point b1 deepening and 3-point cr sweep. Regressed.
- `p3-on` c0025 (ACC, train 0.5830): Constant-margin orthogonal-contraction sweep projecting onto aspect-invariant subspace. Tied parent floor.
- `p3-on` c0026/c0029 (REJ): Ternary searches along the depth axis, targeted two-stage parameter combinations, and interpolation probes. Regressed.
- `p3-on` c0028 (REJ): Multi-anchor NAE/bank seed exploration. Regressed.
- `p3-on` c0030 (REJ): Exact margin placement at 0.002 via `fm.margin_grad`, then aspect-preserving subspace search. Regressed.
- `p3-on` c0031 (ACC, train 0.5697): Full redesign to multi-bank-seed/NAE exploration strategy. Regressed severely.
- `p3-on` c0032r1 (ACC, train 0.5645): Restored proven grid on dynamic nfp=3 bank seeds with `fm.aspect` pre-screening. Regressed due to overly aggressive aspect filter.
- `p3-on` c0033r2 (ERR): Restored two-stage grid. Crashed on parse error.
- `p3-on` c0034 (ACC, train 0.5792, val 0.5928): Proven grid on top nfp=3 bank seed, stripped failed redesigns. Safely recovered partial train floor.
- `p3-on` c0035r1 (ACC, train 0.5803, val 0.5910): Adaptive half-step refinement grid centered on Phase-3 winner's parameters. Marginal train gain, val regression.
- `p3-on` c0036 (REJ): Contraction-depth Pareto ladder using free `fm.margin_step` and warm-eval steps. Regressed.
- `p3-on` c0037 (ERR): Expanded two-stage composed contraction grid with winner-centered refinement. Crashed with `axisymmetric boundaries are not supported`.
- `p3-on` c0038 (REJ, train 0.5791): Adaptive coarse+fine refinement grid with warm-cache exploitation. Regressed.
- `p3-on` c0039 (ERR): Aspect-selected anchor with fine cr micro-sweep. Crashed with `SyntaxError`.
- `p3-on` c0040 (REJ, train 0.5645): Three-seed R/Z-split contraction portfolio on nfp=3 bank seeds. Regressed severely.

## Why it failed
Writers repeatedly predicted that orthogonal structural moves, uncontracted margins, new bank seeds, or finer grid resolutions would raise L. However, without the authentic hardcoded B3-lhhhhappy3 matrix, generic bank seeds lack the structural `objective_L` to reach the competitive frontier. Furthermore, removing R/Z components or aggressively filtering aspect ratios simply starves the batched grid or collapses the plasma geometry, yielding negative validation scores or `axisymmetric boundaries are not supported` runtime errors.

## Verdict
exhausted — Stop using `fm.margin_step`, orthogonal projections, ternary depth searches, split-rate differentials, or multi-bank portfolios to buy honest margin in isolated states. The binding constraint remains the exact hardcoded elite matrix.
