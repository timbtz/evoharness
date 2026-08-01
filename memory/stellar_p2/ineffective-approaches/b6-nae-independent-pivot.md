# B6-nae-independent Pivot (Parameterized NAE Seed Sweep)
Sweeping dynamically generated nfp=3 `fm.seed_nae()` parameters (aspect, elongation, rotational transform, mirror ratio) with the proven R/Z contraction fails to discover a genuinely independent feasible basin and regresses to the incumbent floor.

## How it was tried
- `stellar_p2-s203-38950787` c0005r1 (REJ, train 0.6128): Evaluated several nfp=3 NAE seeds and their proven two-stage contractions alongside the hardcoded incumbent floor. All NAE candidates collapsed, returning the unmodified incumbent floor.

## Why it failed
The writer predicted that parameterized `fm.seed_nae(n_field_periods=3)` could discover an independent, structurally distinct basin without dropping feasibility. The code evaluated these seeds in a batched `eval_many`. However, dynamically generated NAE seeds fundamentally lack the baseline `objective_L` of the public B3-lhhhhappy3 escape. The `seed_nae` function is meant for physical initialization templates, not L-maximized competitive boundaries.

## Verdict
exhausted — Stop using `fm.seed_nae()` to search for high-score independent basins. They lack the baseline physics structure required to be competitive in this benchmark.
