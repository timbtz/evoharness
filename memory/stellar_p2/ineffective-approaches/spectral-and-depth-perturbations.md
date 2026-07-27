# Spectral Bumps, Asymmetric Base Splits, and Deep LF Sweeps
Orthogonal micro-perturbations, multi-axis parameter sweeps, and heavily gated LF feasibility loops either fail to find a Pareto-superior aspect-relief point or collapse into timeout spirals. 
## How it was tried
- `stellar_p2-s105-26196944` c0013 (ACC, train 0.6239): Localized mid-m targeted perturbation bumps (Gaussian widths centered on m=2,3,4) attempting to decouple aspect-relief from the QI-critical m=1 mode. Plateaued below the incumbent.
- `stellar_p2-s105-26196944` c0015 (REJ, train 0.6222): Replaced spectral bumps with an asymmetric `base_R`/`base_Z` split sweep (ratio floating 1.0–1.6), attempting to find a better aspect-elongation operating point. Regressed.
- `stellar_p2-s105-26196944` c0015f (REJ, -inf): Attempted to execute a pure depth sweep (base -4.5e-3 to -6.5e-3) arbitrated by a strict LF feasibility loop (calling `fm.score()` sequentially with tight violation caps). Timed out at 720s.
## Why it failed
Mid-m bumps disrupt the delicate QI balance without unlocking new aspect margin. Decoupling the R/Z base ratio trades off aspect ratio for elongation poorly, yielding no net Pareto-superior gain. Furthermore, custom iterative loops that evaluate candidates one-by-one in Python to enforce strict LF feasibility gates annihilate the eval budget, causing immediate timeouts.
## Verdict
exhausted — Stop isolating perturbations to mid-m modes, splitting the base R/Z ratio, or writing custom LF-gated loops. Keep all evaluations in a single batched `eval_many` call.
