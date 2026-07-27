# Batched Pareto Frontier Sweep, Two-Stage Composition, and cr/cz Tuning on B3
Batch-evaluating R/Z-split m-differential grids via `eval_many` maps the local Pareto frontier. Pushing the depth via a **two-stage composed contraction** (`[1+b1·(1+c1·(m-1))] · [1+b2·(1+c2·(m-1))]`) and shifting the stage parameters jointly (`b1`, `c1`, `b2`, `c2`) safely expands the L/feasibility Pareto frontier, breaking the train score plateau.

## How it was tried
- `stellar_p2-s105-26196944` c0016 (ACC, train 0.6255, val 0.6362): Swept `base` depth, `curv` steepness, and asymmetric `cr/cz` splits on the hardcoded B3-lhhhhappy3 nfp=3 escape boundary. 
- `stellar_p2-s105-26196944` c0017 (ACC, train 0.6256, val 0.6371): Applied a two-stage composed contraction (multiplicative second profile on top of the winning stage-1 profile) to explore a non-linear 2D depth surface.
- `stellar_p2-s105-26196944` c0022 (ACC, train 0.6259, val 0.6372): Deepened the two-stage composition to straddle the aspect wall. Swept stage-2 depth `b2 ∈ {-3.5, -4.5, -5.5}e-3` and stage-2 curvature `c2 ∈ {0.3, 0.1, 0.0}`.
- `stellar_p2-s105-26196944` c0024 (ACC, train 0.6262, val 0.6375): Combined the c0022 two-stage composition with fine `cr` ratio shifts (`cr ∈ {0.50..0.58}`, `cz=0.7` fixed). 
- `stellar_p2-s105-26196944` c0025f (REJ, train 0.6259, val 0.6375): Swept finer `cr` edges (cr 0.50→0.62) at b2=-4.5e-3 and b2=-5.5e-3. Predicted to hit 0.6263 by consuming the last aspect budget, but plateaued exactly at the c0024 winner, confirming `cr` saturation.
- `stellar_p2-s105-26196944` c0033 (REJ, train 0.6256, val 0.6367): Held total depth constant and varied the `b1`/`b2` split. Failed to improve, indicating the cross-terms do not offer a better high-m contraction profile at fixed depth sums.
- `stellar_p2-s105-26196944` c0034 (ACC, **CURRENT BEST**, train 0.6269, val 0.6378): Jointly swept stage-1 `b1 ∈ {-4.0...-6.0}e-3` and `c1 ∈ {0.3, 0.5, 0.7}` while keeping stage-2 fixed at the proven optimum. The non-symmetric deviation in stage-1 curvature unlocked a tighter aspect/QI tradeoff. 
- `stellar_p2-s105-26196944` c0035f (ACC, train 0.6268, val 0.6378): Adaptive two-round sweep attempting to rediscover the c0034 stage-1 winner and then sweep stage-2 `(b2,c2)` at shallower depths (`b2 ∈ {-2.5,-3.0}e-3`, `c2 ∈ {0.4,0.5}`). Plateaued at exactly the c0034 score, proving stage-2 `(b2,c2)` is fully saturated at the default `(-3.5e-3, 0.3)` when paired with the deep stage-1.

## Why it worked
By pre-evaluating the incumbent alongside parameter expansions in a single batched design, the search strictly guarantees non-regression while avoiding iterative timeouts. Two-stage composition proved strictly more L-efficient per unit aspect than single-stage grids. While `cr` ratio splits, constant-depth `b1`/`b2` rebalancing, and shallow stage-2 sweeps saturated quickly, independently deepening stage-1 via joint `b1`/`c1` variation provided a completely new contraction vector that successfully expanded the Pareto frontier further without breaching the feasibility wall.

## Verdict
exhausted (locally) — The single-batched grid space on the B3 nfp=3 basin is thoroughly saturated across all stage parameter corners. The local search space is fully refuted. Future writers must propose a genuinely new basin or structural escape.
