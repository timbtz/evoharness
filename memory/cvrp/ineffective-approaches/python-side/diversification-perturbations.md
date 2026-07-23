# Diversification: perturbations, randomized restarts, randomized scan orders

Injecting diversity (double-bridge kicks, randomized recreate, randomized savings restarts, shuffled scan order) was uniformly score-neutral — the LNS's SA already provides the diversification these changes duplicate.

## How it was tried
- c0026 (-0.096): double-bridge (4-edge) perturbation inside the C loop after local search + noise-randomized savings restarts in Python — neutral.
- c0029 (-0.098): 20% random (vs pure regret) insertion in recreate + perturbation size 60->80 — neutral (also cited in sa-acceptance-and-parameter-tuning.md).
- c0053 (-0.0906): randomized best-improvement scan order in the Python overlay — exact tie with parent costs; changed nothing.
- c0013 (cvrp-s11-66566581): mini-LNS in Python overlay using double-bridge perturbation + re-polish loop. Train -0.117, val -0.416. Rejected.
- c0009 (run cvrp-s19-83885116): C LNS double-bridge restart wrapper. Train -0.1248, val -0.3205. Rejected.
- Related: c0028's restart-every-200-stalls (see sa-acceptance-and-parameter-tuning.md) actively hurt (-0.38).

## Why it failed
- The kernel's Shaw/random ruin + SA acceptance already explores multiple basins; extra kicks either get absorbed (neutral) or displace budget from convergence (harmful).
- The train instances converge to fixed attractors (see performance-analysis/run-metrics.md) well within the budget, so added randomness cannot surface as improvement there.

## Verdict
exhausted. Diversity is not the bottleneck on train instances; do not add perturbation/randomization mechanisms.
