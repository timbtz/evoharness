# C LNS budget slicing, two-call splits, and multi-restart orchestration

Replacing the single 95%-budget C LNS "megacall" with multiple calls, slices, or restart loops consistently starves the C kernel's simulated annealing schedule, causing regressions across all train/val instances.

## How it was tried
- c0001 (run cvrp-s37-18214945): 85% budget main call + short 12% perturbation restart for n>=120. Train -0.258.
- c0002 (run cvrp-s37-18214945): Budget-filling loop alternating 0.8s C slices with Or-opt polish. Train -0.261.
- c0004 (run cvrp-s37-18214945): Two-call split (60% / 35% forward-feeding the best). Train -0.331.
- c0007 (run cvrp-s37-18214945): Competitive multi-restart loop using bounded 0.8-1.2s slices. Train -0.261.
- c0008 (run cvrp-s37-18214945): Two equal 47% budget slices. Train -0.340.
- c0010 (run cvrp-s37-18214945): Alternating C LNS (1.3s) and Or-opt polish for up to 4 rounds on n>150. Train -0.233.
- c0013 (run cvrp-s37-18214945): 60/30 multi-seed dual calls, dropping Or-opt overlay entirely. Train -0.268.

## Why it failed
- The C kernel's SA cooling schedule relies on a continuous time budget to escape local optima. Dividing the budget starves the search, dropping instances to the 27825 attractor (X-n101) and causing massive train regressions.
- The hypothesis that large-n instances (n>=150) do not converge in one 5s call and would benefit from diverse SA basins was refuted: sliced calls fail to find deeper basins because individual slices lack the cooling time.
- Forward-feeding `best` or running perturbation restarts only duplicates the trajectory or destroys incumbent density.

## Verdict
exhausted. Do not generate multi-restart, multi-start, sequential multi-phase C LNS orchestration, or C budget slicing loops.
