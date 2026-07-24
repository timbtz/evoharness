# Sweep construction and dual C LNS basins

Replacing or augmenting the standard savings construction with sweep construction, demand-weighted savings, or multi-basin seeding fails to improve the score and often destabilizes it.

## How it was tried
- c0003 (run cvrp-s37-18214945): Demand-weighted savings (Salhi-Nagy) and sweep-angle tie-breaks. Train -0.134, val -0.583.
- c0011 (run cvrp-s37-18214945): Primary savings (93%) + secondary sweep (4%) C LNS calls. Train -0.156, val -0.213.
- c0014 (run cvrp-s37-18214945): Parallel 15% slices from savings + sweep before a 65% intensification call. UnboundLocalError death.
- c0015 (run cvrp-s37-18214945): Dual-basin search with 72/22 split (savings + sweep) for moderate n. Train -0.308, val -0.230.

## Why it failed
- The C kernel's ruin-and-recreate SA is highly effective at escaping the structural bias of the initial savings construction. Alternative constructions (sweep, demand-weighted) do not survive the SA pass.
- Multi-basin orchestration splits the C LNS budget, causing time-starvation and train regressions.

## Verdict
exhausted. Do not propose alternative initial constructions (sweep, dual-basin, demand-weighted savings) or multi-basin seeding.
