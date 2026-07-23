# Multi-restart, multi-start, and multi-phase C LNS orchestration

Running the C LNS multiple times from different seeds, structurally diverse initial tours, or sequential time-sliced temperature phases to explore different basins was attempted ~25 times and is entirely score-neutral or negative.

## How it was tried
- Multi-seed restarts (same CW start, different LNS seeds): c0021, c0036, c0031 (cvrp-s3, all neutral at best).
- Multi-start diverse initial tours: c0035, c0060 (cvrp-s3, regressed).
- c0002 (cvrp-s11-66566581): budget-safe multi-restart wrapper (0.35s C bursts). Rejected.
- Parallel C LNS calls: c0039, c0028, c0052 (cvrp-s3, failed/crashed).
- c0006 (cvrp-s11-66566581): two overlapping parallel C LNS calls sharing best-so-far. Train -0.197, val -0.638. Rejected for heavy time starvation.
- c0012 (cvrp-s11-66566581): sequential multi-seed deep search (2 chains, 46% budget each). Train -0.174, val -0.550. Rejected for dividing budget.
- c0015 (cvrp-s11-66566581): 50/30/20 progressive multi-call C LNS split. Train -3.246, val -2.679. Catastrophic starvation.
- c0023 (cvrp-s11-66566581): new C entry point `two_phase_lns` to do multi-restart inside C without ctypes overhead. No output.
- c0001 (cvrp-s13-71671014): 3 sequential C calls with static temperature regimes (low f0=0.15, high f0=0.5, balanced f0=0.0) splitting the budget 40/35/25%. Accepted on train -0.1013 / val -0.1801 (noise-band tie). The static f0/f1 split disables the kernel's adaptive schedule.
- c0021 (cvrp-s13-71671014): replaced the c0001 3-phase split with a single full-budget C call but set a `0.30s` overlay reserve (`left = deadline - 0.30`). Train -0.956, val -0.914. Massive starvation caused by stealing exactly ~0.25s (5% budget) from the LNS. 
- c0019 (run cvrp-s17-78412700): sequential intensification. First C call runs near-completion, second call seeds from the best and restarts SA. Train -0.107, val -0.649.
- c0009 (run cvrp-s19-83885116): two sequential C calls (perturbed restart). Train -0.1248, val -0.3206. Rejected.
- c0001r1 (cvrp-s23-89572315): Budget-safe sequential C LNS restarts (3 calls: 60/20/20 split) gated on `n >= 150`. Train -0.169, val -0.303. Rejected: sliced budget starved SA cooling trajectory.
- c0003 (cvrp-s23-89572315): Interleaved 2-3 short C LNS calls (~1.5s each) with Or-opt polish between them. Train -0.139, val -0.449. Rejected for budget division.

## Why it failed
- On small train instances (n<=125), the SA schedule is fully sufficient to converge. Slicing the budget into multiple restarts/phases starves the SA schedule.
- Static multi-phase temperature tuning (c0001) defeats the kernel's built-in adaptive schedule. The C kernel scales its cooling perfectly with the passed time limit. Passing artificially short time slices forces the SA to cool drastically faster than intended.
- Attempting to avoid Python overhead by moving multi-restart to C (c0023) only exposes the candidate to the standard C-kernel silent failure modes.
- Even when explicitly gated on large `n` to avoid harming small instances (c0001r1, c0003), dividing the budget hurts large instances too by preventing the SA from reaching sufficient depth per call.

## Verdict
exhausted. Do not generate multi-restart, multi-start, or sequential multi-phase C LNS orchestration.

## REOPENED at large-n (2026-07-23)
- c0002 (run cvrp-s11-66566581): budget-safe multi-restart wrapper AFTER the first C LNS call improved val (X-n153+X-n176 mean) from -0.379 to -0.058 — far beyond the 0.15 noise band. Private (n=200-303) unchanged (~0.75), so the gain holds to ~n=176 and fades beyond.
- The refutation above was train-only evidence at n<=125. Small-n refutations do NOT automatically transfer: at larger n the C LNS converges slower per restart, changing the restart trade-off.
- Verdict amended: refuted for n<=125; promising for n>=150 — push it toward n>=200 (val now includes X-n214). NOTE: cvrp-s23-89572315 c0001r1/c0003 failed to capitalize on this, meaning simple budget-slicing does not unlock the large-n basin diversity.
