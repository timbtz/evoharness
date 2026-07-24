# Multi-restart, multi-start, multi-phase C LNS, and Python perturbation loops

Running the C LNS multiple times, slicing its budget into phases, or wrapping it in a Python perturbation loop (remove/reinsert/reoptimize) was attempted ~30 times and is entirely score-neutral or negative.

## How it was tried
- Multi-seed restarts (same CW start, different LNS seeds): c0021, c0036, c0031 (cvrp-s3, all neutral at best).
- Multi-start diverse initial tours: c0035, c0060 (cvrp-s3, regressed).
- c0006 (cvrp-s11-66566581): two overlapping parallel C LNS calls. Train -0.197, val -0.638.
- c0009 (run cvrp-s19-83885116): C LNS double-bridge restart wrapper. Train -0.1248, val -0.3205.
- c0007 (run cvrp-s29-99479842): Added early-restart inside C kernel. Train -0.098, val -0.291.
- c0005 (run cvrp-s31-5170619): Bounded Python perturbation loop (eject 5-10% via Shaw, greedy reinsert, re-run C LNS). Train -0.127, val -0.288.
- c0008 (run cvrp-s31-5170619): Removed second C call, replaced with Python perturbation loop + cheap Or-opt(1,2) overlay. Train -0.116, val -0.315.
- c0011 (run cvrp-s31-5170619): Implemented C-side `perturb_and_run` entry point for kick + reoptimize. Train -0.133, val -0.243.
- c0025 (run cvrp-s31-5170619): Removed second C call, replaced with demand-descending savings construction variant + numpy Or-opt overlay. Train -0.119, val -0.273. Rejected.
- c0029 (run cvrp-s31-5170619): Forced dual C LNS basins from independent noisy constructions (60%/35% split). Train -0.285, val -0.449. Massive regression.
- c0039 (run cvrp-s31-5170619): 3 diverse savings starts sliced into parallel budget splits. Train -0.273, val -0.378.

## Why it failed
- On small train instances (n<=125), the SA schedule is fully sufficient to converge. Slicing the budget into multiple restarts/phases starves the SA schedule.
- Static multi-phase temperature tuning defeats the kernel's built-in adaptive schedule.
- Python-side perturbation loops either starve the main C SA budget without finding alternate basins, or just duplicate moves the C kernel already explores natively.
- Alternative initial constructions (descending savings, noisy savings) just yield worse basins that the SA cannot recover from in the restricted remaining budget.

## Verdict
exhausted. Do not generate multi-restart, multi-start, sequential multi-phase C LNS orchestration, or Python perturbation loops. The parent's `cost threshold` gating for a second C call is actively harmful and should be removed.

## REOPENED at large-n (2026-07-23)
- c0002 (run cvrp-s11-66566581): budget-safe multi-restart wrapper AFTER the first C LNS call improved val (X-n153+X-n176 mean) from -0.379 to -0.058.
- Small-n refutations do NOT automatically transfer: at larger n the C LNS converges slower per restart, changing the restart trade-off.
- Verdict amended: refuted for n<=125; promising for n>=150. However, cvrp-s29-99479842 c0012/c0017 show that warm-seeding `best` into a second call still starves the total budget and fails to find new basins.
