# Momentum Line Search
Extrapolating along accepted Gaussian step directions (a momentum line search) squeezes +0.001 to +0.003 train score out of the existing micro-polish budget for free, without breaking val feasibility.

## How it was tried
- `stellar_p2-s17-78763752` c0005 (ACC train 0.6213, val 0.6288): Added `_apply_delta` and `_deltas` helpers to track the vector between a boundary and its accepted child. After any accepted mutation, it arms a `momentum` tuple. On the next loop iteration, it evaluates a single 2x extrapolated boundary.
- `stellar_p2-s17-78763752` c0005f (ACC train 0.6200, val 0.6301): Modified c0005's sequential 2x probe to a batched `eval_many` pair probing 1.5x and 2.5x along the delta. If the far probe (2.5x) wins, it re-arms momentum with the 1.5x grown delta.
- `stellar_p2-s102-48117936` c0011 (train 0.5676, val 0.5785) & c0011f (train 0.5588, val 0.5694): Added the momentum line search to the incumbent escape stack. The mechanism successfully climbed L (p2 score reached 0.6052), but the accepted delta pointed straight back toward the bank seed. This dragged `best_b` deep inside the 1e-3 novelty ball (`bank_dist = 2.49e-4`). The 0.0376 novelty penalty perfectly cancelled the 0.027 L gain, resulting in a train regression. Adding a hard novelty floor on polish acceptance (c0011f) only accelerated the regression by rejecting all outward moves.

## Why it worked / failed
In a razor-thin feasible basin, random isotropic Gaussian steps that succeed usually indicate a favorable constraint-parallel slope. By simply saving the accepted delta (`_deltas(b, *_mats(cands[i]))`) and probing along that vector in the next iteration, the optimizer cheaply capitalizes on local slopes. The batched variant (c0005f) leverages the 2-worker parallelism: pairing the extrapolation probes in a single `eval_many` prevents idling a worker and gathers crude curvature data (near vs. far probe acceptance) simultaneously.
In B3 escapes, the steepest L-gradient points backwards into the bank seed's truncated modes. Momentum without an absolute novelty barrier naturally walks back into the 1e-3 ball (the `mode-recovery-homotopy` trap).

## Verdict
promising — Build on the momentum probe. The sequential version (c0005) scored slightly higher in train (0.6213 vs. 0.6200), but the batched version (c0005f) achieved better val (0.6301 vs. 0.6288). Both safely preserve the margin-aware key as the acceptance arbiter. On escaped boundaries, pair momentum strictly with an out-of-ball hard floor.
