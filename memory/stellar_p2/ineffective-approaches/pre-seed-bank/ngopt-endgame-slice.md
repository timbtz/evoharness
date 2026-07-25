# NGOpt endgame slice
> HISTORICAL — pre-seed-bank program (run s11). Relocated from new-ideas/ after 5 refutations at increasing budgets; this was a refiner-driven idea family.

Spend the final exploit-phase budget on a nevergrad NGOpt slice over the elite's 8-10 coarse (m<=1, |n|<=1) coefficient deltas, batched 2-at-a-time via `fm.eval_many`. Hypothesis: coordinated multi-mode moves can trade constraints where single-coordinate probes cannot.

## How it was tried
- stellar_p2-s11-38566380 c0014f (ACC, tie): Added `_ng_polish` but gated it on `stall >= 8` AND `remaining > SAFETY + 5` inside Phase 4. Never fired (exact tie).
- stellar_p2-s11-38566380 c0015f (-0.4395): Repaired to fire unconditionally on Phase 4 entry. Tied again — ~7 evals too small to find a coordinated move.
- stellar_p2-s11-38566380 c0017f (-0.4395): Widened NG box delta (0.15 -> 0.25) and moved phase entry up (0.88 -> 0.84).
- stellar_p2-s11-38566380 c0021f (-0.4463), c0022f (-0.4440): `exploit_start` pulled to 0.70/0.75, ~16-20 metered evals, NG_DELTA 0.30.
- stellar_p2-s11-38566380 c0024f (-0.4420): `exploit_start` 0.72, NG_DELTA kept 0.15 to isolate the budget variable.

## Why it failed
With up to ~19 evals over ~8 coarse coefficient deltas the slice never beat the -0.4394 plateau; enlarging its budget only sacrificed late-phase coordinate sweeps (-0.44 to -0.446). The landscape around the elite is too flat, or the needed constraint trade is not reachable from these coarse modes.

## Verdict
refuted — tried at 7, 10, 16, and 19 evals; do not retry.
