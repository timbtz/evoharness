# NGOpt endgame slice
Spend the final exploit-phase budget on a nevergrad NGOpt slice over the elite's 8-10 coarse (m<=1, |n|<=1) coefficient deltas, batched 2-at-a-time via `fm.eval_many`. Coordinated multi-mode moves can trade constraints where single-coordinate probes cannot.

## How it was tried
- stellar_p2-s11-38566380 c0014f (ACC, tie): Added `_ng_polish` but gated it on `stall >= 8` AND `remaining > SAFETY + 5` inside Phase 4. Never fired (exact tie).
- stellar_p2-s11-38566380 c0015f (-0.4395): Repaired to fire unconditionally on Phase 4 entry. Tied again, suggesting ~7 evals is too small a budget to find a coordinated move.
- stellar_p2-s11-38566380 c0017f (-0.4395): Widened NG box delta (0.15 -> 0.25) and moved phase entry up (`0.88 -> 0.84`) for more evals.
- stellar_p2-s11-38566380 c0021f (-0.4463), c0022f (-0.4440): `exploit_start` pulled to 0.70/0.75 of budget, giving ~16-20 metered evals. NG_DELTA widened to 0.30.
- stellar_p2-s11-38566380 c0024f (-0.4420): `exploit_start` pulled to 0.72, NG_DELTA kept at 0.15 to isolate the budget increase variable.

## Why it failed / stalled
With ~19 evals over the ~8 coarse coefficient deltas, the NGOpt slice still failed to improve upon the -0.4394 plateau, increasing the budget only resulted in slight regressions (-0.44 to -0.446) due to sacrificed late-phase coordinate sweeps. The search landscape around the elite appears too flat or the coordinated constraint trade (mirror ratio vs slack) is not achievable from these specific coarse modes. 

## Verdict
refuted — The NGOpt endgame slice has been tried at various budgets (7, 10, 16, and 19 evals) and failed to improve the plateau. Do not retry.
