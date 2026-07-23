# SA acceptance and LNS parameter tuning

Changing the simulated-annealing acceptance (removing, tightening, "fixing") or tuning ruin sizes / restart cadence / temperatures never improved the score; removing SA outright was one of the worst regressions of the run.

## How it was tried
- c0014 (-0.90): replaced SA with strict best-acceptance descent + explicit 2-opt* — refuted; SA acceptance is what lets the LNS escape the 55690/27825 attractors.
- c0003 (no output): claimed to fix an SA acceptance "bug" (`cW <= cS` always true) + Or-opt-4 — silent failure, never evaluated.
- c0028 (-0.38): aggressive ruin (up to 18% removals), restart every 200 stalls, looser early SA — pushed X-n101 to the 27825 attractor.
- c0030 (-0.096): tightened SA acceptance + persistent best-tracking + Python 2-opt cleanup — score-neutral.
- c0033 (-4.96): "fixed" per-instance temperature scaling (near-zero `total` guard) + widened 2-opt + ruin spread — catastrophic; the temperature rescale broke acceptance across all three instances.
- c0029 (-0.098): 20% randomized recreate + larger perturbation range — neutral (also listed under diversification-perturbations.md).

## Why it failed
- The incumbent's SA-lite schedule (cooling over elapsed fraction) is co-tuned with ruin sizes and the 5s budget; any single-knob change either does nothing (within eval noise, see performance-analysis/score-noise-and-gate.md) or destabilizes acceptance badly.
- The "greedy until stall-reset" behavior c0003 called a bug is plausibly intentional; nobody demonstrated an actual defect.

## Verdict
refuted for removal/retuning; exhausted for knob-twiddling. Do not touch acceptance or ruin/restart/temperature parameters again without a mechanism-level argument plus repeated evaluations to beat noise.
