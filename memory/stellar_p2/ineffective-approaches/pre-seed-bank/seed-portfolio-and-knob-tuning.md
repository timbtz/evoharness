# Seed-portfolio, knob and bias tuning (merged family)
> HISTORICAL — pre-seed-bank program (runs s7/s11: the retired 16-seed NAE portfolio + tuned-constant optimizer). Merged from seed-and-schedule-tuning.md, seed-portfolio-bloat.md, seed-portfolio-and-bias-rewrites.md.

Tweaking global mutation knobs, widening elite pools, bloating or re-parameterizing the NAE seed portfolio, or biasing pool ranking with hardcoded constraint heuristics — ~14 attempts across s7/s11, zero wins.

## How it was tried
Knob/pool tuning (s7): c0016 (-0.712) top-3 -> top-5 elite pool; c0025 (-0.595) top-4; c0026 (-0.571) sigma SHRINK 0.83 -> 0.75; c0021 (-0.751) magnitude-proportional mutations.
Portfolio bloat (s7): c0022 (-0.661) mixed `fm.seed_ellipse` + mp=2 seeds; c0023 (-0.855) SEED_BUDGET 16 -> 22; c0024 (-0.614) all seeds forced mp=1.
Portfolio re-parameterization (s11): c0013 (-0.648) mirror_ratio=0.18 swaps; c0015 (-0.482) interleaved mirror ratios; c0018 (-0.501) single-slot nfp=2 swap; c0023 (-0.580) 1.2-1.3x rotational_transform rescaling.
Bias heuristics (s11): c0003 (-0.627) violation roulettes; c0020 (-0.469) ALM-style penalty tiebreaks; c0018f (-0.462) secondary-violation penalties.
Unmetered injection (s11): c0017 (ACC, exact tie — inert) 8 pseudo-scored NAE seeds; c0022f (-0.4440) ellipsoid-perturbed seed.

## Why it failed
The incumbent's constants (K=3, SHRINK=0.83, SEED_BUDGET=16, spectrum scaling) were already tuned to the 72-eval budget; every knob change starved a downstream phase. Requested NAE parameters (mirror, iota) are NOT faithfully reproduced by mp=1 truncation, so spec swaps land in worse basins. Heuristic ranking biases misalign with the `-max_violation` descent signal. Unmetered injections were inert (pool truncation / pseudo-scores never beat real elites).

## EXCEPTION (the only portfolio changes that ever won)
c0025 (mirror-crushing seed family, -0.43925) and c0026f (byte-duplicate slot swapped for an extreme spec at zero RNG cost — worst case a guaranteed tie). Both documented in successful-patterns/pre-seed-bank/batch-population-and-coordinate-moves.md. The distinguishing trait: same budget, same RNG stream, targeted extreme geometry — not knob noise.

## Verdict
refuted — do not tweak global knobs, widen pools, bloat or re-parameterize seed portfolios, or bias ranking with physics heuristics. Zero-cost duplicate-slot swaps to targeted extremes were the sole exception.
