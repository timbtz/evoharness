# Dilation Ladders, Aspect Macro-Search, and Relief Moves
Attempts to break deterministic plateaus by using structured isotropic minor-radius dilation ladders (scaling `rc[1:]` / `zs[1:]` by geometric factors) to explicitly probe the L-vs-aspect ratio trade-off curve.

## How it was tried
- `stellar_p2-s101-20239089` c0015 (ACC tie): Replaced invisible Gaussian polish with a radial-expansion search of 3-5 geometrically spaced dilation factors per batch. Tied exactly at 0.6161.
- `stellar_p2-s101-20239089` c0015f (ACC tie): Restored momentum polish but added a margin-banking acceptance branch and a single aspect-relief up-dilation probe (`DIL_UP = 0.0015`) to relieve the aspect wall. Tied exactly at 0.6161.
- `stellar_p2-s101-20239089` c0016 (ACC tie): Swapped binary feasibility gates for continuous weighted keys (`s * feas_weight(feas)`) to allow the dilation ladder to accept high-L candidates with small constraint violations. Tied exactly at 0.6161.
- `stellar_p2-s102-48117936` c0013 (ACC tie): Applied a `-1.6e-3` aspect-relief micro-contraction to the returned escape, intended to relieve the aspect wall. While LF-gated and tied exactly on train/val, the contraction scaled coefficients back toward the bank, dropping `bank_dist` inside the 1e-3 ball (homotopy mirage). 

## Why it failed
Bank seeds sit exactly on the aspect ratio constraint wall (`aspect_ratio ≈ 10.039`, limit 10.0). Any isotropic scaling (dilation or contraction) moves the aspect ratio directly in the wrong direction or neutralizes QI gains. Even with macro-visible amplitudes, the acceptance key cannot find a dilation child that clears the L-regression penalty without violating feasibility. Margin-banking fails because the `LAM=8.0` penalty strictly dominates the tiny margin headroom recovered. Negative contractions pull the boundary straight back into the novelty penalty zone.

## Verdict
refuted — Do not use radial dilation ladders or aspect-relief probes to break polish plateaus. The binding constraint is aspect ratio, and uniform scaling is the exact mathematical direction that violates it or re-enters the ball.
