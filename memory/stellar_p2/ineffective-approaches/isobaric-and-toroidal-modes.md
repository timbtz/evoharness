# Isobaric Mode Rotation and Toroidal n-Neighborhood Selective Contraction
Applying unitary rotations within a single poloidal mode row (`m=1`), or Gaussian spectral bumps targeting specific toroidal (`n`) neighborhoods on top of the two-stage contraction, fails to yield a Pareto-superior aspect/QI tradeoff and strictly regresses or ties the incumbent.

## How it was tried
- `stellar_p2-s203-38950787` c0001 (ACC, train 0.6128): Applied a sub-percent isobaric mode rotation (mixing two `n`-columns) within the dominant `m=1` row. The incumbent floor strictly won via the non-regression key, but the candidate scored ~0.010 train below the floor.
- `stellar_p2-s203-38950787` c0007 (ACC, train 0.6131): Added four toroidal-mode-selective contraction variants (differential `n`-axis scaling on the three largest-amplitude `n`-neighborhoods) alongside the incumbent. The incumbent won by non-regression; the variants fell short.

## Why it failed
Writers predicted that manipulating degrees of freedom orthogonal to the `m`-differential profile—either by redistributing power among `n`-modes at fixed `m` (isobaric rotation) or by targeting specific high-energy toroidal clusters (selective Gaussian bumps)—could discover a better aspect/elongation tradeoff. The code applied exactly these mechanisms. However, just like global `exp(-q|n|)` envelopes (see `toroidal-axis-contraction.md`), selectively perturbing the `n`-axis disrupts the delicate magnetic surface QI coupling without unlocking any aspect relief. The geometry strictly rewards the smooth `m`-indexed linear profile.

## Verdict
exhausted — Stop proposing orthogonal spectral tricks like fixed-`m` isobaric rotations or selective `n`-neighborhood bumps. The `m`-differential split at `(cr,cz)=(0.5,0.7)` remains the strict optimum.
