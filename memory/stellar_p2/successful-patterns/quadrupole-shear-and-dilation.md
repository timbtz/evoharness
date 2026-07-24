# Quadrupole Shear and Aspect-Repair Dilation
Deterministic, zero-RNG geometric transforms in Phase-1 (shear) and Phase-4 (dilation) break the deterministic attractor and sequentially repair constraints.

## How it was tried
- stellar_p2-s11-38566380 c0027 ACC -0.4264: Replaced the first slot of every Phase-1 batch with `_shear_quadrupole`. This deterministically rescaled all m>=1 modes by `1.0 + amp * (-1)^m`. The m=1 modes were shrunk (scale 0.88), breaking the mirror ratio constraint plateau.
- stellar_p2-s11-38566380 c0027f ACC -0.4253: Added a one-shot `_dilate` ladder at Phase-4 (exploit) entry. Evaluated two copies of the elite boundary with all m>=1 rows scaled by 1.03 and 1.08. Smoothly grew the minor radius, dropping the aspect ratio constraint violation (which had become binding after the Phase-1 shear shrank the geometry).

## Why it worked
Both are smooth similarity transforms of the cross-section (m=0 axis harmonics untouched), introducing no geometric kinks. 
- The Phase-1 shear consumes zero RNG for its slot, leaving the RNG sequence for the second batch slot perfectly aligned with the baseline trajectory, minimizing disruption to the finely tuned 72-eval budget distribution.
- The Phase-4 dilation specifically targeted the newly-binding aspect ratio constraint without undoing the mirror ratio gains.

## Verdict
promising — Build on this! The current best uses a fixed `SHEAR_AMP = 0.12` and `DILATE_FACTORS = (1.03, 1.08)`. The next levers are: tuning these amplitudes, adapting shear/dilation dynamically based on constraint metrics, or finding similar zero-RNG geometric transforms that repair iota deficits.
