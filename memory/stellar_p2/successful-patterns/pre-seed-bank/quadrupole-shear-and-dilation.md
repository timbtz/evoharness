# Quadrupole Shear and Aspect-Repair Dilation
> HISTORICAL — pre-seed-bank program (runs s42/s7/s11). Retired 2026-07-24 for the bank-aware seed; see performance-analysis/seed-bank-regime.md. Amplitude constants refer to the retired trajectory; the zero-RNG geometric-transform CONCEPT may still transfer to bank-seed repair.

Deterministic, zero-RNG geometric transforms in Phase-1 (shear) and Phase-4 (dilation) break the deterministic attractor and sequentially repair constraints.

## How it was tried
- stellar_p2-s11-38566380 c0027 ACC -0.4264: Replaced the first slot of every Phase-1 batch with `_shear_quadrupole`. This deterministically rescaled all m>=1 modes by `1.0 + amp * (-1)^m`. The m=1 modes were shrunk (scale 0.88), breaking the mirror ratio constraint plateau.
- stellar_p2-s11-38566380 c0027f ACC -0.4253 (run best): Added a one-shot `_dilate` ladder at Phase-4 (exploit) entry. Evaluated two copies of the elite boundary with all m>=1 rows scaled by 1.03 and 1.08. Smoothly grew the minor radius, dropping the aspect ratio constraint violation (which had become binding after the Phase-1 shear shrank the geometry).

## Why it worked
Both are smooth similarity transforms of the cross-section (m=0 axis harmonics untouched), introducing no geometric kinks.
- The Phase-1 shear consumes zero RNG for its slot, leaving the RNG sequence for the second batch slot perfectly aligned with the baseline trajectory, minimizing disruption to the finely tuned 72-eval budget distribution.
- The Phase-4 dilation specifically targeted the newly-binding aspect ratio constraint without undoing the mirror ratio gains.
- CAUTION (bank era): blunt minor-radius dilation on bank seeds regressed hard (s17 c0004) — bank seeds already sit on the aspect bound. See ineffective-approaches/coordinate-and-phase-probes.md.

## Verdict
historical — sequential constraint repair via smooth zero-RNG transforms was the biggest pre-bank gain (-0.439 -> -0.425). On bank seeds, only margin-aware variants have a chance; naive dilation is refuted there.
