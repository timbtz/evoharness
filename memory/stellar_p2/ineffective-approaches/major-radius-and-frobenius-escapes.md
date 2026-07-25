# Major-Radius Tapering, Frobenius Norms, and R-Shift Escapes
Specific structural escapes for the novelty ball that fail because they miscalculate geometry or directly smash the aspect-ratio constraint wall. Merged from c0014 and c0015 rejection mechanisms.

## How it was tried
- `stellar_p2-s100-89908732` c0014 (Rej, train 0.5725): Applied a radial taper by directly adding `TAPER_STRENGTH = 1.5e-3` to `r_cos[1][ntor]`. Collided instantly with the aspect-ratio wall.
- `stellar_p2-s100-89908732` c0015 (Rej, train 0.5698): Used `_bank_distance` (Frobenius norm) while the harness uses strict max-coefficient (L_inf). Passive drift never accumulated.
- `stellar_p2-s100-89908732` c0016 (Rej, train 0.5849): Merged margin-aware polish with an exact-novelty penalty acceptance key. However, `_bank_dist` silently returned `0.0` for every bank seed larger than the candidate. The penalty was a constant 0.05 with zero gradient.
- `stellar_p2-s100-89908732` c0020 (Rej, train 0.5742): Attempted multi-row shape rotation. Used right-padding which misaligned modes. Worse, it selected return paths via raw `fm.score`, returning a camper at `bank_dist 1.07e-4`.
- `stellar_p2-s101-20239089` c0002 (Rej, val -0.0125): "Orthogonal multi-pivot" escape — spread the displacement across 3+ non-neighbor coefficients so each moves ~40% less. Self-defeating: the guard metric is max-coefficient (L_inf), so smaller per-coefficient moves mean the boundary NEVER cleared 1e-3; the LF gate (also broken, raw `fm.score`) returned an in-ball camper.
- `stellar_p2-s103-71917443` c0003f (train 0.5711, val 0.5693): Attempted aspect-relieving *negative* major-radius shifts (`_r_shift`) on bank #4. The claim was that lowering R0 lowers the aspect ratio, providing headroom. **Refuted mechanism:** The raw bank seed #4 already sits on the mirror-ratio wall (margin ~0.00023); perturbing R0 directly breaks mirror symmetry before any aspect headroom is realized. The candidate regressed train by 0.0338 and was rejected.

## Why it failed
Bank seeds sit extremely close to the hard aspect-ratio limit AND the mirror-ratio limit. Any structural kick large enough to instantly clear the ball translates to a physical size change, cleanly violating geometry. Using standard matrix norms or incorrectly padded shape arithmetic dilutes the penalty of a single large coefficient shift, blinding the optimizer to its true harness score. Isolated R0 tapering/shifts fail because they directly violate the binding mirror/aspect constraints before creating useful novelty distance.

## Verdict
refuted — Do not use isolated radial tapering or major-radius shifts to escape. When matching the harness penalty, ALWAYS use strict max-coefficient (`np.max(np.abs(diff))`) distance on a center-padded canvas, never Frobenius/L2 norms or right-padded shapes.
