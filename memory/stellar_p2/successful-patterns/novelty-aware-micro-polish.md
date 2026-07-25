# Novelty-Aware Micro-Polish and LF Gates
Polishing bank seeds with an acceptance/return key that explicitly subtracts the harness novelty penalty allows the standard micro-polish to passively drift out of the 1e-3 novelty ball.

## How it was tried
- `stellar_p2-s100-89908732` c0011r1 & c0011f (ACC, val 0.5884): Subtracted the exact harness penalty (`0.05 * (1 - bank_dist/1e-3)`) in `key_of()` and `record()`, steering the greedy polish outward.
- `stellar_p2-s100-89908732` c0020f (ACC, val 0.5918): Added an amplified novelty gradient in the acceptance key (ramp to zero at `NOV_D = 1.25e-3`, `NOV_LAM = 0.12`) plus a sigma floor (`SIG_ESC = 0.001`) to maintain escape-scale steps while inside the ball.
- `stellar_p2-s101-20239089` c0001, c0001f: Inherited the `NOV_LAM = 0.12` acceptance key, proving it effectively scales to structural escape candidates and safely pairs with the LF-verified return gate.
- `stellar_p2-s102-48117936` c0013f: Introduced a hard `out_b` tracking (highest novelty-key boundary that is feasible AND `bank_dist >= NOV_MIN`) with the return gate preferring it. The mechanism is sound but NEVER FIRED: no feasible out-of-ball boundary ever existed to track, and the return stayed the in-ball camper (bd 4.63e-4, bit-identical to c0003f). Gate-side novelty enforcement cannot conjure escape supply.

## Why it worked
Micro-polish moves naturally perturb the boundary by ~1e-4. By placing the exact linear novelty penalty in the acceptance key, each tiny step yields +0.005 of score, dominating standard L-gradients. This allows the optimizer to escape the ball using many small, constraint-preserving steps, avoiding the aspect-ratio wall that dooms macro-jumps. 

## Verdict
promising BUT rate-limited — the pattern is val-safe and recovered part of the penalty, yet over the whole B1 branch bank_dist only grew 2.05e-4 -> 3.65e-4. Passive drift alone needs ~4 more branch-lengths to exit. Keep this key as the baseline machinery, but pair it with structural escapes (see `successful-patterns/structural-ball-escape.md`).
