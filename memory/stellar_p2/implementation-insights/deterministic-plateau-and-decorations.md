# Deterministic Plateaus and Dead Decorations
Identical scores across consecutive candidates indicate dead code paths, purely order-preserving decorations, or triggers gated too strictly to ever fire.

## How it was tried
- c0001f..c0010f: All scored EXACTLY -0.43941 (train) / -0.43933 (val).
- stellar_p2-s11-38566380 c0013f (-0.4412) & c0019f (-0.4412): Attempted alternating-sign coordinate sweeps, but tied exactly because the sweep never wrapped to pass 2 within the tight budget.
- stellar_p2-s11-38566380 c0022f (-0.4440): Attempted to include an ellipsoid-perturbed seed and trigger NGOpt unconditionally at phase 4. Tied exact attractor point (-0.4395) because the seed scored worse and left the trajectory untouched.
- stellar_p2-s11-38566380 c0023f (-0.43941): Repaired the second inflation to operate on the best child instead of `pool[0]`. The forced base never displaced the deterministic lock of the coarse parent.
- stellar_p2-s11-38566380 c0024f (-0.4420): Swapped pool unpacking order but tying the deterministic attractor due to insufficient NGOpt slice budget.

## Why it failed
The 0.4394 shaped score reflects a binding constraint (mirror ratio 0.288 vs bound 0.2). Furthermore, the simulator is fully deterministic: `eval_many` on identical inputs returns identical scores.

Stable sorting of tied scores in `_add_candidate` keeps the earlier-inserted parent at `pool[0]`. If code attempts to mutate `pool[0]` instead of the evaluated children (e.g., `FRESH_BOOST` logic on freshly inflated modes), the deterministic simulator guarantees the trajectory stays perfectly inert. Even when correctly forced onto the best child, inflation adds fine modes that the sweep cannot leverage to break the binding constraint.

## Verdict
recurring pitfall — If you score exactly the same as the incumbent to 4+ decimals, verify the mechanism actually fires! Specifically: check if phase boundaries provide enough budget for sweeps to wrap, and ensure logic targets the explicitly evaluated child rather than `pool[0]`.
