# Hardcoded Incorrect (Inferior) B3 Escape Boundary
Hardcoding an incorrect or inferior variant of the B3 escape boundary—resulting in lower objective_L—and pushing deeper depth-contraction fails to recover the proven winning score, as the base boundary quality strictly caps the output.
## How it was tried
- `stellar_p2-s105-26196944` c0010f (REJ, train 0.6221, val 0.6332): Hardcoded an 8×15 nfp=3 mpol=7 variant of the B3 escape matrix (with major radius 1.000001...) and applied an R/Z-split m-differential contraction sweep over `base ∈ [-1e-3, -3e-3]` with off-diagonal `(cr,cz)=(0.5,0.7)`. Evaluated cleanly but failed to clear the `train > 0.6255` bar, capping at 0.6221. 
## Why it failed
The writer predicted that a deeper contraction on this hardcoded feasible nfp=3 escape would raise objective_L past 12.6 and yield a train score > 0.6255. The code evaluated cleanly, indicating the boundary was VMEC-safe and structurally valid. However, the writer transcribed the wrong version of the escape boundary: at its uncontracted baseline, it yields a val of only ~0.627, whereas the proven winning B3 boundary (`s105-72881323` c0028, bank_dist 2.6e-3) yields val ~0.6368. No amount of depth-contraction can recover the missing L if the hardcoded base boundary is structurally inferior or starts at a lower objective_L.
## Verdict
exhausted — Ensure the hardcoded boundary exactly matches the winning ledger (`s105-72881323` c0028/c0075f). A slightly wrong or inferior base boundary will silently cap the score below the submittable threshold.
