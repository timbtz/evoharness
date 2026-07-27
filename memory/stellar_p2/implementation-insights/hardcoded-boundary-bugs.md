# Hardcoded Boundary Matrix Pitfalls (Symmetry, Shape, & VMEC initialization)
Hand-extracting or approximating Fourier boundaries from memory or text leads to guaranteed harness failures, either from Pydantic validation or VMEC physical initialization rules.
## How it was tried
- `stellar_p2-s105-26196944` c0003 (ERR): Handcoded an approximation of the B3 escape with mismatched shapes (5x9 list) and corrupted symmetry. Crashed with `Value error, r_cos for m=0 and n<0 must be 0.0`.
- `stellar_p2-s105-26196944` c0002, c0004, c0005f (ERRs): Hardcoded fabricated or heavily truncated nfp=3 matrices to try and reproduce the winning 0.6398 boundary cheaply. VMEC rejected them instantly with `FATAL ERROR... solver failed during the first iterations... poorly shaped or if it isn't spectrally condensed enough`.
- `stellar_p2-s105-26196944` c0043/c0044r1 (ERR if parsed blindly): Introduced silent typos in the 15-digit decimal hardcoded `r_cos` and `z_sin` arrays (e.g., `...3199682909` instead of `...3199862909`). While VMEC forgave these tiny float changes as valid boundaries, they structurally mutated the exact incumbent matrix. Hardcoding long floats invites silent copy-paste corruption.
## Why it failed
Stellarator-symmetric boundaries strictly require `r_cos` and `z_sin` to maintain structural zero patterns (e.g., `r_cos[0, n<0] = 0`). Furthermore, VMEC requires a highly "spectrally condensed" boundary. Manually altering, truncating, or typo-corrupting hardcoded matrices risks Pydantic rejection or VMEC divergence.
## Verdict
recurring pitfall — Never hand-type or truncate Fourier matrices to save space. If hardcoding is absolutely necessary, copy the FULL authentic decimal matrix verbatim. If adapting or approximating a seed, use `fm.seed_bank(i)` or `fm.seed_nae()` at runtime to guarantee a VMEC-safe baseline.
