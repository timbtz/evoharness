# Blending and recombination of distinct optima
Naive linear blends — mean-crossover, worst-member interpolation, full-matrix convex mixes of pool members or bank seeds — are refuted across both program eras. Merged from worst-member-interpolation.md and new-ideas/early-recombination.md.

## How it was tried
- stellar_p2-s11-38566380 c0016 (`-inf`): mean-crossover of pool[0]/pool[1]; broadcast crash (2,3) vs (3,5) on mixed mp seeds.
- stellar_p2-s11-38566380 c0016f (-0.600): repaired via `_project_like` padding + randomized blend weight. Survived but regressed.
- stellar_p2-s11-38566380 c0026 (-0.5413): interpolated *worst* pool members toward the best elite to "sample geometrically distinct regions". Averaged their constraint penalties instead; left the NAE-stable subspace.
- stellar_p2-s17-78763752 c0001 (train 0.6188 EXACT TIE, val -0.9656): high-mode recombination of the top-2 bank seeds via `_project_like`. The exact train tie proved vlf never felt the perturbation; val collapsed.
- `stellar_p2-s100-89908732` c0008 (val -0.0119): dedicated 16-20 budget units to an initial batched low-order mode-blend phase between top bank seeds (anchored on the lower-resolution seed) before falling back to momentum polish. The blends died at val — the vlf-blindness landmine again.
- `stellar_p2-s100-89908732` c0018 (`-inf`): Tried `_convex_midpoint` between top-2 bank seeds at `t=0.45/0.55` to start halfway out of the 1e-3 ball. Crashed on bank shape mismatch (see numpy-shape-bugs.md).
- `stellar_p2-s100-89908732` c0018f (0.5750): Scrapped the blend and implemented center-padded exact distance. Still regressed slightly compared to passive drift alone.
- `stellar_p2-s101-20239089` c0007 (train 0.6161 EXACT TIE, val 0.6280): Added a gene-pool crossover entry point (`_row_blend` per-row blend of the top-2 bank seeds' escaped variants) in Phase 0. Survived but tied bit-exactly with the parent (c0006r1) — the crossover was a dead decoration that never won a triage slot, and displaced real escape evals under the Phase-0 budget cap.
- `stellar_p2-s103-71917443` c0002 (train 0.5697 EXACT TIE, val 0.5733): Added `_recombine` (alpha-blend of two bank seeds with mismatched canvas padding). The blend was selected as the base, but tying exactly means the deterministic simulation returned the same fallback anchor, and the cross-basin blending consumed budget without altering the escape outcome.

## Why it failed
Linearly blending distinct local optima does not yield a physically intermediate stellarator: it averages constraint penalties and introduces high-mode interference (aspect, QI, elongation) that stays invisible at loose train tolerance. Shape projection fixes the crash, not the physics. When used as a structural entry point (c0007, c0002), it is outcompeted by simple deterministic escapes and leaves the score untouched.

## Verdict
refuted for NAIVE blends. Structured recombination under the margin-aware acceptance key remains untested. Any attempt must clear the vlf-blindness landmine.
