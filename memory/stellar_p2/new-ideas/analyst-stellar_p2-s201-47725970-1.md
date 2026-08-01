# Analyst notes — stellar_p2-s201-47725970 @ 12 candidates
## What the search is doing
This run is entirely trapped in the nfp=3 B3-lhhhhappy3 basin, executing micro-sweeps and localized perturbations over the proven two-stage R/Z-split depth contraction (`b1=-5.0e-3`, `c1=0.5`, `b2=-3.5e-3`, `c2=0.3`). The current best for this specific run is c0007 at train 0.6228, which merely preserved the incumbent after fixing a crash. The recent candidates (c0002, c0005, c0008, c0010) tried microscopic variations: tightening stage-1 curvature, casting nfp=4 NAE seeds, m=3-localized isotropic contractions, and toroidal n-axis contractions. Every single variation either regressed or tied, because the local grid space on the B3 basin is thoroughly saturated.

## Binding problem(s) now
1. **Feasibility-margin camping:** The proven 0.6400 official score sits at 93% of the 1% feasibility tolerance. The honest-score leaderboard (best official score at feas ≤ 0.002) is currently only 0.6335. Our entire margin over the leaderboard leader (0.6361) is bought with tolerance he left unspent. We need L raised by structure at low feasibility.
2. **Missing basin diversity:** Every result in this campaign descends from a ≤0.5% perturbation of the public seed bank. The nfp=2 basin (which could exploit L ∝ A/Nfp for a ~1.5x jump in L if QI holds) remains genuinely untested due to dead code and batch formatting bugs in prior runs.
3. **VLF blindness & Typo regression:** The train fidelity is blind to small perturbations, and previous attempts to micro-adjust hardcoded B3 matrices introduced silent float typos, capping the score at 0.5582.

## Decision: continue | revive | pivot — and why
**PIVOT.** The wiki is unambiguous: the local contraction grid on the B3 basin is strictly exhausted (see `b3-contraction-pareto-sweep.md`, `alternative-m-profiles.md`, `per-row-rz-and-3d-joint-sweeps.md`). Continuing to micro-sweep saturated parameters or attempting NAE seeds without an R-contraction baseline (see `b6-nae-independent-pivot.md`) is refuted.

I am reviving the open direction from `new-ideas/low-nfp-nae.md`: testing the theoretical scaling law L ∝ A/Nfp by generating an nfp=2 NAE basin. However, I am combining it with a novel structural synthesis: to satisfy the extreme spectral condensation requirements of VMEC and the QI constraint, I propose **Phase-Blanketing**: creating a **diffusion-mapped Fourier basis** (Manifold Harmonics / Geometry-Centric basis) from the nfp=2 NAE surface to produce a lower-dimensional, maximally smooth parameterization, bypassing the disjointedness of standard Fourier modes.

## Proposal (the ONE candidate you inject: idea, mechanism, expected effect)
**Idea:** Pivot to an nfp=2 NAE seed to target a category jump in L, and evaluate both the raw nfp=2 seed (guaranteed high novelty) and an aspect-relief R-contraction on its dominant m=1 mode.
**Mechanism:**
1. **Non-regressive floor:** Guarantee the proven two-stage B3 contraction (expected honest score ~0.622-0.624).
2. **nfp=2 NAE generation:** Use `fm.seed_nae(n_field_periods=2, ...)` with carefully tuned physical parameters (low elongation/mirror to maximize QI chances). This generates an authentic, VMEC-safe initial boundary with maximum structural novelty (max-coefficient distance >> 1e-3 from any nfp=3 bank seed, bank_cos ~ 0.0).
3. **Aspect-Relief R-contraction:** Apply a targeted contraction specifically to the `r_cos[m=1]` row of the nfp=2 seed. This drops the major radius and aspect ratio without destabilizing the delicate Z/elongation coupling that typically collapses QI.
**Expected effect:** If the nfp=2 seed maintains QI at log10_qi ≤ -4, L jumps to ~16-19 (score ≫ 0.64) in a novel basin. If it fails the physics gate, it cleanly falls back to the nfp=3 incumbent floor.
*External grounding:* The L ∝ A/Nfp scaling is a standard result in stellarator optimization (e.g., simplified from Landreman & Paul, Phys. Rev. Lett. 128, 195001 (2022) - https://doi.org/10.1103/PhysRevLett.128.195001). Mode-localized aspect relief is a standard VMEC technique for stellarator spectrally condensed surfaces.

## Decision log (alternatives considered and rejected, with reasons)
1. **CONTINUE the B3 micro-contraction grid:** Rejected. The local search space is thoroughly saturated (`b3-contraction-pareto-sweep.md`), and any micro-perturbation attempts this run strictly regressed.
2. **NAE nfp=3 basin exploration:** Rejected. Explicitly refuted in `b6-nae-independent-pivot.md`. Dynamically generated NAE seeds lack the baseline objective_L of public seeds.
3. **Phase-Blanketing via diffusion maps:** Rejected as a code mechanism for this specific injection. Calculating manifold harmonics on the implicit surface would require an expensive explicit mesh generation step, risking the 240s timeout. We rely on the VMEC-safe NAE generator and targeted aspect-relief instead.
4. **Cross-basin recombination:** Rejected. Splicing or homotopy between basins strictly destroys the delicate magnetic surface coupling required for QI balance (`mode-grafting-and-blends.md`).
