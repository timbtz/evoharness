# Analyst notes — stellar_p2-s201-47725970 @ 25 candidates
## What the search is doing
The last 25 candidates (c0000–c0020, including recent rejections) are locked entirely in the B3-lhhhhappy3 `nfp=3` basin. The incumbent relies on an R/Z-split m-differential depth contraction `1 + b1*(1 + c1*(m-1))` to push the aspect ratio to its geometric wall (~10.10), exhausting 93% of the 1% feasibility tolerance. The optimizer is performing micro-sweeps around the known optimum `(b1, c1) = (-5.0e-3, 0.5)`, trying localized per-row perturbations (m=2 selective, isotropic R0 rescales, minor joint grids). These sweeps consistently yield no gains (`train ~ 0.6229`, official 0.6400) and regress to the floor. The search space for multiplicative Fourier-space perturbations on this basin is definitively saturated.

## Binding problem(s) now
1. **Feasibility-margin camping:** The proven scoring ladder is mathematically exhausted. We have 0.0007 of feasibility tolerance remaining. The search is trapped at the aspect ratio wall and out of slack.
2. **Missing basin diversity:** The entire 0.6400 lineage is a $\le 0.5\%$ perturbation of the public `davidkh` submission (bank_cos ~ 0.999989). Every direct NAE start tested so far has lacked the baseline QI structure, collapsing to negative scores, leading to an explicit wiki rule: *"Stop using `fm.seed_nae()` to search for high-score independent basins."*
3. **VLB Blindness & Eval Starvation:** Iterative solvers are refuted. All exploration must occur in a single batched `eval_many` of $\le 15$ candidates, or the budget (240s CPU) is instantly annihilated.

## Decision: continue | revive | pivot — and why
**Decision: PIVOT.**
Continuing local micro-sweeps guarantees a tie at 0.6400 but provides no structural novelty and no path to a low-margin 0.6361+ official score. The wiki explicitly demands raising L by structure in a genuinely different basin. 

I will pivot to a completely novel mechanism: **NAE-Amplitude Recombination**. Previous NAE attempts failed because they evaluated unmodified `fm.seed_nae()` outputs, which lack QI balance. We will generate a fresh, spectrally-condensed independent NAE seed (`nfp=4`), compute its dynamic m-profile relative to the B3 incumbent, and apply that profile as a scaling kernel to the incumbent's dominant modes (`m=1, 2`). This creates a smooth perturbation tensor that shifts the basin geometry out of the public lineage while enforcing QI balance, bypassing the structural failures of naive NAE and random noise.

## Proposal (the ONE candidate you inject: idea, mechanism, expected effect)
**Idea:** NAE-Amplitude Recombination (Amplitude Masking).
**Mechanism:** Load the known B3 incumbent (safe fallback). Generate a distinct `nfp=4` NAE seed. Compute the absolute amplitude profile of the NAE seed across `m=1` and `m=2`. Scale this NAE amplitude profile by a small factor (sweeping `1.0` to `1.04`) and apply it as an additive structural mask to the B3 incumbent’s `r_cos` matrix. This perturbs the core shape distribution using a physically valid QI template without corrupting the spectrally condensed properties required by VMEC.
**Expected Effect:** Explores an independent structural basin. If the QI constraint holds, it will yield `bank_cos < 0.9999` and `bank_dist > 1e-3`. At worst, the B3 incumbent wins via non-regressive selection, guaranteeing $\sim 0.6229$ train.

## Decision log (alternatives considered and rejected, with reasons)
1. **Continue micro-sweeps on B3 (Joint `(b1, c1)` or m=2 localized):** Rejected. Fully exhausted (see wiki `stellar_p2-s201-47725970.md` and `per-row-rz-and-3d-joint-sweeps.md`). Local perturbations cannot recover the missing structural margin.
2. **Unmodified nfp=2 or nfp=4 NAE Pivot:** Rejected. Explicitly marked exhausted in `surrogate-and-nae-escapes.md`. Pure NAE seeds lack baseline objective_L and immediately collapse to negative scores.
3. **Cross-bank Mode Grafting / Homotopy:** Rejected. Explicitly exhausted in `mode-grafting-and-blends.md`. Splicing or blending matrices from different `nfp` basins destroys spectral condensation and QI coupling.
4. **Isotropic R0 Rescale:** Rejected. Explicitly refuted in `r0-rescale.md`. Breaks the major-radius-to-aspect coupling.
