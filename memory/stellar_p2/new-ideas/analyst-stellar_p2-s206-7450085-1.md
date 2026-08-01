# Analyst notes — stellar_p2-s206-7450085 @ 12 candidates
## What the search is doing
The search is trapped in an **isolated state**. Without access to the authentic `B3-lhhhhappy3` hardcoded escape boundary that drove the 0.6400 official winner, the run is forced to rely on dynamically retrieved generic `fm.seed_bank()` nfp=3 seeds. The current best (c0005f, train 0.5810) applies the proven two-stage R/Z-split depth-contraction pipeline to these generic seeds, but they fundamentally lack the baseline `objective_L` required to be competitive. The score gap between this isolated cap (~0.581) and the documented floor (~0.626) is ~0.045, a purely structural physics deficit in the anchor boundary.

## Binding problem(s) now
1. **Missing baseline L:** The binding constraint is the structural gap between generic bank seeds and the optimized public lineage. Every perturbation, blend, or contraction scheme on generic seeds caps at ~0.581.
2. **Saturation of contraction tricks:** The wiki explicitly rules out m=1-targeting (c0003), bank blends (c0005), nfp=2 NAE pivots (c0004, crashed), and all nonlinear m-profiles.
3. **Eval starvation:** High-mode nfp=3 seeds cost 12-27s/eval, starving the budget.

## Decision: pivot — and why
**PIVOT.** Since every local perturbation axis is exhausted and structural topology changes (twists, blends, grafts) have regressed, we need an injection that injects an entirely new degree of freedom: **simultaneous R/Z-split contraction combined with toroidal sub-harmonic injection (`n=nfp` mode bumping)**. This mechanism introduces a coordinate change orthogonal to the multiplicative scaling (which the wiki marks exhausted). By adding a targeted `n=nfp` sub-harmonic to the dominant `m=1` poloidal mode, we directly alter the macrohelical shaping of the plasma without uniform scaling. This guarantees high structural novelty (rapidly escaping the `bank_cos` penalty), acts directly on the aspect ratio geometry, and exploits the fact that the `m=1, n=nfp` harmonic is the primary driver of the global magnetic surface rotational transform.

## Proposal (the ONE candidate you inject: idea, mechanism, expected effect)
**Idea:** Inject a targeted sub-harmonic perturbation into the `m=1, n=nfp` Fourier coefficient (the dominant helical mode) of the top nfp=3 bank seed, layered with the proven R/Z-split m-differential contraction. 
- **Mechanism:** Extract `primary = fm.seed_bank(best_info)`. Apply an additive offset $\Delta$ to `r_cos[1][-1]` (the `n=nfp` column) and `z_sin[1][-1]`, followed by the standard multiplicative R/Z contraction `factor = 1 + base(1 + curv(m-1))` with `base ≈ -5e-3`, `curv = 0.5`, and off-diagonal `(cr, cz) = (0.5, 0.7)`.
- **Expected Effect:** The `n=nfp` injection will deform the helical axis shape, providing a structural lift to `objective_L` independent of the exhausted aspect-ratio tolerance camping. It will push the `bank_cos` meaningfully below 0.9999, unlocking the novelty penalty without destroying QI balance.

## Decision log (alternatives considered and rejected, with reasons)
- **CONTINUE (c0005f contraction grid):** Rejected. The wiki (`stellar_p2-s206-7450085.md`) marks dynamic bank seed sweeps as exhausted, capping at ~0.581 due to missing baseline `objective_L`. Expanding the grid simply burns budget against the structural wall.
- **REVIVE (Exact B3 boundary matrix):** Rejected. As an analyst, I cannot inject an exact 120-float matrix I do not have textually access to; hardcoding from memory guarantees VMEC initialization crashes (`hardcoded-boundary-bugs.md`).
- **PIVOT (nfp=2 NAE basin):** Rejected. `b6-nae-independent-pivot.md` and `stellar_p2-s205-85293087.md` definitively prove dynamically generated NAE seeds lack baseline `objective_L` and collapse QI geometry.
- **PIVOT (Spectral noise / Gaussian bumps):** Rejected. `gaussian-spectral-noise.md` proves undirected perturbations disrupt the delicate spectral condensation required for QI balance.
- **PIVOT (Cross-bank blending):** Rejected. `mode-grafting-and-blends.md` proves same-nfp recombination and convex homotopy blends structurally destroy magnetic surface coupling.
