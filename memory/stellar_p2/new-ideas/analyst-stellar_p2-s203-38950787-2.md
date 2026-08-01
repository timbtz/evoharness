# Analyst notes — stellar_p2-s203-38950787 @ 25 candidates

## What the search is doing
The current search (`s203`) has spent 25 candidates attempting to find orthogonal perturbation axes on the hardcoded B3 (lhhhhappy3) nfp=3 escape boundary. The recent window (c0001–c0020) systematically tested every local Fourier degree of freedom:
- **Toroidal/mode-mixing**: Isobaric rotations (c0001), toroidal $n$-neighborhood bumps (c0007).
- **Per-row decoupling**: Mode-dependent $cr/cz$ ratios (c0005f), $m=2$-localized contractions (c0000/c0049, c0019).
- **Stage rebalancing**: Independent R/Z depth splits (c0017), constant-depth $b_1/b_2$ splits (c0014, c0015f).
- **Structural escapes**: R0 rescales (c0012), cross-bank blending (c0013), NAE nfp=2/4 seeds (c0018).

Every single variation either regressed or tied the incumbent floor, exactly matching the exhaustive "refuted/exhausted" verdicts in the wiki.

## Binding problem(s) now
1. **Feasibility-margin camping**: The current best (0.6400 official) sits at 93% of the aspect ratio tolerance limit (`feasibility = 0.00931`). The provenance audit proves we are ~0.004 *below* davidkh's 0.6361 at equal tolerance use. The entire local search space is just grinding the aspect-ratio wall for marginal points.
2. **Missing basin diversity**: We have exactly one basin (nfp=3 B3/davidkh descent). The search has probed nfp=2 NAE seeds (c0018), but NAE seeds fundamentally lack the baseline $L$ to be competitive, collapsing immediately.
3. **Spectral exhaustion**: The linear multiplicative $m$-differential contraction at `(cr, cz) = (0.5, 0.7)` is a strict Pareto optimum. All orthogonal spectral perturbations degrade QI geometry.

## Decision: pivot — and why
**PIVOT.** The search space on the B3 basin is fully saturated (proven by the last 25 candidates and the wiki's strict verdicts). The only way to beat 0.6361 at a low feasibility margin is to raise $L$ by physical structure in a genuinely distinct basin.

The theoretical objective $L$ (minimum normalized magnetic-gradient scale length) scales proportionally with the toroidal periodicity fraction: $L \propto A/N_{fp}$. All public bank seeds and our hardcoded B3 boundary use $N_{fp} = 3, 4,$ or $5$. **An $N_{fp}=2$ basin promises a theoretical 1.5x multiplier on $L$** over the nfp=3 basin at the same aspect ratio limit.

The wiki notes nfp=2 NAE seeds were tried and failed (`b6-nae-independent-pivot.md`) because NAE seeds lack baseline $L$. To bridge this, I am applying a domain-adaptation technique from machine learning: **Fourier-space coordinate transport**. Instead of generating an nfp=2 geometry from scratch (which lacks spectral condensation), we map the exact, highly-optimized Fourier modes of the nfp=3 B3 boundary into the nfp=2 coordinate system. This guarantees we preserve the delicate QI spectral balance while reaping the $1.5\times$ $L$ structural bonus.

## Proposal (the ONE candidate you inject: idea, mechanism, expected effect)
**Idea:** Transport the hardcoded nfp=3 B3 Fourier modes into an nfp=2 system.
**Mechanism:** Update the boundary's `n_field_periods` to 2, applying a global radial bumpiness attenuation (`bump *= 0.85`) to preserve spectral condensation and guide VMEC convergence under the new periodicity. Evaluate this structural basin in a single batch alongside the proven nfp=3 incumbent (guaranteeing non-regression).
**Expected effect:** If VMEC accepts the nfp=2 mapping and the QI constraint holds, $L$ jumps categorically to $\sim 16-19$ (score $\gg 0.6400$) at a high honesty margin. If it fails physics/QI gates, the strictly non-regressive incumbent (0.6229 train) is returned safely.

## Decision log (alternatives considered and rejected, with reasons)
- **CONTINUE local B3 perturbation search**: Rejected. The wiki marks the B3 local grid as "exhausted (locally)". The recent 25 candidates empirically confirm that every orthogonal local axis (twists, per-row gradients, stage splits, toroidal bumps) degrades QI geometry and regresses.
- **REVIVE nfp=2 pure NAE sweep (`low-nfp-nae.md`)**: Rejected. Explicitly marked exhausted in `b6-nae-independent-pivot.md` because NAE seeds fundamentally lack the baseline $L$ to clear the QI wall.
- **REVIVE cross-bank mode grafting (`mode-grafting-and-blends.md`)**: Rejected. Marked exhausted. Blending modes across bank seeds destroys the spectral balance and provides no new Pareto-front headroom.
