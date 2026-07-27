# Analyst notes — stellar_p2-s105-26196944 @ 24 candidates

The `INJECT/` write is gated by this session's permission mode — the same block the s105-72881323 @36/@48/@60/@72 analysts hit. The complete standalone module is staged in the Write call above and needs approval to land at `INJECT/1-deepen-twostage-to-aspect-wall.py`. Below is the decision record.

---

# Analyst notes — stellar_p2-s105-26196944 @ 24 candidates

## What the search is doing
This run (branch B5) restarts from the campaign winner and re-polishes the **B3-lhhhhappy3 nfp=3 escape** with the one mechanism that works: the R/Z-split m-differential contraction `factor = 1 + base·(1 + curv·(m−1))`, off-diagonal `(cr,cz)=(0.5,0.7)`. Trajectory (train / val / train-aspect / feasibility):
- c0006 bank sweep 0.600 / 0.607 — cheap fallback floor.
- c0013 mid-m saltation bumps 0.6239 / 0.6354.
- **c0016** incumbent + modest depth 0.6255 / **0.6362** / aspect 10.077 / feas 0.00774.
- **c0017** *two-stage composed* contraction 0.6256 / **0.6371** / aspect 10.089 / feas 0.00893 — **current best**.
- c0018 mode-multiplicative (rej), c0019 quadratic profile (rej), c0020 m=1 micro-rotation (rej, and it *broke QI*: log10_qi −3.973 > −4).

The writers found c0017 (composition beats single-stage), then immediately **pivoted away** to three unrelated perturbation axes and regressed. Nobody deepened c0017.

## Binding problem(s) now
1. **The binding constraint is ASPECT RATIO, not QI — and there is unspent budget.** Across every candidate, `feasibility == (aspect_ratio − 10)/10` to the digit (c0016: 10.077→0.00774; c0017: 10.089→0.00893). Elongation (4.37 vs limit 5) and QI (log10_qi −4.0048, and *improving* with depth: −4.0042→−4.0048 from c0016→c0017) both have margin. The wall is aspect ≤ 10.10 (feas 0.010). c0017 sits at **0.0089 — ~0.0011 of aspect budget unspent**, and the official winner ran at feas 0.0096, so this run has not yet gone as deep as the proven 0.6398 boundary.
2. **The two-stage composition is strictly more L-efficient per unit aspect.** c0017 reaches **val 0.6371 at feas 0.0089**, beating the official winner's val 0.6368 which needed feas 0.0096. The composed (multiplied-linear) profile puts contraction where L responds best — this is exactly why quadratic c0019 failed (uncoupled quad coeff) while composition c0017 won (coupled curvature).
3. **The search is throwing away its own best axis.** The HINT is explicit: "push the depth contraction DEEPER … official rises WITH objective_L (aspect→10.10 wall); feasibility budget is ~0.01, stop over-gating on viol≤0.001; val does NOT over-report." c0018–c0020 did the opposite.

## Decision: **continue** — and why
Continue the live best direction and do the one thing the last three writers didn't: **deepen c0017's two-stage composition to the aspect wall.** This is not a refuted axis — two-stage composition is the newest *accepted* improvement, is absent from the exhausted list, and the constraint analysis + HINT both say the last 0.001 of aspect budget converts directly to L. Every genuinely-new mechanism (Gaussian noise, n-axis/toroidal, m=1-selective, exp-anisotropic, grafting/blends, SPSA/coordinate-descent, single-stage power-law/base-shifts, truncation) is marked refuted/exhausted in the wiki; a pivot into any of them is negative-EV. Web search (ConStellaration paper, arXiv:2506.19583) confirms aspect is a first-class hard constraint in the "simple-to-build" P2 benchmark but offers no cheaper micro-mechanism than exploiting the constraint slack directly.

## Proposal (the ONE candidate I inject)
`INJECT/1-deepen-twostage-to-aspect-wall.py`.
- **Idea:** one batched `eval_many` **depth ladder** of two-stage compositions (stage-1 = proven −5e-3 winner; stage-2 `b2 ∈ {−3.5, −4.5, −5.5, −6.5, −7.5}e-3`, `c2=0.3`) that **straddles the aspect wall** (train aspect ≈10.088→10.108), plus 3 profile-shape probes at mid-depth (`c2 ∈ {0.15,0.3,0.5}`, split `(0.6,0.6)`). 9 candidates → fits one batch.
- **Mechanism:** `fm.score()>0 ⟺ feasible`, so selecting the **max-feasible novelty-key** candidate automatically returns the *deepest feasible* point — whichever constraint binds first (aspect, or QI if it turns first) auto-rejects the over-deep ones via a negative score. Novelty key subtracts the bank penalty explicitly (bank_dist ≈0.0026, so no penalty, but computed to be safe).
- **Safety:** LF-verify top-3 as a **landmine gate only** (drop LF-infeasible = crossed the wall at the tighter gate tolerance; never re-rank feasible survivors — lf-selection-bug.md); fallback = the exact single-stage −5e-3 winner (never the raw parent or diagonal — anchor-floor-rejection-bug.md). Budget-guarded to ≤9 train + 3 LF evals, well inside 240 s.
- **Expected effect:** a boundary at aspect ≈10.096–10.099 (feas ≈0.0096–0.0099), **val > 0.6371, official > 0.6398** — a new campaign best. Falsifiable: if L does *not* rise with aspect past 10.089, or QI turns binding before the wall, the ladder returns c0017/floor unchanged (train ≤ 0.6256) and the "deepen" thesis is refuted.

## Decision log (alternatives considered and rejected)
- **Deeper single-stage base shift** — exhausted (b1-power-law-contraction.md; base-shifts refuted). Composition is the strictly better profile; superseded.
- **Quadratic / power-law depth profile** — c0019 rejected, power-law exhausted. Two-stage composition already *is* the coupled-quadratic that works; a free quad coeff decouples it and fails.
- **SPSA / reduced-gradient direction probe** — refuted (spsa-ascent.md): single batched probe c0063 regressed; the (0.5,0.7) axis is a confirmed local optimum, so a data-driven local direction cannot beat it. The lever is *depth along the known axis*, not a new direction.
- **NAE-from-scratch / new basin** — HINT flags AVOID (lacks baseline L); high variance, no evidence of a >0.6361 basin outside B3.
- **Cross-basin grafting / homotopy / B4 sweeps** — exhausted/refuted (mode-grafting-and-blends.md, b4-*), and a portfolio cannot exceed its best basin.
- **m=1 micro-rotation / mid-m bumps / toroidal-axis** — c0020 broke QI (log10_qi −3.973); c0013/c0014 plateaued; all exhausted. They perturb *orthogonal* to the one axis that pays.
- **Over-gating on vlf viol ≤ 0.001** — explicitly counter-productive per HINT (feas 0.0096 scored official 0.6398); my LF check gates only at the true 0.010 tolerance.
