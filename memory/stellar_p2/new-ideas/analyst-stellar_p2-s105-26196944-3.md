# Analyst notes — stellar_p2-s105-26196944 @ 36 candidates

The `INJECT/` write is gated by this session's permission mode — the same block every prior analyst in this campaign hit (see `analyst-stellar_p2-s105-26196944-2.md`, `-s105-72881323-3/4.md`). The complete standalone module is staged in the Write call above and needs approval to land at `INJECT/1-cooptimize-twostage-depth-split.py`. Below is the decision record.

---

# Analyst notes — stellar_p2-s105-26196944 @ 36 candidates

## What the search is doing
Branch B5-risoliao6 restarts from the campaign winner and re-polishes the **B3-lhhhhappy3 nfp=3 escape** with the one mechanism that works: the off-diagonal R/Z-split m-differential contraction `factor = 1 + base·(1 + curv·(m−1))`, `(cr,cz)=(0.5,0.7)`, now composed in **two stages**. Verified ladder:
- c0016 single-stage depth+cr/cz sweep → train 0.6255 / val 0.6362.
- c0017 **two-stage composition** → 0.6256 / 0.6371.
- c0022 deepened stage-2 straddling the aspect wall (`b2∈{-3.5,-4.5,-5.5}e-3`, `c2∈{0.3,0.1,0.0}`) → 0.6259 / 0.6372 (accepted; **b2=-3.5e-3 won**, deeper lost).
- **c0024 current best**: two-stage + fine `cr∈{0.50..0.58}` → **0.6262 / 0.6375**, official 0.6398 pending readj.
- **c0025→c0030: six straight rejects/errors** — every one a *new orthogonal axis* bolted on: N-dependent toroidal taper (c0025 ERR), finer-cr (c0025f rej), adaptive 3-stage coordinate-descent (c0026, an SPSA budget trap → 0.558), NAE third basin (c0028 rej), opposite-sign concave 3rd stage (c0029 rej), high-|n| toroidal damping (c0030 rej).

## Binding problem(s) now
1. **The search is spending every candidate on axes the wiki already marks refuted/exhausted** (toroidal/n-axis, NAE-from-scratch, iterative coordinate-descent, opposite-sign profiles) instead of the axis the HINT explicitly names. Six consecutive non-improving candidates = local saturation of the *sampled* directions, not of the mechanism.
2. **Aspect ratio is the sole binding constraint and the two-stage composition is L-efficient against it** (`feasibility == (aspect−10)/10` to the digit; elong 4.37/5 and log10_qi ≈ −4.005 *improving with depth* both have margin — per the @24 analyst). The proven ceiling is `feas ≈ 0.0096` (official 0.6398); c0024 sits just under it. Blind "deeper total depth" is *saturated* (c0022 showed deeper stage-2 lost).
3. **A specific interaction was never tested.** Stage-1 has been **frozen at the single-stage optimum** `(b1=−5e-3, c1=0.5)` since c0017; only stage-2 depth/curvature and `cr` were ever tuned. The composed row factor is `1 + (b1+b2) + [0.5·b1+0.3·b2]·(m−1) + b1·b2·(quadratic-in-m)`. At **fixed total `T=b1+b2`** (fixed aspect cost), shifting depth toward the steeper stage-1 raises the high-m damping slope **and** the quadratic-in-m content — the modes that set the worst-curvature min-L — *for free on aspect*. Nobody has swept this split.

## Decision: **continue (sharpened)** — and why
Not a pivot: every genuinely-orthogonal axis is refuted in the wiki and the HINT (toroidal, NAE, grafting, SPSA, truncation, anisotropic-exp, noise). Not a blind revive: RisoLiao#6/B4 basins have a lower L ceiling (risoliao6-basin-extension.md, b4-*.md). The evidence points at **one untested corner of the winning mechanism** — the stage-1↔stage-2 depth split at ~constant total depth. It is on-axis with the HINT ("push heavier aspect-relief… official rises with L"), physically motivated (decouples high-m damping from aspect cost), budget-safe (one batched `eval_many`, ~13 evals on the fast mpol=7 B3 boundary), and strictly non-regressing (incumbent is the floor). It is the highest-EV move that isn't already refuted.

## Proposal (the ONE candidate injected)
`INJECT/1-cooptimize-twostage-depth-split.py`. **Idea:** sweep the *split* of the two-stage contraction depth — `(b1,b2)` pairs across three total-depth bands (`T≈−8.0/−8.5/−9.0e-3`, straddling `feas≈0.0096`), with `c1=0.5, c2=0.3, cr=0.5, cz=0.7` fixed at proven values to isolate the split. 12 split points + the exact c0024 incumbent as a guaranteed floor, all in one batched `eval_many`; selection by the novelty-penalized key (`raw − 0.05·max(0,1−bank_dist/1e-3)`, penalty 0 here since contraction moves *away* from the 2.6e-3 bank ball), with an LF landmine gate that reverts to the incumbent on collapse (never re-ranks — lf-selection-bug.md). **Expected effect:** a stage-1-heavy split (e.g. `b1≈−6e-3, b2≈−2.5e-3` at `T=−8.5e-3`) beats train 0.6262 by ≥0.0004 by damping the min-L-setting modes without extra aspect cost; if none beats the incumbent, the two-stage L/aspect frontier is confirmed saturated and the branch should stop polishing and lock 0.6398.

## Decision log (alternatives considered and rejected, with reasons)
- **PIVOT to a data-driven ascent direction (batched finite-diff / SPSA):** refuted — spsa-ascent.md (all iterative loops timeout; the one budget-safe single-batch probe c0063 still hit floor; "(0.5,0.7) axis suboptimal" premise refuted).
- **Toroidal / n-axis / combined (m,n) contraction:** exhausted — toroidal-axis-contraction.md, and c0025/c0030 just re-failed it this window.
- **New feasible basin (NAE-from-scratch, third basin):** rejected — surrogate-and-nae-escapes.md + c0028 (NAE lacks baseline L); HINT explicitly says AVOID.
- **Cross-basin grafting / homotopy / blends:** exhausted — mode-grafting-and-blends.md.
- **Deeper *total* depth / power-law / quadratic single-stage profile:** exhausted — c0022 showed deeper stage-2 lost; alternative-m-profiles.md (quadratic c0019 refuted). My proposal deliberately holds total depth ≈constant and moves only the *split*, which is why it isn't the same as these.
- **RisoLiao#6 / B4 deep contraction (revive):** rejected — lower L ceiling (risoliao6-basin-extension.md, b4-deep-contraction.md).
- **Literature pivot:** the ConStellaration paper (arXiv:2506.19583) and the scale-length paper (arXiv:2309.11342 — min `L=B/|∇B|` is set by the sharpest-curvature region and dictates coil proximity) *confirm* the mechanism-level reasoning (smooth the worst-curvature feature to raise min-L) but offer no new budget-compatible optimizer this search hasn't tried; they justify the split proposal rather than replace it.

Sources: [ConStellaration (arXiv:2506.19583)](https://arxiv.org/abs/2506.19583), [Magnetic Gradient Scale Length (arXiv:2309.11342)](https://arxiv.org/pdf/2309.11342)
