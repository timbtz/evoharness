# Analyst notes — stellar_p2-s105-26196944 @ 48 candidates

The INJECT write is permission-gated (as every prior analyst in this campaign hit). The complete standalone module is staged in the Write call above and needs approval to land at `INJECT/1-low-nfp-nae-basin.py`. Below is the decision record.

---

# Analyst notes — stellar_p2-s105-26196944 @ 48 candidates

## What the search is doing
This run polishes ONE basin: the **B3-lhhhhappy3 nfp=3 escape**, deepened by the two-stage R/Z-split m-differential contraction `[1+b1(1+c1(m−1))]·[1+b2(1+c2(m−1))]`, off-diagonal `(cr,cz)=(0.5,0.7)`. Verified trajectory (train / val / aspect / feasibility, from ledger mining):

| id | train | val | aspect | feas | L | log10_qi |
|----|-------|-----|--------|------|---|----------|
| c0016 | 0.6255 | 0.6362 | 10.0774 | 0.00774 | 12.510 | −4.0042 |
| c0024 | 0.6262 | 0.6375 | 10.0961 | 0.00961 | 12.524 | −4.0062 |
| **c0034** | **0.6269** | **0.6378** | **10.0963** | **0.00963** | 12.537 | −4.0025 |

c0035f/c0037r1 are refiner preservations of c0034 (identical metrics). **c0034 is the current best; official 0.6398, already submittable.** Since c0034, every candidate (c0035–c0040: n-axis contraction, power-law exponent, per-row R/Z, 3-D joint, twist) **rejected, errored, or tied**. The basin is in terminal stagnation.

## Binding problem(s) now
1. **Aspect ratio is the sole binding constraint and it is jammed against the wall.** c0034 sits at aspect 10.0963 / feasibility 0.009628 = **96.3 % of the 0.010 budget**; only 0.00037 feasibility (0.0037 aspect units) remain. `feasibility == (aspect−10)/10` to full precision. Every other channel has slack (elong 4.36<5, mirror 0.197, |iota/nfp| 0.262). **The mechanism that produced every gain in this campaign — walking aspect toward 10.10 to raise L — is exhausted.** L cannot rise further via aspect.
2. **The local search space is fully refuted.** The wiki's ineffective-approaches list now covers *every* perturbation axis on this basin: all m-profile shapes (quadratic/exp/quantized/power-law), all n-axis and (m,n) contractions, m=1-selective, per-row R/Z, radial translation, twist, rotation, Gaussian noise, constant-depth b1/b2 splits, cross-basin grafting/blends, homotopy, NAE-alongside, and all iterative optimizers (timeout). Continuing = grinding the same ≤0.001 gains.
3. **One multiplicative lever on L has never been pulled: field-period count.** The ConStellaration paper ([arXiv:2506.19583](https://arxiv.org/html/2506.19583)) states the objective scales as **L̃_∇B ∝ R₀/Nfp = aA/Nfp** — i.e. normalized L ∝ **A/Nfp**. With A pinned at the wall, **Nfp is the only remaining knob**, and the *entire* campaign (B1/B3 nfp=3, B4 nfp=4, bank seeds) has never seriously searched **nfp=2**. A grep of this run's ledger finds ~0 low-nfp exploration.

## Decision: **PIVOT** — and why
Continue is dead (aspect wall reached; every local axis refuted). Revive offers nothing: the abandoned solutions in this run/campaign are all nfp=3/4 variants that regressed to floor. The paper's own scaling law hands us the untried lever — at fixed aspect=10, **nfp=2 gives ~1.5× the L of nfp=3** (12.5 → ~18–19 → score ~0.9 *if feasible*). The known risk is QI: lower Nfp worsens omnigenity, and QI margin is already thin (log10_qi −4.0025). But near-axis (NAE) QI quality *improves* with aspect (1/A expansion), and we seed at aspect 9–10 where NAE is at its best; whether log10_qi holds ≤ −4 is the single empirical question worth one candidate. With free `seed_nae` seeds and a hard fallback to the rediscovered c0034 frontier, **downside is zero and upside is a category jump plus a genuinely novel basin** (distinct nfp ⇒ large bank_dist ⇒ novelty-clean, and rewarded by the cross-run archive).

## Proposal (the ONE candidate injected)
**`INJECT/1-low-nfp-nae-basin.py` — grow a fresh nfp=2 QI basin exploiting L∝A/Nfp.** (See `new-ideas/low-nfp-nae.md` for the distilled mechanism).
