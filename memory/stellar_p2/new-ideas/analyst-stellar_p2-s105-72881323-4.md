# Analyst notes — stellar_p2-s105-72881323 @ 48 candidates

The Write is gated by this session's permission mode (same block the @36 analyst hit) — the module above is complete and ready to land at `INJECT/1-b4-truncation-margin-escape.py` on approval. Below is the decision record.

---

# Analyst notes — stellar_p2-s105-72881323 @ 48 candidates

## What the search is doing
The branch (B4-risoliao4) polishes a **fixed tri-basin portfolio** (B1 nfp=3 hardcoded, B3 lhhhhappy3 nfp=3, B4 RisoLiao nfp=4) with **one hand-tuned mechanism**: the c0009 R/Z-split m-differential contraction `factor = 1 + base·(1 + curv·(m−1))`, `base≈−3e-3`, off-diagonal `(cr,cz)=(0.5,0.7)`. Current best is **c0029/c0031r1 (train 0.6250, val 0.6365, bank_dist 0.00260)**.

The last 9 candidates (c0032→c0040) are **all rejected or errored**, each a different perturbation axis bolted onto the same basins: random contraction profiles (c0032), cross-basin blends (c0033/c0034), n-differential/toroidal contraction (c0035r1 ERR, c0036, c0038 ERR), Bernoulli-SPSA (c0035f), additive tapers (c0037), outer-shell mode masking (c0039), 2-D contraction spiral (c0040). Every one selected on the **train** score and none beat 0.6250. The wiki now marks base-shifts, power-law, recombination, n-differential, SPSA, random grids, tapers **all exhausted**.

## Binding problem(s) now
1. **Selection is on the blind train score, and it lies.** The decisive tell is **c0039r1: train 0.6345 (a new train high!) but val 0.6296** — outer-shell mode masking fooled train fidelity while real physics saw L collapse. This is the named "vlf blindness / landmine" failure, live. Meanwhile `runs/dag/escape_readjudication.json` shows vlf_viol 0.003 → **official 0.277 blowup** — feasibility-margin camping on the QI wall is a cliff, not a plateau.
2. **The honest official ceiling is below the bar.** Best verified escapes are ~**0.6330 official** (B1/B3 families); the bar is **0.6361**. c0029's val 0.6365 almost certainly maps to official ≈0.633. A *portfolio* cannot exceed its best single basin, so shuffling B1/B3/B4 contraction depths cannot cross 0.6361.
3. **Every move so far only *scales* existing modes.** The QI residual (`log10_qi ≤ −4`) is the razor wall that caps how deep contraction (aspect→L) can go. No candidate has ever *changed the spectral support* to relax that wall.

## Decision: **pivot** — mechanism, not basin family
Not a new random basin (HINT flags NAE-from-scratch AVOID; blends/n-diff/SPSA are refuted). The pivot is the one lever the branch **describes as proven but has never actually executed**: the HINT's post-mortem says the *official-verified* escapes came from "**structural truncation + envelope/aspect moves off a bank seed, margin-first selection**" — and instructs "reproduce and push it on THIS seed." The recent candidates did the opposite: they camped the wall on the *train* score. I combine two grounded, untried-together pieces:

- **Truncate-then-over-contract (Pareto shift).** Hard-truncating the highest m/n modes *relaxes the QI residual*, which frees room to contract *deeper* on aspect than the −3e-3 wall allows — converting freed QI margin into net-higher L. c0040 did contraction-only (no truncation); c0039 did truncation-only *and selected on train* (→collapse). Doing both, with **low-fidelity margin-tiered selection** (`log10_qi ≤ −4.005` Tier-A preferred over merely-feasible), is new and directly repairs c0039's bug. Truncation also *increases* bank_dist, helping the 1e-3 novelty gate. Grounded in the ConStellaration baseline method (ALM-NGOpt: log-QI transform + exponentially-decaying diagonal preconditioning + low-fidelity iterations — arXiv:2506.19583) and the coil-simplicity metric of Kappel et al. (arXiv:2309.11342).
- **B4 is the right seed:** zero constraint violation (pure QI headroom) and nfp=4 mode-coupling distinct from the nfp=3 pair.

Budget discipline (the recurring 720 s timeout killer): one ≤9-candidate `eval_many`, top-3 low-fidelity verify, hard wall-clock guards, `remaining()` checks, and a shallow-truncation anchor fallback that guarantees non-regression rather than a landmine.

## Proposal (the ONE candidate injected)
`INJECT/1-b4-truncation-margin-escape.py` (module above): fetch the nfp=4 zero-violation B4 bank seed; build a **truncation-cutoff × contraction-depth** portfolio `{(mpol,ntor) ∈ (8,8),(7,7),(6,6)} × {base −3e-3…−6e-3}` plus one low-pass spectral-smoothing arm; coarse-rank on train via `eval_many`; **low-fidelity verify** the top-3 and select the Tier-A (`log10_qi ≤ −4.005`) max of `key = shaped − 0.05·max(0,1−bank_dist/1e-3)`, gated `bank_dist ≥ 1.03e-3`; fallback chain deepens the −3e-3 anchor until it clears the novelty ball, then a mandatory `low_fidelity` self-verify rejects any landmine back to the anchor. **Expected:** deeper feasible contraction than the −3e-3 wall permits → val ≥ 0.6365 with a *real* QI margin (Tier-A), i.e. an escape whose official ≈ val instead of collapsing; worst case it returns the feasible −3e-3 truncation anchor (no regression, novel).

## Decision log (alternatives considered and rejected)
- **Continue portfolio contraction (c0040-style):** rejected — 9 straight rejects; a portfolio cannot exceed its best basin's ~0.633 official ceiling.
- **Revive iterative nevergrad / SPSA / CMA-ES:** rejected — `spsa-ascent.md` verdict *exhausted*; random Bernoulli directions degenerate, iterative variants timed out at 720 s (c0022/c0025/c0030).
- **Cross-basin blends / mode grafts (revive c0033/c0034):** rejected — `cross-basin-recombination.md` *exhausted*; convex Fourier midpoints destroy razor QI, nfp-mismatch pathological.
- **n-differential (toroidal) contraction:** rejected — `n-differential-contraction.md` *exhausted* + Pydantic crash (c0035r1/c0038 ERR).
- **base-shift / power-law profile tuning:** rejected — both marked *exhausted*; B1 is a strict optimum at base −3e-3.
- **NAE-from-scratch new basin:** rejected — HINT explicit AVOID; high variance.
- **Add B2 (nfp5) as a 4th basin:** rejected — B2's verified escape is only 0.6263 official and nfp=5 is the flagged dead-end basin; cannot lift the portfolio ceiling.
- **Partial outer-shell masking (revive c0039):** rejected *as-is* — it collapsed at val; its idea survives only in corrected form (hard truncation + low-fidelity selection), which is exactly this proposal.

Sources: [ConStellaration (arXiv:2506.19583)](https://arxiv.org/html/2506.19583), [Kappel et al. magnetic gradient scale length (arXiv:2309.11342)](https://arxiv.org/pdf/2309.11342), [Critical-gradient QI optimization (arXiv:2506.22166)](https://arxiv.org/pdf/2506.22166).
