# Analyst notes — stellar_p2-s105-72881323 @ 60 candidates

The Write is gated by this session's permission mode — the same block the @36 and @48 analysts hit. The complete module is ready in my Write call above and needs approval to land at `INJECT/1-exp-anisotropic-topbank-escape.py`. Below is the decision record.

---

# Analyst notes — stellar_p2-s105-72881323 @ 60 candidates

## What the search is doing
For 60 candidates the branch **B4-risoliao4** has polished a **fixed tri-basin portfolio** — B1 (nfp=3, hardcoded, official 0.6330), B3 (lhhhhappy3, official 0.6330), B4 (RisoLiao, official 0.6236) — with **one hand-tuned mechanism**: the c0009 R/Z-split m-differential contraction `factor = 1 + base·(1+curv·(m−1))`, `base≈−3e-3`, `(cr,cz)=(0.5,0.7)`. Best is **c0046 (train 0.6250, val 0.6365, bank_dist 0.0026)** — a bit-identical reproduction of c0029.

The recent window (c0042→c0050) is **all rejected/errored**: per-mode rotation (c0042 ERR), strict-LF tri-basin (c0043), bank-#4 truncation (c0044), (base,curv) micro-probes (c0045/c0045f), adaptive coordinate descent (c0047 ERR), cross-basin blends (c0048 ERR), n-differential toroidal taper (c0049), R/Z sign-pair perturbation (c0050). Every one selected on the **train** score; none beat 0.6250. The wiki now marks base-shifts, power-law, recombination, n-differential, SPSA, random grids, tapers, mode-erasure, and truncation-on-B4 all **exhausted**.

## Binding problem(s) now
1. **The honest official ceiling of every basin the branch touches is ~0.633 — below the 0.6361 bar.** `runs/dag/escape_readjudication.json` is decisive: `vlf_shaped 0.6192 / vlf_viol 0.00093 → official 0.6330`; `0.6190 / 0.00075 → 0.6321`; B2 family `0.6161 / 0.00394 → 0.6263`; and `0.6044 / 0.00299 → official 0.0` (feas 0.277 blowup). c0046's val 0.6365 maps to official ≈0.633. **A portfolio cannot exceed its best basin**, so no contraction depth on 0.6236/0.6330 seeds can ever cross 0.6361. The branch is structurally doomed and about to hit its 60-no-improvement terminate rule.
2. **The one basin near the bar (~0.636 top bank seed) has never been engaged.** c0001–c0003 timed out on high-mode bank seeds (12–27 s/eval), so the branch retreated to cheap 0.633-capped basins and stayed there. The near-bar basin was abandoned for *eval cost*, not for physics.
3. **Selection is on the blind train score, which over-reports** (c0039r1: train 0.6345 high, val 0.6296 collapse — live vlf-blindness). Margin razor alone is insufficient (vlf 0.003 → official 0.277).

## Decision: **pivot** — engage the only basin that can cross the bar
Both prior analysts (@36, @48) correctly diagnosed "pivot mechanism, not basin," but each stayed inside the 0.633-capped families (strict-LF selection; truncation on B4). Given the readjudication evidence, that is provably insufficient — **the mechanism pivot must be attached to the ~0.636 seed**, the sole basin with headroom to 0.6361. The eval-cost objection that blocked this dissolves once high modes are suppressed/truncated: the scaled seed evaluates cheaply, exactly like the "simple" 1.4 s boundaries.

The move is grounded in **Exponential Spectral Scaling (arXiv:2509.16320)** — mode-dependent, **anisotropic (m,n)** Fourier scaling, the technique class used for this exact ConStellaration boundary problem. This matters for novelty: `bank_dist` is scale-normalized, so **uniform rescaling earns zero distance** (why the search's contractions all sit at fixed 0.0026 off B1); an *anisotropic* exp move is the minimal coordinated multi-mode perturbation that both leaves the 1e-3 ball structurally and gently raises L by suppressing high-m wiggle — the "coordinated multi-mode move that leaves the ball while holding feasibility" the task names as the open game.

## Proposal (the ONE candidate I inject)
**`INJECT/1-exp-anisotropic-topbank-escape.py`** — *anisotropic exponential (m,n) escape off the top bank seed, margin-first, with a verified fallback.*

- **Mechanism:** locate the highest-scoring feasible bank seed via `fm.seed_bank_info()`. Build a ~6-variant portfolio applying `factor(m,n)=exp(−(a_m·m + a_n·|n|))` (R/Z-split `a_m`, per the proven aspect↔R / elongation↔Z coupling) plus a light `mpol/ntor` cap. This (a) suppresses high modes → raises L and **collapses eval cost**, (b) is non-uniform → earns real `bank_dist`, (c) the `a_n` taper exploits the paper's finding that QI needs toroidal resolution.
- **Selection (HINT rules):** one `eval_many` at train fidelity; keep `shaped>0.6243 ∧ feasibility≤0.006 ∧ log10_qi≤−4.003 ∧ bank_dist≥1.03e-3`; **strict LF-verify** top 3 (`viol≤0.0012`, guarding the 0.003→0.277 cliff); rank by `key = shaped − 0.05·max(0,1−dist/1e-3)`.
- **Non-regression:** fallback = **contracted B1_PRIMARY** (verified feasible, official ~0.632, dist 0.0025). Returned unchanged if nothing escapes+verifies — so the injection can only help.
- **Expected effect:** at least one variant LF-verifies feasible at `bank_dist≥1.03e-3` with train shaped >0.6243, giving the branch its first candidate whose *basin* can reach official >0.6361 — a strictly better bet than any 0.633-capped polish. Falsifiable: if all variants collapse QI or stay in-ball, it returns the fallback, cleanly disproving "the top seed is reachable by scaling alone."
- **Budget:** capped/scaled variants eval cheaply; single `eval_many` (~30 s) + ≤3 LF (~30 s) with hard `T_NO_LAUNCH=170 s` / `DEADLINE=225 s` guards. No timeout risk.

## Decision log (alternatives considered and rejected)
- **Continue tri-basin contraction polish (c0043–c0050 family)** — rejected: readjudication proves the ceiling is 0.633 < 0.6361; mechanically guaranteed to miss the bar. Wiki marks the axis exhausted.
- **Strict-LF selection on existing basins (@36 proposal)** — rejected: correct gate, wrong basins. It protects official score but cannot lift the 0.633 ceiling.
- **Truncation on B4 (@48 proposal / c0044)** — rejected: already tried, train 0.6237; B4's basin caps at 0.6236 official regardless.
- **SPSA / nevergrad / coordinate-descent ascent** — rejected: refuted (wiki `spsa-ascent`), and every attempt (c0022, c0025, c0030, c0047) starved/timed out. Full-D noisy gradients are the wrong tool for 12–27 s evals.
- **NAE-from-scratch new basin** — rejected: HINT flags AVOID; high-variance, no evidence any NAE basin reaches 0.636.
- **Cross-basin recombination / mode-grafting** — rejected: wiki `cross-basin-recombination` refuted (breaks QI balance); c0033/c0034/c0048 errored/regressed.
- **Uniform deeper contraction on the top seed** — rejected: `bank_dist` is scale-normalized → uniform rescaling earns zero novelty; would be refused at export. Anisotropy is mandatory, hence the exp (m,n) form.
- **Derivative-free trust-region surrogate (arXiv:2510.27396 / linear-subspace DFO)** — considered as a stronger long-term mechanism but rejected for *this* injection: fitting even a local linear model costs a design-of-experiments the 240 s deadline can't afford on 12–27 s evals. Noted as the natural follow-up once the top-seed basin is shown reachable.

**Sources:** [Exponential Spectral Scaling (arXiv:2509.16320)](https://arxiv.org/html/2509.16320) · [High-dimensional DFO via trust-region surrogates in linear subspaces](https://www.sciencedirect.com/science/article/abs/pii/B9780443288241505330) · [Distributed DFO / Trust-Region (arXiv:2510.27396)](https://arxiv.org/html/2510.27396)
