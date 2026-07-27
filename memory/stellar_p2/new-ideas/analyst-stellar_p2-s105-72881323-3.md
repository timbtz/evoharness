# Analyst notes — stellar_p2-s105-72881323 @ 36 candidates

The Write is blocked by this session's permission mode (every write — `/tmp`, the run dir, and `INJECT/` — is gated and I can't self-approve). The complete module is ready in my final Write call above and needs approval to land at `INJECT/1-vlf-gated-spsa-ascent.py`. Below is the decision record.

---

# Analyst notes — stellar_p2-s105-72881323 @ 36 candidates

## What the search is doing
The run is polishing a **fixed set of hardcoded escape basins** (B1 nfp=3, B3-lhhhhappy3 nfp=3, B4-risoliao nfp=4) with **one hand-tuned mechanism**: the c0009 R/Z-split m-differential contraction `factor = 1 + base·(1 + curv·(m−1))`, `base=−3e-3`, off-diagonal `(cr,cz)=(0.5,0.7)`. Candidates are static `eval_many` grids over `(base, cr, cz)` and shrink-to-fit padding to add basins. Selection is on **train/val**.

Trajectory: val 0.6349 (c0009 B1) → 0.6350 (c0026 base-shift) → 0.6360 (c0027 B3, c0028 B3-batched) → **0.6365 (c0029 tri-basin, current best, bank_dist 0.00260)**. Every iterative/nevergrad attempt (c0022, c0025, c0030) **timed out at 720 s** — eval starvation. The wiki marks B1 grid, base shifts, power-law, recombination, and iterative nevergrad all **exhausted**.

## Binding problem(s) now
1. **The search optimizes the wrong metric.** `runs/dag/escape_readjudication.json` (official scores) is decisive: vlf_shaped 0.6192 / vlf_viol 0.00093 → **official 0.6330**; vlf_shaped 0.6044 / vlf_viol 0.00299 → **official 0.0 (feas 0.277 blowup)**. Low-fidelity shaped + a **strict viol ≤ 0.001** gate ranks and protects official correctly; **train/val over-report** (c0029's val 0.6365 almost certainly maps to official ≈ 0.633, *below* the 0.6361 bar). Climbing val is feasibility-margin camping — the named recurring failure.
2. **One fixed search direction.** Contraction moves along a single hand-picked axis; the razor QI wall caps each basin's L. No candidate has ever estimated a data-driven ascent direction — every attempt to (nevergrad) starved the budget.
3. **No out-of-ball basin has crossed 0.6361 officially** (demonstrated ceiling 0.6330). Continuing pure contraction cannot win.

## Decision: **pivot** — and why
Pivot the **selection metric and the search mechanism** (not the basin family — a blind NAE-from-scratch is high-variance and HINT-flagged AVOID). Two grounded levers the search has never combined:
- **Select on low-fidelity, tiered by margin** (feasibility ≤ 0.001 preferred over merely-feasible), always subject to `bank_dist ≥ 1.03e-3`. This is the exact metric the re-adjudication proves tracks official and catches the 0.003→0.277 blowups.
- **SPSA random-direction line search** in place of the exhausted contraction axis. SPSA estimates a full-dimensional descent direction from **2 evals regardless of the ~200-D Fourier space** ([Spall/SPSA](https://www.emergentmind.com/topics/simultaneous-perturbation-stochastic-approximation-spsa), [arXiv:2203.03075](https://arxiv.org/abs/2203.03075)); the shaped-score cliff gives **free Wang–Spall "switch updating"** — ascend L when feasible, restore feasibility when not ([Wang & Spall, CDC03](https://www.jhuapl.edu/spsa/PDF-SPSA/wang_spall_CDC03.pdf)). Everything is batched (2–3 `eval_many` calls with hard budget/wall guards), so it **cannot starve** like the iterative nevergrad runs.

## Proposal (the ONE candidate injected)
`INJECT/1-vlf-gated-spsa-ascent.py` (complete standalone `solve(fm, rng)`, contract matching the current best; **pending write approval**).
- **Idea:** vlf-gated SPSA constrained ascent.
- **Mechanism:** (A) build a contraction floor over B1 (hardcoded, cheap, out-of-ball) + the best cheap nfp=3 bank seeds; (B) from the best-margin floor point, one–two SPSA rounds — 3 relative Bernoulli directions × 2 probes → pick the improving `(dir,sign)` → 4-step batched line search along it, objective `J = shaped − 8·max(0, feas − 0.001)` (margin-first); (C) return the boundary maximizing a tiered key `(margin-tier, shaped)` with `bank_dist ≥ 1.03e-3`, self-verified at low_fidelity, with **contracted-B1 as a guaranteed feasible out-of-ball fallback** so it never regresses to garbage.
- **Expected effect:** finds an L-gain direction the single contraction axis cannot, and — critically — returns the boundary that survives *officially*, not the one with the prettiest val. Falsifiable: if SPSA yields no direction with higher low-fid L at feas ≤ 0.001, it returns the floor and the pivot is refuted for this basin family (→ next step is a genuinely new higher-L basin).

## Decision log (alternatives considered and rejected)
- **Continue contraction polishing (more `base`/`curv`/basin grids).** Rejected: wiki marks B1-grid, base-shifts, power-law, recombination all *exhausted*; the escape data shows this raises val while official is stuck at the 0.633 ceiling. c0030 (deep-B3 contraction) already timed out.
- **Iterative nevergrad / CMA-ES surrogate.** Rejected: wiki *exhausted* — c0022/c0025/c0030 all timed out at 720 s; ~2–15 s/eval makes ask/tell loops infeasible. (SPSA's 2-eval-per-gradient batched form is the budget-safe replacement.)
- **Central-difference gradient over K modes.** Rejected: 2K evals is starvation on any headroom basin; SPSA gets a full-dim direction in 2.
- **NAE-from-scratch / new basin.** Rejected for *this* injection: HINT-flagged AVOID, high-variance, and prior NAE attempts failed to reach feasibility. Kept as the fallback direction if vlf-gated SPSA confirms the escape family is capped below 0.6361.
- **Micro-polishing / returning a bank seed.** Rejected: export-refused inside the 1e-3 ball; worthless per novelty bar.
- **Anchor SPSA on c0029's exact B3 boundary.** Wanted but not feasible: its coefficients live only as escaped JSON inside the ledger; hand-extracting 8×15 matrices is error-prone and I cannot run Python (all execution/writes are permission-gated this session). Mitigated by reconstructing the escape region at runtime from cheap nfp=3 bank seeds + the proven contraction, with hardcoded B1 as the guaranteed floor.
