# nfp=3 Rebase + R/Z-Split Depth Contraction — the SUBMITTABLE 0.6398 escape
Abandon the pinned high-nfp bank seed; rebase to a hardcoded nfp=3 escape boundary
(B1 c0003f or B3-lhhhhappy3, both official ~0.633, bank_dist ~2.5e-3) and deepen it
with an off-diagonal R/Z-split m-differential contraction that walks aspect ratio into
its ~10.10 wall. This is the campaign high-water mark: **official/private 0.6398,
submittable, ABOVE the 0.6361 bar** — and it is a depth-contracted 0.633 escape, NOT a
new basin. The "0.633 ceiling" the in-run analysts assumed was an unreadjudicated myth.

## How it was tried (winning lineage)
- `s103-473410` c0085f (refiner, ACC, train 0.6192 / val 0.6314, bank_dist **0.002535**):
  the decisive move — abandoned the exhausted pinned bank-#4 (nfp=5/RisoLiao nfp=4) basin,
  in-ball for 84 candidates, and rebased to the proven nfp=3 B1 escape boundaries. First
  true out-of-ball feasible return of the branch.
- `s103` c0091 (ACC, train 0.6211): weighted blend of B1_PRIMARY×B1_SECONDARY (w=0.50,
  **same-nfp same-family** — NOT cross-basin) + aspect-relief contraction. Structurally novel.
- `s103` c0099/c0105 (ACC, train 0.6221 / val 0.6334 / **private 0.6352, submittable**):
  per-poloidal-mode (m-differential) contraction on the 0.50 B1 blend, contracting high-m
  more aggressively to relieve aspect while holding low-order QI.
- `s105-72881323` c0009 (ACC, train 0.6243): split the m-differential into **separate R (r_cos)
  and Z (z_sin) multipliers** `factor = 1 + base·(1 + curv·(m−1))`, base≈−3e-3, off-diagonal
  `(cr,cz)=(0.5,0.7)`. R↔aspect, Z↔elongation decoupled. Best off-diagonal split; the diagonal
  (cr=cz) is NOT a safe floor (anchor-floor-rejection-bug.md).
- `s105` c0027/c0029 (ACC, train 0.6248/0.6250): hardcoded the B3-lhhhhappy3 nfp=3 escape
  (official 0.6330) as an additional basin; tri-basin (B1+B3+B4) contracted portfolio in one eval_many.
- `s105` c0075f (refiner, ACC, train 0.6255 / val 0.6368 / **private 0.6398, submittable**):
  a cheap nfp=3 depth sweep that dropped the slow B4 evals and deepened the contracted
  B3 escape. The run's and campaign's best — `_contract(B3, d, 0.5, 0.7)`, bank_dist 0.002608.
- `s105` c0088r1 (ACC, = the SAME boundary): an nfp=2 NAE-pivot idea that failed and fell back
  to the contracted-B3 floor; run_end best_id, official 0.6398.
- `s105-26196944` c0017 (ACC, train 0.6256, val 0.6371): two-stage composed contraction
  `factor = [1+b1·(1+c1·(m-1))] · [1+b2·(1+c2·(m-1))]` evaluated in a batched sweep.

## Why it worked
Depth contraction raises objective_L (min gradient scale length) by pushing aspect toward its
wall; on a proven 0.633 nfp=3 escape this lifts the OFFICIAL score, not just val — private
0.6398 > val 0.6368 > train 0.6255 (positive generalization gap). The refiner (Opus) drove both
decisive jumps (c0085f rebase, c0075f floor). bank_dist is scale-normalized, so the off-diagonal
R/Z split (asymmetric, not uniform) earns real out-of-ball distance (~2.6e-3) that survives export.
Feasibility crept to **0.0096** yet the boundary is official+submittable — the strict vlf viol≤0.001
gate the HINT insisted on was NOT necessary for these contracted nfp=3 escapes.

## Verdict
promising — this is the proven winning recipe. Next: push the same depth-contraction FURTHER on
the B3/B1 nfp=3 escape (it was frozen at c0075f by a false-ceiling belief and a 60-cand stall, not by
a real wall); explore heavier aspect-relief now that we know official tracks L upward and feasibility
budget is ~0.01, not 0.001. Keep the exact off-diagonal incumbent as the non-regressing fallback.
