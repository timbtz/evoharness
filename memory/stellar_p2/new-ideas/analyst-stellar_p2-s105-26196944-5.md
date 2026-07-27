# Analyst notes — stellar_p2-s105-26196944 @ 60 candidates

The injection module is authored (pending the write-permission approval to `INJECT/1-axis-excursion-aspect-relief.py`). Here is my decision record.

---

# Analyst notes — stellar_p2-s105-26196944 @ 60 candidates

## What the search is doing
Every candidate this run re-confirms and then perturbs one hardcoded object: the **B3‑lhhhhappy3 nfp=3 escape** deepened by the two‑stage R/Z‑split linear m‑contraction `[1+base·(1+curv·(m−1))]²`, base≈−5e‑3, (cr,cz)=(0.5,0.7). That boundary is the incumbent — **official 0.6398, val 0.6378, train 0.6269, bank_dist 2.6e‑3** — and it is submittable (beats the 0.6361 bar, >1e‑3 from every bank seed).

The recent window (c0042–c0050) is a graveyard of ties and regressions, all landing on the *exact* incumbent: n‑axis m‑shift (c0042, tie), 2nd‑stage cr split (c0043, rej), R0 rescale (c0044, tie), low→high‑m transfer (c0045, **t=0 wins**), joint two‑lever sweep (c0045f, rej), toroidal‑n differential (c0046, rej), rotation‑coupled 3rd stage (c0047, rej), nfp=2 NAE (c0048, rej), row‑1 depth multiplier (c0049, rej), major‑radius tilt on m=1 sidebands (c0050, tie). Not one moved train off 0.6269.

## Binding problem(s) now
I pulled the incumbent's metrics from the ledger (re‑scored 6×): **aspect_ratio 10.096 → normalized violation 0.0096 (the max, BINDING); log10_qi −4.0025 → margin ~0.0025 log (near‑active); edge_mirror 0.1967 → margin 0.0033 (near‑active); elongation 4.36 → 0.64 slack (loose).** This is a **constrained KKT vertex**: three near‑simultaneously‑active constraints. That is the mechanical reason *everything ties* — the feasible cone at the incumbent admits no improving direction along the m≥1 contraction lattice. Deeper base (−6e‑3, tried in c0045 round‑1 and c0002) pushes aspect over the wall; any redistribution that holds aspect (m‑transfer) also holds L (t=0 optimal). L rises only with contraction, contraction only raises aspect, and aspect is already at the wall. This is a **real wall, not the "false 0.633 ceiling"** the HINT warned about — that ceiling was already broken to reach 0.6398.

## Decision: pivot — and why
**CONTINUE is dead** (15+ consecutive ties on the saturated m≥1 lattice). **REVIVE is dead**: bank‑#6/RisoLiao contraction "plateaus immediately" (risoliao6‑basin‑extension), B4 is Pareto‑blocked at ~0.633, NAE‑from‑scratch lacks baseline L. So I **PIVOT to the one lever no candidate has touched and that attacks the actual binding constraint (aspect): the magnetic‑axis excursion — the m=0, |n|≥1 sidebands** `r_cos[0][ntor+1:]` (up to 0.057) and `z_sin[0][ntor+1:]` (up to 0.178). Aspect ratio for a QI configuration is inflated by axis wiggle at fixed minor cross‑section; **straightening the axis is a geometrically direct aspect reducer that is orthogonal to every m≥1 knob and distinct from R0‑mean rescale** (r0‑rescale, refuted, touches only `r_cos[0][ntor]`). If aspect drops, the freed budget funds a deeper L‑raising contraction. Grounded in the multi‑objective QI‑stellarator literature where aspect/complexity trade against QI along the axis shape (arxiv 2304.08698; magnetic‑gradient‑scale‑length ↔ coil/axis geometry, iopscience 10.1088/1741‑4326/ae69fa).

## Proposal (the ONE candidate injected)
`INJECT/1-axis-excursion-aspect-relief.py`. One batched `eval_many` of 8: the exact incumbent (floor) + straighten fractions s∈{0.03,0.06,0.10} at base −5e‑3 (isolate the aspect/QI response) crossed with deeper base ∈{−5.5e‑3,−6.0e‑3} (spend freed aspect budget on L). Selection by the novelty‑aware key `score − 0.05·max(0,1−bank_dist/1e‑3)`; the top improver must **strictly beat the incumbent's measured key AND survive a low‑fidelity feasibility gate** (QI margin is only ~0.0025 log, so straightening genuinely risks flipping QI infeasible — the LF gate catches that landmine). Fallback = the exact 0.6398 incumbent → **zero downside**. Expected effect: if axis straightening lowers aspect without breaking QI, objective_L clears 12.62 and train >0.6269 (official >0.6398); otherwise a clean, documented refutation of the axis‑excursion lever for future analysts.

## Decision log (alternatives considered and rejected)
- **Deeper base on m≥1 (continue):** refuted — c0045 round‑1 and c0002 show −6e‑3 loses (over the aspect wall). Aspect is at 10.096/10.10.
- **R/Z‑split‑ratio sweep to exploit the 0.64 elongation slack:** reduces to (cr,cz) sweep, already established strict‑optimal at (0.5,0.7) (per‑row‑rz‑and‑3d‑joint‑sweeps, c0043) — refuted.
- **m‑transfer / m=1‑selective aspect relief:** m‑transfer's neutral t=0 won (c0045); m1‑selective refuted. The vertex has no aspect‑neutral improving direction.
- **Bank‑#6 (violation 0.0, more headroom) contracted:** risoliao6‑basin‑extension marks it "plateaus immediately"; adds archive value but no new best, doesn't reset the stall.
- **NAE from scratch / nfp=2 / cross‑nfp graft:** surrogate‑and‑nae‑escapes, c0048, mode‑grafting — all refuted (no baseline L within 72 evals).
- **SPSA / iterative ascent along a measured feasibility gradient:** refuted (timeouts; the one budget‑safe batched probe c0063 still regressed).
- **m=0‑sideband *addition* (increase wiggle):** c0001 attempted an additive m=0 tilt but timed out (bad seed) and c0050 added to m=1 sidebands (tie). **Straightening (subtractive) the m=0 sidebands to reduce aspect is untried** — hence this injection.

Sources: [multi‑objective stellarator trade‑offs](https://arxiv.org/pdf/2304.08698), [gradient scale length ↔ coil/axis complexity](https://iopscience.iop.org/article/10.1088/1741-4326/ae69fa), [ConStellaration benchmark](https://arxiv.org/html/2506.19583).
