# Instance structure: customers-per-route (n/k) governs mechanism payoff

Generalization failures between val and private were NOT about instance size n — they track n/k (customers per route). Mechanisms that win on long-route instances regress short-route ones.

## Evidence (head-to-head, 3 evals each, 2026-07-23)
- c0024 (multi-restart + segment-shift lineage) vs c0035 (plain Or-opt overlay):
  - X-n176-k26 (6.8 c/r): c0024 0.09-0.29 vs c0035 0.74-0.80 — c0024 wins
  - X-n214-k11 (19.5 c/r): c0024 0.33-0.75 vs c0035 0.76-0.94 — c0024 wins big
  - X-n200-k36 (5.6 c/r, private): c0024 0.66-0.70 vs c0035 ~0.52 — c0035 wins
  - X-n251-k28 (9.0 c/r, private): c0024 0.65-0.83 vs c0035 0.45-0.64 — c0035 wins
  - X-n303-k21 (14.4 c/r, private): c0024 0.81-0.86 vs c0035 0.86-0.94 — c0024 slightly wins
- Mechanism reading: segment-shift/Or-opt moves have more useful positions per route when routes are long; with short dense routes (k large) the payoff shifts to inter-route balance and the C LNS itself.

## Consequence
- val was rebuilt 2026-07-23 to match private's n/k regime: X-n153-k22 (7.0), X-n228-k23 (9.9), X-n242-k48 (5.0). Improvements must now pay on short/mid routes to survive the gate.
- When judging any approach, note WHICH n/k regime the evidence comes from; "refuted" or "successful" without a regime qualifier is incomplete (this already flipped multi-restart once).

## Verdict
promising as an analysis lens; see new-ideas/regime-dispatcher.md for the constructive use.

## Budget regime (2026-07-23)
- Same lesson as n/k, on the time axis: mechanisms are budget-conditional. Ours beats PyVRP-HGS ~3x at 6s/instance; PyVRP wins at 30-60s (0.49-0.88% vs our untested-at-30s).
- Accepted candidates now also get a val30 score (val instances at 30s). A candidate that improves val30 without losing val is MORE generalizable and preferred; the run reports both Pareto champions (best.py / best30.py).
