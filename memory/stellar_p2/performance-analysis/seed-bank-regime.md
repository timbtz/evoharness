# The seed-bank regime: the game is now beating 0.636
fm.seed_bank exposes 12 public leaderboard P2 submissions (official 0.40-0.636, all nfp=3). The novelty bar is now BEATING the bank max AND leaving the 1e-3 max-coefficient ball.

## Measured (vlf = train fidelity)
- Top-7 bank seeds hold 0.60-0.62 at vlf (official 0.61-0.64).
- 3 seeds (official 0.434-0.537) flip INFEASIBLE at vlf (violations 0.0106-0.0122).
- COST: high-mode bank boundaries take 12-27 s/eval at vlf vs 1.4 s for mp=1.
- L headroom: bank best L~12.3 at vlf; score 1.0 needs L=20.

## The Novelty Wall (`bank_dist < 1e-3`)
The harness penalizes candidates inside a 1e-3 max-coefficient ball around any bank seed. The penalty exactly matches: `0.05 * (1 - bank_dist/1e-3)`.
- **The objective trap:** `fm.score()` returns the *raw* shaped score and is BLIND to the novelty penalty. If an optimizer uses `fm.score()` for its acceptance key, it will ALWAYS prefer the unpenalized optimum of the bank seed over any boundary that escapes the ball (losing ~0.05 of score).
- **FIXED tooling (2026-07-24 evening, B2+):** `fm.bank_dist(boundary)` now returns the exact guard metric for free (no eval budget, no hand-rolled padding). Acceptance key: `score - 0.05 * max(0, 1 - fm.bank_dist(b)/1e-3)`.
- **Escape velocity:** Polish micro-steps (sigma=0.0012) perturb coefficients by ~1e-4. Passive random walk is too slow. 

## Triage & Eval Starvation
- Evaluating top seeds sequentially wastes time; batch via `eval_many`.
- Measured (B1, s100 run): c0012 top-7 triage (0.5679) wasted evals for nothing; c0021 skipping the best seed's metered eval regressed (0.5728); c0021f batching top-3 into ONE `eval_many` worked (0.5894). Batch, don't skip.
- Deadlocked re-probes rapidly burn budget. See `ineffective-approaches/momentum-variants.md`.

## Breaking the Wall (Branch B2 Update)
- **Feasible escape achieved!** `stellar_p2-s101-20239089` c0001 crossed 1e-3 (bank_dist 0.00125) feasibly using mode-truncation + single-pivot escapes (see `successful-patterns/structural-ball-escape.md`).
- **The QI Trade-off:** Escaping the ball exposes the QI constraint wall immediately. Activating a `QI_LAM=1.0` penalty safely bought QI margin but heavily starved the L-polish (train dropped 0.038). 

## Verdict
Structural escapes (truncation/pivot) are the way out of the 1e-3 ball. Pair them with an LF-verified return gate to survive the vlf-to-lf fidelity gap. Future candidates must carefully tune `QI_LAM` to survive the QI wall without destroying raw L-score.
