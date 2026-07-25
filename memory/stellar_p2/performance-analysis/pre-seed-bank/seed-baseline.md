# Seed optimizer baseline and the constraint landscape (from-scratch NAE era)
> HISTORICAL — pre-seed-bank program (runs s42/s7/s11). Feasibility was never reached from NAE seeds at candidate eval scale; the wall was crossed 2026-07-24 by adopting fm.seed_bank instead. See performance-analysis/seed-bank-regime.md for the current game.

The seed (NAE portfolio + decayed Gaussian greedy, 72 evals) ends at shaped -0.665; feasibility was the whole early game and QI the wall.

## Measured (2026-07-23, train split, deterministic)
- Raw NAE seeds (mp=1): shaped -0.95..-1.08. Dominant violations: edge mirror ratio ~0.39-0.42 (bound 0.2), edge iota/nfp ~0.04-0.13 (bound >= 0.25), log10(qi) ~ -1.9..-2.2 (bound <= -4).
- Seed after 72 evals: shaped -0.665 (mirror 0.32, iota 0.084, log10qi -1.40, L 3.4, aspect 8.1). All three violations shrink together, none is closed.
- iota/nfp is nearly a FREE parameter at seed time via generate_nae's rotational_transform argument — but requested rt is NOT reproduced by the mp=1 truncation (asked 0.84-0.9, measured iota*nfp ~ 0.1-0.4). Higher mp seeds reproduce iota better but wreck mirror ratio (0.51+) and elongation.
- Official baseline comparison: ALM needed thousands of lf evals to reach feasibility. Nobody crossed the QI wall in 72 vlf evals from a raw NAE seed; best shaped scores plateaued at -0.425 (s11 c0027f).

## Why it still matters
- Selection pressure works below feasibility: shaped = -max_violation is a clean descent signal — relevant again if repairing the 3 infeasible-at-vlf bank seeds (tolerance campers).
- The archive keeps every frontier boundary: any feasibility crossing seeds everything after.

## Verdict
historical — superseded by the seed-bank regime; retains the constraint-landscape numbers and the shaped-score descent mechanics.
