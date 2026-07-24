# Seed optimizer baseline and the constraint landscape
The seed (NAE portfolio + decayed Gaussian greedy, 72 evals) ends at shaped -0.665; feasibility is the whole early game and QI is the wall.

## Measured (2026-07-23, train split, deterministic)
- Raw NAE seeds (mp=1): shaped -0.95..-1.08. Dominant violations: edge mirror ratio ~0.39-0.42 (bound 0.2), edge iota/nfp ~0.04-0.13 (bound >= 0.25), log10(qi) ~ -1.9..-2.2 (bound <= -4).
- Seed after 72 evals: shaped -0.665 (mirror 0.32, iota 0.084, log10qi -1.40, L 3.4, aspect 8.1). All three violations shrink together, none is closed.
- iota/nfp is nearly a FREE parameter at seed time via generate_nae's rotational_transform argument — but requested rt is NOT reproduced by the mp=1 truncation (asked 0.84-0.9, measured iota*nfp ~ 0.1-0.4). Higher mp seeds reproduce iota better but wreck mirror ratio (0.51+) and elongation.
- Official baseline comparison: ALM needed thousands of lf evals to reach feasibility. Nobody crosses the QI wall in 72 vlf evals from a raw NAE seed; expect shaped scores in (-0.7, -0.2) for good candidates for a while.

## Why it matters
- Selection pressure works below feasibility: shaped = -max_violation is a clean descent signal. Candidates that trade violations (e.g. crush mirror at the cost of iota) don't move max-violation — coordinated descent wins.
- The archive keeps every frontier boundary: once ANY candidate crosses into feasibility (score > 0), that boundary seeds everything after.

## Verdict
promising — treat "reach feasibility at all" as the current open problem; score in (0, 0.431] is the next one; 0.431+ is the session claim.
