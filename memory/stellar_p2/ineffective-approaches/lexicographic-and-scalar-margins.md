# Lexicographic, Margin-Banking, and Scalar Margin Gates
Attempts to fix the deterministic 0.6161 plateau by replacing the tuned scalar `_key` acceptance arbiter with a two-tier lexicographic comparator, a continuous feasibility weight, a margin-banking branch, or lower SAFE thresholds. All fail because they attack the acceptance gate instead of the root cause: invisible or strictly L-rejecting polish mutations.

## How it was tried
- **Lexicographic Acceptance Gate (`stellar_p2-s101-20239089` c0008, c0011):** Replaced scalar penalty with a two-tier lexicographic comparator keyed on a margin bucket. Tied bit-exactly.
- **Lexicographic QI Tiebreak in LF Gate (`stellar_p2-s101-20239089` c0007f, c0008f, c0010f, c0011f):** Added a tie band to the LF return selector to prefer deeper QI margins. Scored bit-identically.
- **Lowered SAFE Threshold (`stellar_p2-s101-20239089` c0014f):** Dropped `SAFE` from 0.006 to 0.003 to engage the `LAM=8.0` penalty. Tied bit-exactly.
- **Margin-Banking (`stellar_p2-s101-20239089` c0015f):** Added an acceptance branch to accept candidates that preserved L but lowered feasibility violation by `MARGIN_EPS` to bank headroom. Tied bit-exactly.
- **Continuous Feasibility Key (`stellar_p2-s101-20239089` c0016):** Replaced binary feasibility cutoffs with `s * feas_weight(feas)`. Tied bit-exactly.
- **Margin-Robust LF Gate (`stellar_p2-s101-20239089` c0013f):** Dropped greedy descent for an LF gate tiebreak swapping near-L-tied finalists for the highest official margin. Tied bit-exactly.
- **Generic Feasibility Margin Gate (`stellar_p2-s102-48117936` c0002f):** Added an explicit QI-margin ratchet and a generic feasibility pad to buy headroom. Escaped successfully but the acceptance key certified based on generic `feasibility <= SAFE_RET`, missing that it had camped exactly on the QI wall (`log10_qi = -3.988`). Val collapsed to -0.28.
- **Margin-Razor Return Track (`stellar_p2-s102-48117936` c0009f, c0010f):** Replaced best-feasibility tracking with a dedicated razor tracker that only populated if `feasibility <= 0.003 AND log10_qi <= -4.0`. Tied bit-exactly at 0.5789 because no candidate ever reached the target margin headroom.

## Why it failed
At the constraint wall, the L-gradient and QI/aspect-gradients are strictly opposed. Restructuring the acceptance gate cannot generate score progress out of thin air. Lexicographic and tiebreak edits require a pool of distinct, near-tied finalists to select from, but the vlf-blindness death spiral means the pool only ever holds the Phase-0 escape winner. Continuous keys, margin-banking, and generic feasibility gates fail because they blind themselves to specific constraint wall-camping (like QI); any mutation the simulator can see strictly lowers L by violating the tight physical bounds, so the acceptance gate correctly rejects it. Hard margin-razor targets (0.003) simply stay empty.

## Verdict
exhausted — Do not restructure the scalar acceptance key to escape plateaus. The issue is mutation generation, not selection logic. If verifying feasibility, do not trust generic aggregate margins—explicitly check `log10_qi`.
