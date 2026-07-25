# Feasibility-Margin-Aware Polish
Polishing bank seeds with a greedy margin-aware acceptance key prevents tolerance-camping and guarantees val fidelity survival.

## How it was tried
- stellar_p2-s17-78763752 c0001f ACC train 0.6197, val 0.6286: Dropped the refuted top-2 recombination sweep from c0001. Applied a margin-aware penalty to the greedy polish step: `key = score - 8.0 * max(0, feasibility - 0.006) - 0.02 * [qi_camping]`. Changed the return logic to select the best boundary with `feasibility <= 0.0075` and `log10(qi) <= -4.05`.
- stellar_p2-s17-78763752 c0005/c0005f (ACC, current best family) keep this key as the acceptance arbiter under the momentum line search — see momentum-line-search.md.

## Why it worked
The parent (c0001) ignored constraints to maximize raw train score, resulting in a score of 0.6188 train but -0.9656 val (crushed by the fidelity gap). By heavily penalizing candidates sitting extremely close to the 0.01 feasibility violation limit, this candidate safely navigated away from the constraint bounds. The explicit return filter guaranteed that no tolerance-camping boundary could be submitted, bridging the train-val gap perfectly (val 0.6286 > train 0.6197).

## Verdict
promising — Always use a margin-aware polish (e.g., `LAM=8.0` penalty) when starting from highly optimized bank seeds. The next levers are tuning `SAFE` (0.006), `SAFE_RET` (0.0075), and `QI_SAFE` (-4.05) to see how much further the score can be pushed without re-triggering val collapses.
