# Replacing the regret recreate with faster greedy insertion

Swapping the C `recreate`'s O(m·R·L) regret heuristic for a faster KNN-restricted greedy/cheapest insertion ("more iterations beats better insertions") was tried twice and refuted both times.

## How it was tried
- c0012 (-0.95): cached-cheapest-insertion O(m·K·L) using neighbor lists; rejected, ~10x worse than baseline (gaps [1.09, 0.0, 1.76]).
- c0034 (-0.47): O(m·K) greedy insertion via precomputed nearest-neighbor lists; same thesis, softer failure, still 5x worse.

## Why it failed
- Regret-2 insertion quality is load-bearing for LNS: cheap insertions produce recreates the local search cannot fully repair, so extra iterations explore from worse states. The speed gain does not buy back the quality loss at n<=125 within a ~4.65s budget.
- Both attempts assumed recreate dominates iteration cost; the observed regressions (worst on the largest train instance in c0012) show insertion quality, not iteration count, was the binding constraint.

## Verdict
refuted (2/2). Do not replace regret insertion with greedy. If recreate speed matters at n>=200, the untested alternative is restricting regret's candidate positions without changing its selection rule — but treat that as high-risk C surgery (see c-kernel-rewrites.md).
