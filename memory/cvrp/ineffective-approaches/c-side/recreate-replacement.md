# Replacing or micro-optimizing the regret recreate

Swapping or tweaking the C `recreate`'s O(m·R·L) regret heuristic failed in all attempts.

## How it was tried
- c0012 (cvrp-s3-45465035): cached-cheapest-insertion. ~10x worse than baseline.
- c0034 (cvrp-s3-45465035): O(m·K) greedy insertion. 5x worse.
- c0061 (cvrp-s5-54146615): recreate micro-optimization. Silent failure.
- c0002 (cvrp-s11-66566581): demand-urgency tie-break (`DEM[c] > bdc`). Train -0.1317 (rejected, worse than baseline -0.1064).

## Why it failed
- Regret-2 insertion quality is load-bearing for LNS.
- Tinkering with recreate tie-breaks (like prioritizing high-demand customers) actively degrades recreate quality on small train instances without proven large-n benefits.

## Verdict
refuted (4/4). Do not replace or tweak regret insertion tie-breaks.
