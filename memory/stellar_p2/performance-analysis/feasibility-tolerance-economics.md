# Feasibility-tolerance economics — where our score above 0.6361 actually comes from

Written 2026-07-27 after the campaign's first above-bar submittables. Read this
before claiming a leaderboard win.

## The official rule (verified in the pinned package, not inferred)

`constellaration==0.2.6`, `problems.SimpleToBuildQIStellarator`:

```python
_DEFAULT_RELATIVE_TOLERANCE = 0.01
def is_feasible(m):  return np.all(normalized_violations(m) <= 0.01)
def compute_feasibility(m): return max(max(normalized_violations(m)), 0.0)
```

Five normalized constraints — aspect ratio, edge iota/nfp, log10 QI, edge mirror
ratio, max elongation — each `(value - bound) / |bound|`. A design is FEASIBLE
while every violation stays under **1 %** of its own bound. Score is
`L / 20` (minimum normalized magnetic gradient scale length), independent of how
much tolerance you spend.

So the tolerance is a free resource, and L rises as you push aspect ratio into it.

## Where our winners sit

| boundary | official P2 | feasibility | % of tolerance | aspect ratio |
|---|---|---|---|---|
| c0045 (`3-dac057eed0c1b2a8`) | 0.6400 | 0.00931 | **93 %** | 10.093 |
| c0088r1 (`3-024648d4008d3302`) | 0.6398 | 0.00959 | 96 % | 10.096 |
| c0105 (`3-7b691c0fd6211a95`) | 0.6352 | 0.00410 | 41 % | 10.041 |
| davidkh (bank #0, leaderboard #1) | 0.6361 | 0.00075 | **7.5 %** | 10.0075 |
| lhhhhappy (bank #3, our winner's ancestor) | 0.6257 | 0.0 | 0 % | ≤ 10 |

The max violator is always aspect ratio; the other four constraints are slack.
`compute_feasibility` is fidelity-invariant (identical to 1e-15 at vlf and at
official) because aspect ratio is pure boundary geometry — no VMEC solve in it.

## The exchange rate: ~0.92 score per unit feasibility

Three independent estimates agree:

1. **Paired official evals, same lineage.** c0105 → c0045: Δscore 0.0048 over
   Δfeas 0.0052 ⇒ slope **0.92**.
2. **Archive regression**, 307 unique feasible out-of-ball boundaries:
   `p2 = 0.6193 + 0.912·feas` (R² 0.17 — low because the set mixes basins, but the
   slope is the same).
3. **Direct low-margin evidence.** s17 c0005f: official 0.6335 at feas 0.00144.
   Best archived out-of-ball boundary under feas 0.005: 0.6345.

## Consequence — state this whenever the 0.6400 is quoted

Normalizing our winner to davidkh's margin (0.00075):

    0.6400 − 0.92 × (0.00931 − 0.00075) ≈ 0.632

**At equal tolerance use we are ~0.004 BELOW the leaderboard leader, not 0.004
above him.** Our entire margin over 0.6361 is bought with tolerance he left
unspent. It is a legal submission — `is_feasible` is the official predicate and we
pass it — but it is *tolerance camping*, one of the three gaming modes flagged at
project start (with resolution gaming and HV padding).

## Risk of the thin margin

Not numerical: the evaluator is deterministic and the metric is exact geometry, so
0.00931 reproduces bit-for-bit under the same package version (confirmed —
re-verified from the archive in a clean container: 0.6400 / 0.0093).

The risk is **version and definition drift**. At 93 % of tolerance, anything that
moves aspect ratio or its bound by >0.7 % of the bound — a package update, a
changed `_DEFAULT_RELATIVE_TOLERANCE`, a different VMEC resolution in the scoring
service — flips us to score 0.0. davidkh at 7.5 % has ~13× that headroom. Precedent
exists in this project: B2 c0006r1 landed at feasibility 0.0106 and scored exactly
0.0.

## What would be a real win

A boundary that beats 0.6361 at feasibility ≲ 0.002. That means raising L by
structure, not by aspect ratio — the open direction remains a genuinely different
basin (see new-ideas/low-nfp-nae.md). Suggested working rule for future runs: track
`score at feas ≤ 0.002` as a second, honesty-preserving leaderboard alongside raw
official score.
