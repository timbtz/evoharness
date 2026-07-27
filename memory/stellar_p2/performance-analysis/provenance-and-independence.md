# Provenance — what the harness found on its own vs. what it inherited

Written 2026-07-27, from the full run history. This page exists so nobody quotes
"we beat the ConStellaration P2 leaderboard" without the qualifier.

## Without the public seed bank, the search never became feasible

Runs before `seed_bank.json` existed (nothing from any public submission anywhere
in the loop — NAE / ellipse starts only):

| run | budget | best shaped | official P2 | max violation |
|---|---|---|---|---|
| s42 (smoke) | $0.29 | −0.491 | **0.0** | 0.519 |
| s7 | $1.53, 32 cands | −0.4911 | **0.0** | 0.517 |
| s11 (Fable-refiner campaign) | $3, 12 h, 21W/3T/4L refiner | −0.4253 | **0.0** | 0.425 |

The official tolerance is 0.01. These runs ended **42–52× over it**. Not "a bit
below davidkh" — not on the board at all. The QI wall was never crossed at
candidate-eval scale, which is exactly why the seed bank was approved (2026-07-24).

`B6-nae-independent`, the branch designed to retry an own-basin start with
everything learned since, has **never been run** — it is still `pending` in
runs/dag/state.json. Until it runs we have no evidence that the harness can reach
a feasible QI boundary unaided.

## With the seed bank, every scoring result descends from someone's submission

| run / branch | best | official | bank_dist | nearest public seed |
|---|---|---|---|---|
| s17 | c0005f | 0.6335 | ~3.3e-4 | davidkh (near-copy; export guard refused it) |
| B1 s100 | c0008f | 0.6286 | 3.65e-4 | in-ball |
| B2 s101 | c0006r1 | 0.0 | 1.25e-3 | escaped, but feasibility 0.0106 → 0 |
| B3 s102 | c0003f | 0.6193 | 4.6e-4 | in-ball |
| B4 s103 | c0105 | 0.6352 | 2.56e-3 | davidkh |
| B4 s105 | c0088r1 | 0.6398 | 2.61e-3 | davidkh |
| B5 s105 | c0045 | **0.6400** | 2.64e-3 | davidkh |

## How far the 0.6400 winner actually is from davidkh's boundary

Measured on the exported submission vs `seed_bank.json` #0 (davidkh, official
0.6361), same (8,15) `nfp=3` mode matrix:

- max single-coefficient difference / R0 = **2.64e-3** (the export guard's metric;
  guard threshold 1e-3)
- ‖Δ‖ / ‖seed‖ over all Fourier coefficients = **0.47 %**
- cosine similarity = **0.999989**

The wiki previously called the winner "the hardcoded B3-lhhhhappy3 escape". By
distance that is wrong: it is nearer to **davidkh** (2.64e-3) than to lhhhhappy
(3.25e-3), which is consistent with the lineage — the B1/B3 "escapes" were
themselves polished descendants of the s17 elites, and those were davidkh
near-copies.

## Honest summary

The optimizer did not independently discover a competitive stellarator boundary.
It took the #1 public submission, perturbed it by ~0.5 %, and spent feasibility
tolerance that its author left unspent (see feasibility-tolerance-economics.md) to
land 0.0039 above him. Normalized to equal tolerance use it is ~0.004 *below* him.

What the harness genuinely demonstrated: it can find the local L/aspect-ratio
Pareto direction from a good starting boundary and walk it precisely — a real but
much narrower claim than "beat the leaderboard".

The 1e-3 export guard turns out to be too weak to carry the "novel" adjective on
its own. 2.6e-3 clears it while cosine similarity to the source is 0.999989. Either
raise the bar or stop calling guard-clearing results novel.
