# Blind mutations and untargeted tweaks (pre-tracing runs)

Across the pre-tracing runs (cvrp-s1-40028152, cvrp-s1-45827564, cvrp-s3-51556548), 120+ candidates tried blind genetic-style mutations, small C-kernel refactors, and LNS knob tweaks without a mechanism-level hypothesis; none beat the -0.0906 baseline cluster.

## How it was tried
- 120+ candidates over three runs: random code mutations, cosmetic C refactors, and parameter jitter (SA temperature, ruin sizes, random restarts). Best outcomes only *approached* the baseline (-0.0906); many crashed (see implementation-insights/c-failure-modes.md, pre-tracing bullet) or regressed heavily.
- The same knob families were re-tried with named hypotheses in run cvrp-s3-45465035 and still failed — see sa-acceptance-and-parameter-tuning.md (c0028, c0029, c0030, c0033) and c-local-search-modifications.md (c0023, c0031).

## Why it failed
- Untargeted changes either land inside the eval noise band (see performance-analysis/score-noise-and-gate.md) or break co-tuned kernel invariants; there is no gradient to climb by mutation at the current plateau.

## Verdict
exhausted. Do not generate mutation-style candidates. Every proposed change needs a concrete mechanism and an expected per-instance effect; otherwise it re-runs this 120-candidate experiment.
