# g3 run: Structural Perturbations, Spectral Shifts, and NAE Probes
In run `g3`, attempts to escape the 0.5843 local floor via orthogonal structural perturbations, warm-ladder harmonic bumps, or nfp=2 NAE basin pivots consistently failed to beat the incumbent floor, regressing or tying perfectly.

## How it was tried
- `g3` c0011 (REJ, train 0.5805): Added a non-destructive warm-sequence of orthogonal harmonic bumps targeting the n-axis. Tied the parent floor.
- `g3` c0015 (ACC, train 0.5843): Added a three-stage composed contraction sweep with negative stage-3 curvature to explore non-monotone m-profiles. Safely tied the parent floor.
- `g3` c0016 (REJ, train 0.5644): Replaced pipeline with a batched `nfp=2` `seed_nae` pivot probe to target the `L ∝ A/Nfp` category jump. Regressed as dynamically generated NAE seeds lack baseline L.
- `g3` c0017 (ACC, train 0.5843): Added a finer cr/cz-split contraction grid targeting asymmetric R/Z decoupling. Tied the parent floor exactly.
- `g3` c0018 (ACC, train 0.5843): Added a toroidal-helicity spectral ladder (rotating spectral weight from the dominant m=1 harmonic into adjacent n-offsets). Tied the parent floor exactly.

## Why it failed
Writers predicted that shifting toroidal weight, adding negative curvature stages, or pivoting to nfp=2 NAE basins would probe unexplored QI structures or achieve a category jump. The code applied these mechanisms. However, all local structural tricks (redistribution, negative stage curvature) merely fall back to the inferior B3 floor, while nfp=2 NAE seeds fundamentally lack the baseline objective_L to be competitive. The structural physics deficit in the hardcoded base matrix strictly limits any local search.

## Verdict
exhausted — Stop proposing local spectral tricks or nfp=2 NAE pivots in isolated states. They cannot overcome the baseline structural physics deficit.
