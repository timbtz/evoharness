# Where the score lives

Train instances 1-2 (X-n101, X-n110) sit at their optima in every healthy candidate; the entire residual train gap is X-n125 stuck at cost 55690 (+0.27%), and the real objective — private n=200-303 — is only weakly visible from train.

## How it was tried (observed anatomy)
- Healthy-candidate fingerprint: costs exactly (27591, 14971, 55690) = gaps (0.0, 0.0, 0.27), train -0.0906. Dozens of distinct programs converge to these attractors, so train ties are the norm.
- X-n125 attractor: 55690 vs best-known 55539. Broken past twice: c0035 -> 55688 (kept; private 0.581%) and c0017 -> 55551 (a semantic no-op — luck; see score-noise-and-gate.md). c0040 also hit 55688.
- X-n101 disruption signature: when interleaved Python work or multi-start orchestration starves the C LNS, X-n101 lands on a second attractor at 27825 (+0.85%): c0027, c0028, c0036, c0038, c0046, c0048, c0049. Per-instance seconds rise 4.65 -> 4.8-4.85.
- Time budget: solver uses ~4.65s of the 5s train budget; the winning overlay costs ~0.01s; every harmful overlay costs ~0.2s.
- Private (6s, X-n200/251/303): best c0035 = 0.581% mean (0.51/0.54/0.69), improving on earlier runs' 0.70-0.80%. Train ~-0.09 is compatible with private anywhere in 0.58-0.75% — train does not predict private rank.
- Infrastructure failure (unrelated to algorithm): c0097, c0099, c0101, c0102 failed with `docker: pull access denied for evoharness-cvrp-eval`. Ignore these as eval harness issues, not candidate defects.

## Why it matters
- Train is saturated: instances 1-2 give zero gradient, instance 3 is one integer-cost attractor away from optimal. Almost all remaining headroom is on n>=200, which train cannot see.

## Verdict
promising as targeting guidance: judge candidates by whether they plausibly help n>=200 (scaling behavior, per-instance seconds, val), not by train deltas within the -0.09 cluster. Watch for the 27825 signature as an immediate red flag.
