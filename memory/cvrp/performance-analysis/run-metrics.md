# Where the score lives

Train instances 1-2 (X-n101, X-n110) sit at their optima in every healthy candidate; the entire residual train gap is X-n125 stuck at cost 55690 (+0.27%), and the real objective — private n=200-303 — is only weakly visible from train.

## How it was tried (observed anatomy, run cvrp-s3-45465035)
- Healthy-candidate fingerprint: costs exactly (27591, 14971, 55690) = gaps (0.0, 0.0, 0.27), train -0.0906. Dozens of distinct programs converge to these attractors, so train ties are the norm, not evidence of "no change".
- X-n125 attractor: 55690 vs best-known 55539. Broken past only twice: c0035 -> 55688 (kept; private 0.581%) and c0017 -> 55551 (a semantic no-op — luck; see score-noise-and-gate.md).
- X-n101 disruption signature: when interleaved Python work or parameter changes starve/deflect the C LNS, X-n101 lands on a second attractor at 27825 (+0.85%): c0027, c0028, c0036, c0038. Per-instance seconds also rise 4.65 -> 4.8-4.85. A 0.85 on instance 1 + 4.85s is a time-starvation diagnosis, not a move-quality one.
- Time budget: solver uses ~4.65s of the 5s train budget; the winning overlay costs ~0.01s; every harmful overlay costs ~0.2s.
- Private (6s, X-n200/251/303): best c0035 = 0.581% mean (0.51/0.54/0.69), improving on earlier runs' 0.70-0.80%. Run history: cvrp-s1-40028152 train -0.0 / private 0.796%; cvrp-s1-45827564 train -0.0894 / private 0.704%; cvrp-s3-51556548 train -0.0876 / private 0.748%; cvrp-s3-45465035 train -0.0894 / private 0.581%. Largest instance is consistently worst; train ~-0.09 is compatible with private anywhere in 0.58-0.75% — train does not predict private rank.

## Why it matters
- Train is saturated: instances 1-2 give zero gradient, instance 3 is one integer-cost attractor away from optimal. Almost all remaining headroom is on n>=200, which train cannot see — c0035's noise-level train gain was a real 0.12-0.22pt private gain.

## Verdict
promising as targeting guidance: judge candidates by whether they plausibly help n>=200 (scaling behavior, per-instance seconds, val), not by train deltas within the -0.09 cluster. Watch for the 27825 signature as an immediate red flag.
