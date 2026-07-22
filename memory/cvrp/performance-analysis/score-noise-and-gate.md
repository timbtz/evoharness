# Eval noise and holdout-gate misfires

Single-evaluation train/val deltas smaller than ~0.1 are noise — proven by a semantically identical program swinging both scores — and the holdout gate therefore both rejected the run's best train score and accepted a -0.37 regression.

## How it was tried (evidence)
- c0017 vs parent c0016: the diff is variable renames, comments, and inlining `okc` — verified semantically identical C, zero Python diff. Yet train went -0.0906 -> -0.0072 (best train of the ENTIRE run, X-n125 at 55551) and val went -0.0236 -> -0.1602. Sole possible cause: run-to-run variance (anytime, wall-clock-coupled SA trajectory). So single-run noise is at least ~0.09 train / ~0.14 val.
- Gate rejected c0017 (best train ever) on its unlucky val draw; gate accepted c0038 (train -0.3745, a genuine interleaving regression per run-metrics.md) on a val tie of -0.0236.
- Val (X-n153) is coarse: the value -0.0236 recurs across most accepted candidates (c0002, c0013, c0016, c0024, c0035, c0038, c0044, c0046, c0048, c0051), so "val not worse" is nearly always satisfiable and discriminates little.
- Mitigating structure: parent selection stayed on the best-train accepted candidate (c0035 parented all of stretch 3), so bad acceptances did not redirect the search — they only wasted acceptances.

## Why it matters
- Writers repeatedly narrated noise as signal ("the previous attempt regressed because...") for deltas inside the noise band, generating spurious lessons.
- Any claimed improvement below ~0.1 train needs repeated evaluation or private/val corroboration before being believed; c0035 is trusted because of its 0.581% private result, not its train edge.

## Verdict
promising as a decision rule: treat |delta train| < 0.1 as a tie; distrust single val readings; re-evaluate would-be new bests before promoting; interpret an accepted-but-worse-train candidate as a gate artifact, not progress.
