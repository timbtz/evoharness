# Inter-route 2-opt / 2-opt* in the Python overlay

Adding inter-route tail-swap (2-opt*) or inter-route 2-opt moves to the Python overlay was tried 18 times in run cvrp-s3-45465035 (~31% of its candidates) and NEVER beat the incumbent; the family is exhausted. Note: c0027 was itself already a KNN-filtered Python overlay (not a C replacement) — retrying it "more carefully as an overlay" retests the same dead idea.

## How it was tried
All variants kept the C kernel byte-identical (verified for c0027, c0038, c0046, c0048, c0051); every one added inter-route 2-opt/2-opt* to `_py_or_opt1`/`_py_local_search`.
- Heavy/interleaved variants — actively harmful: c0027 (-0.384; also ran the overlay as a pre-pass on the CW start), c0036 (-0.380), c0038 (-0.375, accepted only via val tie), c0046 (-0.129, accepted), c0055 (-0.199). All show the disruption signature: per-instance time 4.65s->4.8-4.85s and X-n101 falling from optimum 27591 to the 27825 attractor (+0.85%). The Python work starves the C LNS.
- Careful/final-polish variants — exactly score-neutral: c0039 (-0.0918), c0041/c0042/c0045/c0047/c0052/c0053 (all -0.0906 with byte-identical costs 27591/14971/55690), c0043 (-0.093), c0044 (-0.0918, accepted), c0048 (-0.0954, overlay on every candidate), c0049 (-0.110), c0051 (-0.0906, accepted). Identical costs mean the 2-opt* move found ZERO improving moves — the C kernel's `tails()` already leaves its output 2-opt*-optimal within the KNN neighborhood.
- Buggy variant: c0050 (IndexError in the new move code; fixed by c0051, still no gain).
- Sub-variants that changed nothing: two-customer swaps (c0040), randomized scan order (c0053), 1.0s slices (c0040), strict prefix-sum capacity checks (c0044).

## Why it failed
- Redundant neighborhood: the C kernel already performs inter-route 2-opt* (tail exchange) inside `try_moves`; its output has no KNN-local 2-opt* improving moves left. The overlay can only re-verify optimality (neutral) or burn time (harmful).
- Attempts c0038-c0055 each assumed "c0027 failed because it was a replacement/unfiltered — a careful overlay will differ". False premise: c0027 was already a KNN-filtered Python overlay; its failure cause was time-starvation, which the "careful" retries only partially avoided and never turned into a gain.

## Verdict
exhausted/refuted. Never add inter-route 2-opt or 2-opt* to the Python overlay again in any form. If cross-route restructuring is the target, it must come from a different mechanism (e.g. changing what the C LNS explores), not from post-hoc Python moves the C output is already optimal against.
