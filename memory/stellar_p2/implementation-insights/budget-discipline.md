# Budget Discipline: Batched LF Verification & Eval Cost
Low-fidelity (LF) verification of multiple candidates causes timeouts when evals are slow (~8-12s each on mpol=8 B1 boundaries). 
## How it was tried
- `stellar_p2-s105-72881323` c0001-c0003: Timed out (720s) on Bank #4 (12-27s/eval).
- `stellar_p2-s105-72881323` c0014: Tested a 25-candidate R/Z-split × n-shift grid. Brushed the 240s wall-clock deadline due to accumulating 8-12s evals.
- `stellar_p2-s105-72881323` c0047 (timeout): An adaptive coordinate-descent search with an 18-cell coarse grid + 8-cell half-step refinement exceeded the 720s limit.
- `stellar_p2-s105-72881323` c0060 (ERR): Attempted a 20-eval mode-grafting portfolio. Timed out at 720s and was killed with a `SyntaxError`.
- `stellar_p2-s105-72881323` c0071 (ERR): Attempted to restore the tri-basin portfolio but introduced ragged B4 matrices into a batched `eval_many` call. Crashed with `ValueError: setting an array element with a sequence` (detected shape `(9,)`).
- `stellar_p2-s105-72881323` c0046 (ACC, 217s wall-clock): Barely passed with a ~24 candidate portfolio + 2 LF verifies.
- `stellar_p2-s105-26196944` c0001 (timeout 720s): Attempted a 5-eval sweep on an authentic hardcoded B3-lhhhhappy3 nfp=3 escape boundary. The boundary's high spectral resolution (mpol=10/ntor=5) caused VMEC evals to cost 60s each, starving the budget.
- `stellar_p2-s105-26196944` c0006 (ACC, 24s wall-clock): Executed a 14-candidate multi-seed bank sweep safely in one batched `eval_many`. Bank seeds evaluate extremely fast (~1-2s each) compared to hardcoded authentic boundaries.
- `stellar_p2-s105-26196944` c0012 (ERR, timeout 720s): Attempted a 7-eval batched sweep of selective subdominant-mode damping (m>=4). High eval costs on mpol=7 boundaries annihilated the budget.
## Why it worked / failed
The eval cost scales with boundary complexity and search dimension. Evaluating >15 candidates sequentially annihilates the budget. Iterative coordinate descent and large portfolios are structurally incompatible with the 240s/720s limits. Furthermore, padding nfp=4 (B4) matrices with ragged lengths and passing them as native lists to `numpy.array()` causes hard crashes in the evaluator. High-resolution (mpol>=7) hardcoded boundaries also trigger massive eval times even if only evaluated a handful of times.
## Verdict
recurring pitfall — ALWAYS cap portfolio sizes to fit within a single batched `eval_many` train phase + a batched top-K LF phase. Keep portfolios ≤15 evals. Pad matrices cleanly to uniform dimensions before passing to `eval_many`. Avoid mpol>=7 hardcoded boundaries unless mapped to a fast nfp=3 representation.
