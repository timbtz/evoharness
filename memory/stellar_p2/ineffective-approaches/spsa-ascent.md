# Iterative Optimizer Ascent (SPSA / nevergrad / coordinate descent)
Any ask/tell or sequential-step optimizer starves the 240s/720s eval budget and times out —
the evals (~8-12s nfp=3, 12-27s nfp=4) are far too slow for iterative loops.
## How it was tried
- `s105-72881323` c0022, c0025, c0030 (timeout 720s): iterative nevergrad / CMA-style
  ask-tell ascent loops on the contracted basins. Never completed a generation.
- `s105` c0047 (timeout 720s): adaptive sequential coordinate-descent (18-cell coarse +
  8-cell half-step refinement). Exceeded the wall.
- `s105` c0063 (train 0.6211, in gradient-and-rotation-escapes.md): a single batched
  random-subspace direction probe — the ONE budget-safe variant — still regressed to floor.
- Prior branches: SPSA random-direction line search proposed repeatedly (analyst @36/@60) as
  a "2-eval gradient" fix, but never landed (injections permission-blocked); the underlying
  premise (that the (0.5,0.7) contraction axis is suboptimal) is refuted — see below.
## Why it failed
Score ~8-27s/eval makes per-coordinate or per-generation sequential estimation infeasible in
budget (budget-discipline.md). The one axis that matters (aspect-relief depth) is already found
by the cheap batched R/Z-split grid; c0073/c0075 confirmed (0.5,0.7) is a strict local optimum,
so a data-driven direction cannot beat it locally anyway.
## Verdict
refuted — do NOT run iterative/ask-tell optimizers. All ascent must be a single batched
`eval_many` design (grid or ±perturbation), never a sequential loop.
