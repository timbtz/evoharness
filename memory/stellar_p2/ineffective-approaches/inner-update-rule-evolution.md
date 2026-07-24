# Evolving the inner numeric update rule
Re-deriving gradient-descent/CMA-es-style update mathematics inside the candidate is the wrong altitude — verdict from the pre-build domain research.

## Why it fails
- A hand-rolled adaptive update rule is strictly dominated by CMA-es / Nevergrad NGOpt, which are importable in-sandbox. 72 evals cannot even estimate a covariance matrix in 40 D.
- The paper already shows scipy trust-constr and COBYLA FAIL to reach feasibility on these problems (both remained infeasible) — do not rediscover this with budget.

## What to do instead
Evolve the SCAFFOLD: seeding, preconditioning/scaling, constraint shaping (log-transforms, penalty schedules), mode continuation, restart/portfolio logic, crash handling, batch strategy (fm.eval_many), when to call which library optimizer. Inner solvers stay library calls.

## Confirmed in-run (stellar_p2-s42-27282945 score_only smoke AND stellar_p2-s7-30186401)
Smoke run: 5 of 14 generations re-rolled a hand-written CMA-ES-style optimizer (c0008, c0009, c0010, c0011, c0012 — no anti-re-roll digest in score_only mode). Four crashed on numpy shape bugs; the one that ran (c0009, -0.692) scored WORSE than the plain greedy incumbent (-0.491). Zero wins.
Run stellar_p2-s7-30186401: c0001, c0002, c0003, c0006 all attempted custom ALM proxies, gradient consensus vectors, or signed rank-based direction estimation directly on constraint violations. All regressed heavily (-0.64 to -0.67) because an 80-D problem cannot fit enough axes in 72 evals, and custom metrics misalign with `fm.score`'s clean `-max_violation` signal.
stellar_p2-s11-38566380 c0021 (-0.501): Injected an ALM-style mirror-ratio finite-difference gradient descent during Phase 1. The bidirectional probes cost 2 evals per axis, halving axis coverage and scoring worse than the standard greedy descent.
stellar_p2-s11-38566380 c0024 (-inf): Swapped pool unpacking order, crashing. Attempted to replace Phase 1 with a bracketing line-search per elite.

## Verdict
refuted (external + in-run evidence) — flag any candidate that reimplements a numeric optimizer core as a repeat. If covariance adaptation is wanted, import nevergrad and give it a budget slice; never hand-roll.
