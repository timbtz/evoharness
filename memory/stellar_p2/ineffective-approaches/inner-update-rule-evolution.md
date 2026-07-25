# Evolving the inner numeric update rule
Re-deriving gradient-descent/CMA-es-style update mathematics inside the candidate is the wrong altitude — verdict from the pre-build domain research.

## Why it fails
- A hand-rolled adaptive update rule is strictly dominated by CMA-es / Nevergrad NGOpt, which are importable in-sandbox. 72 evals cannot even estimate a covariance matrix in 40 D.
- The paper already shows scipy trust-constr and COBYLA FAIL to reach feasibility on these problems (both remained infeasible) — do not rediscover this with budget.

## What to do instead
Evolve the SCAFFOLD: seeding, preconditioning/scaling, constraint shaping (log-transforms, penalty schedules), mode continuation, restart/portfolio logic, crash handling, batch strategy (fm.eval_many), when to call which library optimizer. Inner solvers stay library calls.

## Confirmed in-run (stellar_p2-s42-27282945 score_only smoke AND stellar_p2-s7-30186401)
Smoke run: 5 of 14 generations re-rolled a hand-written CMA-es-style optimizer; all scored worse than plain greedy.
Run stellar_p2-s7-30186401: c0001-c0006 attempted custom ALM proxies and gradient consensus vectors. All regressed heavily because custom metrics misalign with `fm.score`'s clean `-max_violation` signal.
stellar_p2-s11-38566380 c0021 (-0.501): Injected an ALM-style mirror-ratio finite-difference gradient descent. The bidirectional probes cost 2 evals per axis, halving axis coverage.
`stellar_p2-s100-89908732` c0017 (-0.5236): Switched back to a pure-NAE seed start bypassing banks, hoping to find an independent feasible basin. Refuted as nobody crosses the QI wall from NAE seeds at candidate eval scale.
`stellar_p2-s100-89908732` c0019 (0.5688): Attempted orthogonal-complement projection subtracting a seed-vector from mutations to push into a physically unconstrained nullspace. Ended up returning a verbatim bank seed (inert).
`stellar_p2-s101-20239089` c0012 (`-inf` timeout): low-fidelity finite-difference gradient ascent over 20-30 coefficients blew the 720 s CPU limit (40-60+ slow lf evals). c0017 (`-inf`): hand-rolled anisotropic CMA-ES over top Fourier coefficients crashed outright.
`stellar_p2-s102-48117936` c0001 (`-inf`): Ripped out the solve loop to write a custom dual-anchor interleaved margin-aware polish with separate ratchets. It crashed instantly with `NameError: _viol` and undefined `_key` methods.
`stellar_p2-s102-48117936` c0004 (-0.598): Attempted a purely independent NAE-basin growth via low-fidelity-filtered feasible-descent search from multiple NAE seeds. Again, nobody crosses the QI wall from NAE seeds at candidate scale.
`stellar_p2-s102-48117936` c0010 (val -0.0115): Replaced the deterministic escape with an NAE-basin seed + polish loop fallback. The NAE seed proved deeply infeasible, so the loop drifted the raw bank seed back into the ball.
`stellar_p2-s102-48117936` c0009 (`-inf`): Replaced the deterministic Phase-0 with a randomized-shift restructure; crashed before any eval.
`stellar_p2-s102-48117936` c0012 (train 0.566, val 0.575): Ripped out the Phase-0 deterministic portfolio for a streaming multi-mechanism generator. The deterministic Phase-0 fallback returned an in-ball camper.
`stellar_p2-s102-48117936` c0016/c0016f (both val −0.0115): Injected a Phase-0 NAE `seed_hop` entry replacing a keep-7 probe; the NAE entry failed at train scale and the deterministic fallback returned the same in-ball camper — third NAE-as-winner refutation of the branch (after c0004, c0010).

## Verdict
refuted (external + in-run evidence) — flag any candidate that reimplements a numeric optimizer core, attempts a full solve loop rewrite, or abandons the bank basin as a repeat.
