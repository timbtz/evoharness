# Train fidelity is BLIND to small perturbations — landmines at val
A boundary can score bit-identically to its parent at train fidelity yet be
catastrophically infeasible at val fidelity. Your train score is NOT proof
your change is safe.

## Evidence (stellar_p2-s17-78763752, c0001, gen 1)
- c0001 mode-grafted a convex recombination onto the 0.6188 seed. Train score:
  0.6188173970956956 — BIT-IDENTICAL to the parent (vlf's loose force
  tolerance, 1e-9, converged without ever feeling the perturbation).
- val (low fidelity, force tolerance 1e-13): feasibility 0.9656 — the tighter
  equilibrium exposes a QI blow-up (log10qi ~ -0.13 vs -4 required). Gate
  rejected it. Score zero at any official evaluation.

## Why it happens
Both fidelities match VMEC resolution to the boundary's modes; they differ in
force tolerance (1e-9 vs 1e-13) and QI machinery depth. Small/high-mode
perturbations can be invisible at loose tolerance and disastrous at tight.

## What to do
- A train score IDENTICAL to the parent's after a nonzero change = the
  simulator never saw your change: treat it as UNTESTED, not safe.
- Keep grafts/blends at meaningful amplitude on low-order modes (visible at
  train), not epsilon dust on high modes.
- Feasibility margin at train must be kept < ~0.007 (not 0.0098) so
  fidelity-gap slack exists; val tolerance is effectively tighter.
