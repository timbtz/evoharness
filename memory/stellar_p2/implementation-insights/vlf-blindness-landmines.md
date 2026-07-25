# Train fidelity is BLIND to small perturbations — landmines at val
A boundary can score bit-identically to its parent at train fidelity yet be catastrophically infeasible at val fidelity. Your train score is NOT proof your change is safe.

## Evidence
- **stellar_p2-s17-78763752 c0001 (gen 1):** Mode-grafted a convex recombination onto the 0.6188 seed. Train score: 0.61881739... (BIT-IDENTICAL to parent). Val feasibility 0.9656 — the tighter equilibrium exposes a QI blow-up (log10qi ~ -0.13). Score zero at any official evaluation.
- **stellar_p2-s100-89908732 c0005f:** Escaped the 1e-3 novelty ball via a directed drift jump. Train score looked excellent (0.5886), but the identical vlf->lf gap mechanism doomed it: val landed at -0.0113.
- **stellar_p2-s101-20239089 c0001:** Escaped the ball via mode-truncation. Train score looked excellent (0.6140), but the boundary sat dangerously close to the QI constraint wall (`log10_qi = -3.981`). An unpenalized QI metric let this slip through train, highlighting how structural escapes instantly hit hidden constraint walls.
- **stellar_p2-s102-48117936 c0002 / c0003:** Applied structural escapes (dilation/homotopy) that successfully exited the ball (bank_dist ~ 0.0028). Although they maintained a generic feasibility margin ( squeaking under 0.01), they sat strictly on the QI wall (`log10_qi = -3.988` / `-3.357`). Train passed, but val collapsed to -0.28.
- **stellar_p2-s102-48117936 c0012f/c0014/c0014f/c0015 (val ≈ −0.011 to −0.014):** cropping `r_cos`/`z_sin` to minimal mode support lowered VMEC resolution at EVERY fidelity — train and even the LF gate certified boundaries val rejects. See `ineffective-approaches/canvas-cropping.md`.
- **stellar_p2-s102-48117936 c0010:** Attempted to seed from NAE independently. The NAE seed was infeasible at train scale, so it fell back to an in-ball bank seed drift. The boundary landed at `bank_dist = 0.000386` and `log10_qi = -3.994`—invisible at train, but collapsing to val -0.0115 at validation.

## Why it happens
Both fidelities match VMEC resolution to the boundary's modes; they differ in force tolerance (1e-9 vs 1e-13) and QI machinery depth. Small/high-modes perturbations can be invisible at loose tolerance and disastrous at tight. Furthermore, bank seeds sit exactly on constraint walls; crossing the wall is invisible to `fm.score` but fatal to `fm.eval`.

## What to do
- A train score IDENTICAL to the parent's after a nonzero change = the simulator never saw your change: treat it as UNTESTED, not safe.
- Keep grafts/blends at meaningful amplitude on low-order modes (visible at train), not epsilon dust on high modes.
- Feasibility margin at train must be kept < ~0.007 (not 0.0098) so fidelity-gap slack exists; val tolerance is effectively tighter.
- A generic feasibility margin is NOT ENOUGH. Explicitly enforce `log10_qi < -4.0` on all escapes, or higher-fidelity QI machinery will trigger catastrophic val collapses.
