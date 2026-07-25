# Variable-shape matrix bugs kill candidates
Boundary matrices have DIFFERENT shapes per seed and after inflation ((2,3) at mp=1, (3,5) after one pad, (8,15) or (11,21) for bank seeds). Any blending, crossover, or stacking assumes one global shape and will crash if not projected or padded to a common canvas first.

## How it was tried
- stellar_p2-s42-27282945 (smoke): 4/15 candidates died writing population/ES code that stacked unequal shapes.
- stellar_p2-s11-38566380 c0016 (`-inf`): Phase-1 mean-crossover `_recombine_mean` failed broadcast `(2,3) vs (3,5)` because pool mixed mp=1 and mp=2 seeds.
- stellar_p2-s11-38566380 c0016f (-0.600): Fixed c0016 by projecting the partner's coarse modes into pool[0]'s canvas via `_project_like` before blending.
- stellar_p2-s11-38566380 c0024f (`-inf`): Phase-1 bracketing line-search unpacked `pool[ei]` with the assumption `(boundary, score)` instead of `(score, boundary)`, passing a `float` to `_mutate` instead of a matrix dictionary.
- stellar_p2-s11-38566380 c0026 (`-inf`): Tried to implement `_interpolate` across worst/best pool members but crashed during matrix unpacking/operations.
- `stellar_p2-s100-89908732` c0018 (`-inf`): Crashed attempting `_max_coeff_dist` between two bank seeds of shapes (8,15) and (11,21) because `_convex_midpoint` used right-padding `np.pad(..., (0, tc - shape))` which misaligns columns and breaks broadcasts.
- `stellar_p2-s100-89908732` c0016f (val -0.687): center-padded canvas was MIS-centered, silently zeroing every bank distance — the novelty penalty became a constant with no gradient and the run regressed hard. A padding bug can be score-poisoning without crashing.
- `stellar_p2-s100-89908732` c0007 (`-inf`): dual-basin variant called `_mats(None)` when bank triage yielded no base — guard every optional base/seed path against None before matrix ops.
- `stellar_p2-s101-20239089` c0008 (`-inf`): Triggered `_mats(None)` crash by passing `best_anchor = None` into `_escape_pivot` on the NAE fallback path. Fixed in c0008f by guarding `if b is None: return None`.
- `stellar_p2-s102-48117936` c0001 (`-inf`): Massive structural rewrite of the solve loop referenced `_viol` and `_key` which were never defined. Additionally, `_extract` tried to return `score` before it was assigned locally, crashing instantly on run.
- `stellar_p2-s102-48117936` c0009 (`-inf`): Rearchitected the key function to store metrics (`m`) directly on the boundary object via `c._m = m`. Boundaries are plain `dict`s, so setting attributes crashed instantly with `AttributeError`.

## Why it failed
The search pool maintains diverse mode dimensions. Arithmetic across two different members (`rc0 + w*(rc1 - rc0)`) assumes matching row/col counts. Developers frequently forget the exact tuple ordering of the internal pool structure, leading to TypeErrors during unpacking. For bank distance, padding must be CENTER-ALIGNED (shifting `n=0` off-center registers modes incorrectly and crashes). Fallback paths (e.g., NAE seeds) must guard against `None` bases. Large restructures frequently reference helper functions that were never actually implemented in the code block. In Python, trying to set custom attributes (like `obj._m = m`) on standard `dict` instances will throw `AttributeError`.

## Verdict
recurring pitfall — Always validate `shape` before arithmetic. Pad-to-common-canvas or use a projection slice (`rc1[:rows, ntor_t-dn:ntor_t+dn+1]`) to align the source onto the target's shape safely. Double-check tuple unpacking order. Guard all optional base/seed inlets against `None`. Never attach attributes dynamically to plain dicts; use a side-cache or pass metrics as extra arguments.
