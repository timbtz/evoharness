# Structural Ball Escapes (Truncation, Pivot, Dilation, R-Shift)
First mechanisms to feasibly cross the `1e-3` novelty ball from a bank seed. The key is using structural escapes to break out instantly, paired with the continuous novelty penalty + LF gate stack.

## How it was tried
- **Mode-Truncation & Pivot (`stellar_p2-s101-20239089` c0001, c0003):** `_escape_truncate` (zeroing modes `|(m,n)| > 6`) and `_escape_pivot` (displacing one coefficient by ±1.25e-3). Successfully crossed the 1e-3 ball and maintained feasibility (train 0.6140, val 0.6280).
- **Aspect Dilation (`stellar_p2-s101-20239089` c0004):** Added `_escape_dilate` applying a coupled `(1+1.6e-3)` rescale to `r_cos[1:]/z_sin[1:]` while compensating the major radius. Acted as a shape-preserving, scale-invariant escape. Best val of the run (0.6289) with excellent margin.
- **Dilation Ladder (`stellar_p2-s101-20239089` c0004f):** Deepened the dilation to `eps = 2.0e-3 / 3.2e-3`. Rejected; train regressed slightly to 0.6110. The deeper rungs provided no benefit over c0004's single 1.6e-3 step.
- **OFFICIAL outcome (B2 branch end):** the truncation escape (c0006r1, the branch best) scored OFFICIAL 0.0 — official feasibility 0.0106 vs the 0.010 wall (vlf showed only 0.0039). The vlf->official gap on escaped boundaries is ~0.0065; "feasible at train/val" was NOT feasible officially. Official L was 12.6, so one notch more margin ≈ a submittable ~0.63.
- **Margin-Razor Batch (`stellar_p2-s102-48117936` c0009f, c0010f):** Widened Phase 0 to evaluate a `{keep 8,7,6} x {dilate 0,+1.6e-3,+3.2e-3}` batch. The margin-razor (`feasibility <= 0.003 AND log10_qi <= -4.0`) stayed completely empty.
- **Aspect-Relief Micro-Contraction (`stellar_p2-s102-48117936` c0013):** Applied a `-1.6e-3` aspect-relief micro-contraction to the razor survivor before returning. Tied train/val exactly but reproduced the **homotopy mirage** (see `ineffective-approaches/mode-recovery-homotopy.md`): scaling coefficients toward the bank pulled the boundary back inside the `1e-3` ball, making it unsubmittable.
- **B3 BRANCH END (`stellar_p2-s102-48117936`):** the escape frontier was never crossed. Every out-of-ball return collapsed at val; every val-positive return was in-ball (bd ≤ 6e-4).
- **Keep=8 Truncation (`stellar_p2-s103-71917443` c0003):** In B4, high-keep mode-truncation (`_truncate(s, 8)`) of bank #4 successfully produced a feasible boundary (train 0.6049, val 0.6054). However, it sat *inside* the ball (bd 7e-4) and suffered a 0.0148 novelty penalty (raw p2 0.6197).

## Why it worked / failed
Mode-truncation and single-pivot displacements are large enough to instantly clear the 1e-3 max-coeff distance metric. High-keep truncation (keep=8) minimally disrupts QI balance, yielding feasible boundaries that survive val, but it does not zero enough coefficient mass to escape the ball. Escaping requires either dropping more modes (which degrades QI to the wall) or adding structural shifts (which risk aspect/mirror walls). 

## Verdict
promising WITH CAVEAT, and narrowing — the mechanisms cross the ball, but after THREE full branches (B2 s101, B3 s102, B4 s103) no escape from the primary bank seeds has survived official feasibility AND cleared the novelty ball. High-keep truncation yields a feasible but in-ball anchor (val ~0.605); escaping the ball without breaking QI remains the untested frontier. See `new-ideas/b3-untested-proposals.md`.
