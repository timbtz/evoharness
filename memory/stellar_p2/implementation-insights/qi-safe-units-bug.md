# The QI_SAFE Units & Dead Metric Key Bug
Historically, `QI_SAFE = -4.05` was unreachable due to unit mismatches. More recently, `_qi()` reading the non-existent `"log10_qi_residual"` key completely disabled the QI penalty.

## How it was tried
- `stellar_p2-s100-89908732` c0010f analysis: Demanding `log10_qi <= -4.05` in normalized units was impossible. `safe_b` was permanently `None`, silently falling through to return `raw_b` (the QI tolerance camper).
- `stellar_p2-s101-20239089` c0001: The `_qi()` helper fetched `m.get("log10_qi_residual", -5.0)`. Because the metrics dict key is actually `log10_qi`, the penalty was constantly zero. The candidate feasibly escaped the 1e-3 ball (bank_dist 0.00125) but sat exactly on the QI wall (`log10_qi = -3.981`), risking an official-fidelity flip.
- `stellar_p2-s101-20239089` c0001f: Corrected the key to `log10_qi`. This repaired the mechanism but immediately caused a 0.038 train score regression because the `QI_LAM=1.0` penalty was too strong relative to the L-gradient.
- `stellar_p2-s101-20239089` c0003: Fixed the key and set `QI_SAFE = -3.98`. With the boundary sitting exactly at `log10_qi = -3.981`, `max(0, -3.981 - (-3.98))` was exactly 0. The correction was inert.
- `stellar_p2-s102-48117936` c0002 / c0002f: Attempted a feasibility-pad (`_qisteer`) before structural escapes, but measured its success entirely on generic `feasibility` decreases. It failed to actually move `log10_qi`, so the escape started and stayed on the QI wall (log10_qi -3.988). Val collapsed to -0.28.
- `stellar_p2-s102-48117936` c0003: Implemented a massive custom `_constr_viol` that tried to re-derive max normalized violation from raw aspect, QI, and mirror ratios. It critically failed to match `fm.score`'s clean `-max_violation` signal, allowing a catastrophic boundary (log10_qi -3.357) to pass train filters.
- `stellar_p2-s102-48117936` c0005: Introduced a dedicated Phase-0 QI-margin ratchet (`_phase_shift` and `_scale_row`) aimed at reaching `QI_TARGET = -4.05`. The `safe_b` swap permanently failed to find a child passing this hard gate, burning budget and falling back to a penalized in-ball camper (train 0.5622). 
- `stellar_p2-s102-48117936` c0009f, c0010f: Attempted to populate a margin-razor (`_note_escape`) demanding `feasibility <= 0.003 AND log10_qi <= -4.0`. As escapes starting at `-4.0` cannot move QI via Gaussian mutation or truncation, the razor tracker stayed permanently empty.

## Why it failed
A mismatch between the raw log10 space of the diagnostic metrics and the normalized violation space of the fitness function, or simply guessing dictionary keys. Gates and margins must be stated in the metric's own normalized units, or they silently disable safety paths. Custom violation aggregators misread raw values as normalized violations, destroying the search gradient. Furthermore, demanding `log10_qi <= -4.05` from an escape starting exactly at `-4.0` via low-mode phase shifts is practically impossible; hard base-swaps on unreachable bounds simply burn budget.

## Verdict
recurring pitfall — Sanity check your metric keys. If a `safe_b` or return path is unexpectedly `None`, verify that your margin bounds are physically achievable. When fixing dead penalty logic, ensure your lambda weights are gently tuned so they do not dominate the L-signal. Do not re-derive max violation locally; rely on `fm.score > 0`.
