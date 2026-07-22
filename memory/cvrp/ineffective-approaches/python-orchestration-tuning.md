# Python orchestration and time-budget tuning

Reshuffling the Python driver (slice lengths, validity-check frequency, where the overlay runs in the loop) was neutral-to-negative in every evaluated attempt.

## How it was tried
- c0015 (-0.107): continuous small-burst C calls, removed per-loop `_valid` checks — slightly worse than baseline.
- c0040 (-0.096): reduced LNS slice 1.5s->1.0s to give the overlay more end-time — neutral (bundled with 2-opt* moves, see inter-route-2opt-star-overlay.md).
- c0055 (-0.199): removed "redundant" Python checks to lengthen C time — regression (also bundled 2-opt*).
- c0048 (-0.095, accepted): ran the overlay on every LNS candidate instead of only best — no gain, more Python time.
- Related proposals that never scored (c0009, c0037, c0054 — parse_error) are tracked in new-ideas/persistent-c-search-state.md, not here.

## Why it failed
- Evaluated variants only moved time between components whose marginal value is below eval noise (see performance-analysis/score-noise-and-gate.md); none changed what the search does, only when.

## Verdict
exhausted for micro-tuning (slice sizes, check frequency, overlay placement). The one orchestration change with a real mechanism — a single long C call preserving SA state — is untested; see new-ideas/persistent-c-search-state.md before attempting anything in this space.
