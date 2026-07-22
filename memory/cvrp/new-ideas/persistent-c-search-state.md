# Persistent C search state: one long LNS call instead of 1.5s slices

The Python outer loop re-packs `best` into `cseq` before every ~1.5s `lns` slice, resetting the C kernel's SA/exploration state each call; running ONE long C call for the full remaining budget (Python polish only at the end) has a concrete mechanism behind it and has never been evaluated.

## How it was tried
- c0054 (run cvrp-s3-45465035): full diagnosis — outer loop discards C working state (`cand_f`) by restarting each slice from `best`, so the SA trajectory resets every 1.5s; proposed a single long call + final-polish-only overlay. Died to parse_error, never scored.
- c0037: same direction (drop the interleaved overlay, give C the full budget, one final comprehensive polish). parse_error, never scored.
- c0009: related (module-level compile caching, fewer/longer LNS calls, capacity-splitting validity fallback). parse_error, never scored.

## Why it is worth trying
- Mechanism checks out against the code: the temperature schedule cools over elapsed fraction, and each slice restarts from `best`, so late slices burn budget re-converging instead of exploring.
- Consistent with the time-starvation evidence in ineffective-approaches/inter-route-2opt-star-overlay.md: interleaved Python work between slices costs ~0.2s and measurably degrades X-n101. Removing the slicing removes that whole failure surface.
- Risk profile is Python-only orchestration (C kernel untouched), unlike the refuted C rewrites.

## Verdict
promising, untested (0 evaluations in any run). Implement minimally: one `lns` call with the full remaining budget minus ~0.3s, then the existing Or-opt 1-3 polish (successful-patterns/python-oropt-polish-overlay.md), then `_valid` check. Do not bundle with any other change, so the effect is attributable.
