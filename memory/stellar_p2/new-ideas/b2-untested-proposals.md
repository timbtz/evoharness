# Untested proposals from branch B2 (analyst reports, s101) — status after B3
Absorbed from the B2 analyst reports (analyst-stellar_p2-s101-20239089-2/-3), proposed against the 0.6161 plateau. Branch B3 (s102) executed several of them; statuses updated so nobody re-tries a now-refuted item.

## Status per proposal
- **Escape-portfolio breadth in Phase 0 — PARTIALLY EXECUTED, weak form failed:** B3 ran `{keep 8,7,6} × {dilate 0,+1.6e-3,+3.2e-3}` batches on seed #3 only (c0004f/c0009f/c0010f); the margin-razor (`feasibility ≤ 0.003 AND log10_qi ≤ −4.0`) stayed EMPTY — no rung was margin-interior. The multi-seed version was only tried as c0006's raw-`fm.score` tournament, which is a selection bug, not a test of the portfolio. Multi-seed portfolio with novelty-aware keys + tolerance-camper repair: STILL UNTESTED (folded into `new-ideas/b3-untested-proposals.md`).
- **Dual-basin / two-anchor interleaved polish — STILL UNTESTED.** Precondition (two feasible anchors) was never met in B3.
- **Calibrated visibility floor with tie≠stall semantics — SUPERSEDED:** B2/B3 visibility variants (grow-on-tie, forced-visible amplitudes) all tied; visibility is necessary but NOT sufficient because the escape is a strict local L-optimum pinned on the aspect/QI walls (`implementation-insights/decorations-and-invisible-polish.md`). Only worth revisiting from a margin-interior start point.
- **QI-margin repair ratchet — REFUTED in B3 (c0002f, c0005):** the inverted-acceptance ratchet cannot move QI because no generator in the stack deepens `log10_qi` from −4.0 (`implementation-insights/qi-safe-units-bug.md`); it paid L for nothing. Corollary now on record: escapes must PRESERVE QI from a QI-interior start (c0007f: interior at bd 3.1e-5 exists), not repair it afterwards.
- **CPU-aware truncation dial — REFUTED in B3 as canvas cropping (c0012f/c0014/c0014f/c0015):** re-truncating/cropping to cut eval cost lowers eval resolution at every fidelity and returns val-dead boundaries. See `ineffective-approaches/canvas-cropping.md`.

## Verdict
mostly consumed — the two survivors (multi-seed novelty-keyed portfolio incl. camper repair; dual-basin polish) plus the new B3 proposals live in `new-ideas/b3-untested-proposals.md`. Keep this page only as the B2→B3 status record.
