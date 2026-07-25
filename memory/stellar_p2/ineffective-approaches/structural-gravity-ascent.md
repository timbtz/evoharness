# Structural & Gradient-Guided Macro-Escapes in Polish
Attempts to push the boundary out of the 1e-3 ball using gravity-controlled ascent (scaled gradient steps), freeze removals, or hard novelty floors during the polish phase. All fail by falling victim to the vlf-blindness death spiral or acting as dead decorations.

## How it was tried
- **Pivot Freeze Removal & Hard Novelty Floor (`stellar_p2-s101-20239089` c0009f):** Removed the `_freeze_pivot` (which dragged trials back to the bank seed) and held the escape with a hard `bank_dist >= NOV_FLOOR` acceptance floor. Tied exactly at 0.6161.
- **Gravity-Controlled Ascent (GA) (`stellar_p2-s101-20239089` c0010, c0011):** At `stall==8`, extracted the direction of the last accepted step, scaled it 4x normal sigma, and injected a batched pair to escape the ball. Tied exactly at 0.6161. 
- **Hard Novelty Floor on Escape Polish (`stellar_p2-s102-48117936` c0011f):** Applied a hard floor on `best_b` rejecting any polish step that re-entered the 1e-3 ball. Without any visible outward moves available, the polish loop simply rejected everything and tied the plateau exactly.

## Why it failed
c0009f was conceptually sound (freeze removal clears the coefficients), but it missed the true root cause: polish mutations are epsilon dust. Removing the freeze doesn't matter if the simulator still can't see the base mutation. The GA move (c0010) was dead code: it gated on `best_dist < NOV_D`, but the Phase-0 escape starts already at `best_dist == NOV_D` (1.25e-3), making the strictly-less-than check permanently false. Forcing a hard novelty floor on escapes simply starves the polish loop of valid backward gradients while offering no outward alternatives.

## Verdict
refuted — Structural macro-escapes inside the polish loop cannot defeat the deterministic plateau if base mutations are invisible. Stop decorating the polish loop with macro-escapes and fix the vlf-blindness first.
