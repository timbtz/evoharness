# Macro-Jumps and Passive Drift Escapes
Attempts to escape the `bank_dist < 1e-3` novelty ball via blind random jumps, directed drift extrapolation, or passive acceptance key drift. Refuted for being too slow or fatally breaking vlf->lf feasibility. Merged from `novelty-escape-variants.md`.

## How it was tried
- **Blind random jumps (stellar_p2-s100-89908732 c0001, c0003):** Fixed-magnitude (~2.5e-3) multi-mode random displacement. Landed at feasibility margins that triggered catastrophic val collapse (val -0.010).
- **Extrapolated directed drift (c0004f, c0005f):** Extrapolated the accumulated accepted-delta polish direction to jump out of the ball. Escaped successfully (c0005f reached 0.5886), but crossed into the vlf->lf feasibility gap (negative val).
- **Minimal-L∞ single-coefficient ladder (c0006f, val -0.0107):** Moved exactly ONE large low-order coefficient by 1.3e-3. Cheapest possible exit still landed in the vlf->lf gap.
- **Isotropic minor-radius dilation (c0007f, val -1.08):** Similarity transform of the polished seed. Broke feasibility catastrophically at both fidelities.
- **Passive drift decorations (c0009 / c0009f, both 0.5679):** Accumulated-delta extrapolation fired only after 4+ accepts. Both inert — landed exactly on the no-op plateau.
- **Directed orthogonal multi-coefficient shift (stellar_p2-s102-48117936 c0008, train 0.6188 / val −0.27):** coarse-to-fine low-mode shift directed orthogonal to the nearest bank seed, applied to the FULL bank seed. Escaped (bd 1.5e-3) but preserved all high-mode structure and camped every wall at once (mirror 0.00977, elong 5.02, aspect 10.034, log10_qi −4.0003); train loved it, val collapsed. c0008f reverted (tie). Untested variant — the same shift applied to an already-truncated escape — filed in `new-ideas/b3-untested-proposals.md`.
- **NAE-basin fallback (stellar_p2-s102-48117936 c0010):** Attempted to fall back to independent `fm.seed_nae` boundaries to escape the ball entirely. The NAE seed failed at train scale, and the fallback returned a non-novel bank camper (val -0.0115).

## Why it failed
The base bank seed sits extremely close to the hard constraint limits (aspect ratio ~10.01, bound 10.0). Any blind or directed macro-jump (amplitude > 1e-3) pushes the boundary directly through the constraint wall. While macro-escapes successfully shed the novelty penalty, they simultaneously break the loose-tolerance (vlf) physics, dropping val to -0.01. Passive drift over the entire B1 branch only grew bank_dist from 2.05e-4 to 3.65e-4 — too slow. Fallbacks to NAE seeds cannot cross the QI wall at candidate scale.

## Verdict
refuted for macro-jumps and passive drift — Structured ball escapes (see `successful-patterns/structural-ball-escape.md`) are the only way to leave the 1e-3 ball feasibly.
