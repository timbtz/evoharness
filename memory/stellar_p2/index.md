# Memory index — stellar_p2 (ConStellaration P2, simple-to-build QI stellarator)

Claim ladder: beat ALM baseline 0.431 -> beat ExLLM 0.505 -> approach leaderboard #1 0.6361.
NOVELTY BAR: submittable = official > 0.6361 AND max-coeff distance >= 1e-3 from EVERY bank seed.
`fm.score()` is BLIND to the novelty penalty. Acceptance keys MUST explicitly subtract it: `fm.bank_dist(b)` returns the exact guard metric.
bank_dist is SCALE-NORMALIZED (2026-07-25): boundaries are divided by their own R0 before the diff — uniform rescaling is physics-null and earns ZERO distance.
LEAK CAVEAT (2026-07-25): ALL pre-fix boundary-level verdicts (val ties, "escapes die at val", bank_dist values) are suspect — see implementation-insights/boundary-report-leak.md before trusting old evidence.
OFFICIAL-FEASIBILITY RAZOR: the vlf->official feasibility gap is ~0.0065 on ESCAPED/truncated boundaries. Target vlf violation <= ~0.003 for escapes.

## Current Best
OFFICIAL best: c0005f of `stellar_p2-s17-78763752` — OFFICIAL 0.6335 for the exported boundary (a real boundary+score pair), but unsubmittable near-copy.
CAVEAT: every other pre-2026-07-25 per-candidate val/private/bank_dist number below mixed boundary identities (boundary-report-leak.md). B4 s103 "c0003 val 0.6054" and B2 "c0006r1 val 0.6280 / officially infeasible" describe LEAKED boundaries, not those candidates' returns.
SUBMITTABLE (official-verified 2026-07-25, runs/dag/escape_readjudication.json): B1 c0003f family — OFFICIAL 0.6330 (viol 0.0071) and 0.6321 (viol 0.00075) at norm dist 2.5e-3; B2 c0006r1 family — OFFICIAL 0.6263 at 1.24e-3. First genuinely novel results; bar 0.6361 not yet beaten (gap 0.0031). Counter-lesson: B3 c0002's escape held vlf viol 0.003 but blew up to 0.277 officially — target vlf viol <= 0.001 and LF-verify.

## successful-patterns
- structural-ball-escape.md — truncation/pivot/dilation escapes; B4 keep=8 truncation yields feasible in-ball anchor (val 0.605)
- multi-seed-triage.md — evaluating top-2 bank seeds in Phase 0
- novelty-aware-micro-polish.md
- margin-aware-polish.md
- momentum-line-search.md
- lf-eval-and-calibration.md
- baseline-alm-tricks.md
- pre-seed-bank/batch-population-and-coordinate-moves.md
- pre-seed-bank/quadrupole-shear-and-dilation.md

## ineffective-approaches
- inner-update-rule-evolution.md — incl. NAE-as-winner refuted x3 in B3
- heuristic-constraint-biasing.md
- coordinate-and-phase-probes.md
- blending-and-recombination.md — incl. B4 cross-basin `_recombine` tie
- momentum-variants.md
- major-radius-and-frobenius-escapes.md — incl. B4 negative R-shift wall-camper
- macro-jump-and-passive-escapes.md — incl. B3 orth-shift wall-camper
- structural-gravity-ascent.md
- lexicographic-and-scalar-margins.md
- dilation-ladders.md
- mode-recovery-homotopy.md — REFUTED incl. negative contraction (B3 c0007/c0011/c0013)
- canvas-cropping.md — REFUTED: cropping lowers eval resolution at every fidelity
- pre-seed-bank/schedule-rewrites.md
- pre-seed-bank/seed-portfolio-and-knob-tuning.md
- pre-seed-bank/ngopt-endgame-slice.md

## implementation-insights
- boundary-report-leak.md — READ FIRST: pre-2026-07-25 boundary verdicts suspect; scale-normalized bank_dist
- vlf-blindness-landmines.md
- lf-gate-selectors.md — gate fixes cannot conjure escape SUPPLY (B3 c0013f)
- numpy-shape-bugs.md
- risky-seed-modifications.md
- decorations-and-invisible-polish.md — B4: wall-clock guards starving Phase 0 supply
- qi-safe-units-bug.md
- refiner-guidance.md — B3: 7/16 refinements were plateau-tying reverts

## performance-analysis
- seed-bank-regime.md
- fidelity-dial.md
- pre-seed-bank/seed-baseline.md

## new-ideas
- b3-untested-proposals.md — margin-filtered interior escapes, QI-preserving re-triage, train-key misalignment constraint
- b2-untested-proposals.md — B2→B3 status record: most items executed/refuted in B3

## Open directions (REBUILT 2026-07-25 after the boundary-report leak fix — pre-fix val evidence is suspect)
1. HARVEST THE MASKED ESCAPES: the archive holds out-of-ball boundaries the leak hid — B1 c0003f family (norm dist 2.5e-3, vlf viol down to 0.00075, shaped ~0.619), B3 c0002/c0008 (0.60-0.62), B2 c0006r1 family (dist 1.24e-3 normalized = structural). Official re-adjudication: runs/dag/escape_readjudication.json. Rebuild escape portfolios from THESE mechanisms and return the margin-best escape — returns are now scored correctly.
2. ESCAPE SELECTION RULE (still sound): one eval_many portfolio (truncation / dilation ladder / pivots), select max shaped subject to vlf viol <= 0.003, log10_qi <= -4.005, bank_dist >= 1.03e-3; NO L-key in escape selection; polish L after margin. Bank-camper fallback.
3. RE-LITIGATE "REFUTED AT VAL" CLAIMS: any pre-fix refutation whose evidence was a val collapse (escape families, homotopy, orth-shift wall-campers) evaluated the WRONG boundary and needs one honest retest before being trusted. Genuinely safe refutations: canvas cropping (resolution follows matrix shape), NAE-as-winner (train-side evidence), gate/acceptance decorations (bit-identical ties were partly leak artifacts, but decorations also never changed the return).
4. BASIN DIVERSITY still untouched: multi-seed novelty-keyed portfolio, dual-basin polish, firewalled NAE/second-seed archive arms. Mystery: s17 c0001 archived shaped 0.6188 at norm dist 0.162 (!) viol 0.0098 — possibly a genuinely distinct basin, uninvestigated.
5. BUDGET-WALL STARVATION: ensure Phase 0 escape portfolios and LF verification gates actually execute (`reserve()` must not exceed remaining wall-clock).
- analyst-stellar_p2-s103-71917443-1.md — in-run Fable analysis @ 10 cands (leak-era run; boundary-level claims suspect)
