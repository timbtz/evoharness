# Untested / partly-tried coordinated-ascent mechanisms (absorbed from analyst notes)
Absorbs the 8 in-run Opus analyst rounds of `s105-72881323`. IMPORTANT CONTEXT: every analyst
round shared one premise — "the official ceiling of every basin this branch touches is ~0.633,
below the 0.6361 bar, so pivot away from contraction." **run_end FALSIFIED this**: the branch's own
contracted-B3 floor (c0075f) readjudicated to official **0.6398, submittable** — the analysts never
readjudicated it and extrapolated a false 0.633 cap from prior-branch escapes. Also, all 8 analyst
INJECT writes were permission-blocked, so zero analyst candidates ran. Treat these as LOW priority.

## Ideas proposed
- **vlf-gated SPSA constrained ascent** (@36): margin-tiered selection (feas≤0.001 preferred) + a
  2-eval SPSA direction. UNTESTED as written — but iterative SPSA is refuted for budget reasons
  (see ineffective-approaches/spsa-ascent.md); only a single-batched form could ever run.
- **Low-D model-based trust-region quadratic DFO** (@72/@96): fit a diagonal quadratic of
  shaped_score across ~6 dominant shaping modes in one batched ±design, step to the constrained
  optimum. PARTLY TRIED and FAILED — writers executed exactly this at s105 c0085f/c0087; it hit the
  QI wall and fell back (see ineffective-approaches/surrogate-and-nae-escapes.md).
- **Anisotropic exp (m,n) escape off the top bank seed** (@60): TRIED and REFUTED — c0055f/c0057/c0060f
  collapsed to the floor (ineffective-approaches/top-bank-anisotropic-escape.md).

## Verdict
mostly refuted/exhausted. The one genuinely open thread is a SINGLE-BATCHED margin-tiered selection
key — but the far higher-value direction is the opposite of what the analysts urged: keep depth-
contracting the nfp=3 B3/B1 escape (it reached official 0.6398 and was NOT capped). See
successful-patterns/b1-rebase-and-escapes.md.
