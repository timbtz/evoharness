# Proposed-but-never-scored variants (silent deaths)

Several small, mechanism-backed ideas from run cvrp-s3-45465035 died before producing a score (no output or parse_error); they carry zero evidence for or against and should not be treated as refuted.

## How it was tried
- c0003 (no output): claimed SA acceptance defect — `cW <= cS` allegedly always true, making acceptance greedy until stall-reset — plus recreate early-pruning and Or-opt-4. The C edit never emitted a solution, so the claimed bug was neither confirmed nor fixed. First step for a retry: verify whether the defect is real before changing anything.
- c0008 (no output): generalize Or-opt in `try_moves` to segment length 4. Small, plausible change; the implementation silently failed. Note Or-opt-4's cousin (length 1-3 extensions) was score-neutral in C (see ineffective-approaches/c-local-search-modifications.md), so expected value is modest.
- c0010 (parse_error): demand-urgency tie-break in recreate + ruin/cooling tuned for n=200-303 specifically. Never scored. Interesting solely because it targets the large-instance regime where the real headroom lives (see performance-analysis/run-metrics.md) — but the tuning half overlaps the refuted family in sa-acceptance-and-parameter-tuning.md; only the recreate tie-break half is genuinely untested.

## Why it is worth keeping
- These failed for infrastructure reasons (silent crash/hang, unparseable output), not on merit; losing them silently would bias the record toward believing they were tried.

## Verdict
promising-to-neutral, untested. Priority order for a future writer: (1) new-ideas/persistent-c-search-state.md first — strongest mechanism; (2) c0010's demand-urgency recreate tie-break, evaluated for large-n effect; (3) c0003's SA claim, verification only; (4) c0008's Or-opt-4, lowest value. Implement each unbundled.
