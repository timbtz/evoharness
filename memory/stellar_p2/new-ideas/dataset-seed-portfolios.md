# Dataset-mined seed portfolios
The public HF dataset (proxima-fusion/constellaration, ~158k configs with metrics) contains QI-optimized boundaries far closer to feasibility than any generate_nae output.

## Status
- Untested here. Candidates cannot download it (sandbox is offline), but the HARNESS side could curate a few dozen good boundaries into a seed file the task exposes via fm (e.g. fm.seed_bank(i)) if we decide to allow it.
- Rules: no bulk download; provenance recorded; it changes the claim ("from public seeds" vs "from scratch") so it must be reported honestly if used.
- ExLLM (arXiv 2502.12845, P2=0.505) shows LLM-guided proposals over a good starting population go far without any inner optimizer.

## Verdict
promising but a POLICY decision, not a candidate trick — decide at harness level before any run uses it.

## Update 2026-07-24
DECIDED and implemented as fm.seed_bank (12 public P2 submissions, not the 158k dataset). See performance-analysis/seed-bank-regime.md — that page now governs.
