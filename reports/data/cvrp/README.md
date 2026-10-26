# Reviewed CVRP evidence

This directory contains frozen historical programs and fresh regression results.
It does not contain an implemented VRPTW solver or new LLM-generated candidates.
Interpret the numbers using the [routing assessment](../../cvrp.md).

- `candidates/ancestor.py`: shared evolved ancestor `c0014`.
- `candidates/score_only.py`: score-only endpoint `c0033r1`.
- `candidates/memory.py`: memory endpoint `c0035`.
- [candidate-provenance.json](candidate-provenance.json): source run IDs, source
  ledger hashes, program hashes and selected historical run-end fields.
- [recheck-20261026.json](recheck-20261026.json): seed and both endpoints, three
  repeats each on validation and historical private; 18 evaluations.
- [val30-probe-20261026.json](val30-probe-20261026.json): seed, ancestor, and both
  endpoints at 30 seconds per validation instance; one evaluation each. This is
  an exploratory budget probe, not a repeated ranking.
- [ancestor-recheck-20261026.json](ancestor-recheck-20261026.json): three private
  repeats each for the seed and shared ancestor, in a separate follow-up batch.
- [visual-provenance.json](visual-provenance.json): built-in image-generation prompts
  and targeted edit for the conceptual diagrams. Figures contain no measured plots.

Historical programs are unmodified copies of this project's ignored local run
artifacts. Treat them as evaluator inputs; do not import them on the host or edit
them as active source. Edit the seed under `src/evoharness/tasks/cvrp/seed/` instead.
Algorithm attribution is preserved in the source and repository `NOTICE.md`.

Each benchmark JSON records its exact source hashes and environment. The initial
18-evaluation recheck preceded the additional fail-closed container-policy fix;
its Docker image remained available throughout. Subsequent runs use strict
container policy. No scoring or solver-code change was made between these runs.
The Docker base, compiler and NumPy are not yet pinned for rebuilding; the image
ID identifies the environment actually used.

Reproduce the first experiment from the repository root:

```bash
uv run evoharness fetch cvrp
uv run evoharness cvrp benchmark \
  --candidate score_only=reports/data/cvrp/candidates/score_only.py \
  --candidate memory=reports/data/cvrp/candidates/memory.py \
  --splits val private --repeats 3 --output .local/runs/cvrp-recheck.json
```

The current seed is included automatically. Use a new output path for each run.
Reruns need not reproduce exact costs: these solvers stop by wall time. Three
timing repeats are a diagnostic, not a substitute for independent solver seeds
and independent evolution campaigns.

Additional experiments:

```bash
uv run evoharness cvrp benchmark \
  --candidate ancestor=reports/data/cvrp/candidates/ancestor.py \
  --candidate score_only=reports/data/cvrp/candidates/score_only.py \
  --candidate memory=reports/data/cvrp/candidates/memory.py \
  --splits val30 --repeats 1 --output .local/runs/cvrp-val30-probe.json
uv run evoharness cvrp benchmark \
  --candidate ancestor=reports/data/cvrp/candidates/ancestor.py \
  --splits private --repeats 3 --output .local/runs/cvrp-ancestor-recheck.json
```
