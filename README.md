# EvoHarness

A small, open-config LLM code-evolution harness: an LLM (z.ai GLM) rewrites a scored
Python function generation after generation; five independent design axes are pluggable
two-option switches, so all 2⁵ optimizer variants assemble from one config — from code
or from the web UI.

| Axis | A | B |
|---|---|---|
| feedback | `score_only` | `reflections` (ReEvo-style short/long-term) |
| gate | `public_only` (train-greedy) | `holdout` (val split) |
| search | `greedy` (1+1) | `islands` (4 islands, ring migration) |
| knowledge | `off` | `wiki_fs` (task wiki, inject or tool mode) |
| roles | `single_strong` | `split_roles` (cheap writer) |

Tasks: `binpacking` (FunSearch online bin packing), `circles` (26 circles in unit
square), `tsp` (tour construction + fixed 2-opt polish). Each task has four splits —
train (selection), val (holdout gate only), public (reporting), private (end-of-run
only, never in prompts) — so the generalization gap is measured, not assumed.
Objectives: `quality`, `quality_per_dollar`, `time_capped`.

## Quickstart

```bash
uv sync
echo 'ZAI_API_KEY=...' > .env             # z.ai key; base URL defaults to the coding endpoint
uv run python tasks/tsp/fetch.py          # one-time: download TSPLIB instances
uv run pytest tests/ -q                   # golden tests (LLM mocked, ~2 min)

# Web UI: config panel + A/B race view with live score-vs-$ chart
uv run uvicorn api.server:app --host 0.0.0.0 --port 8000

# One run from code
uv run python -c "from core.config import Config; from core.loop import run; \
  print(run(Config(task='binpacking', budget={'max_usd': 1.0, 'max_calls': 60, 'max_seconds': 900})))"

# Screening: all 32 switch combos at tiny budget, then the main-effects report
uv run python experiments/screening.py --task binpacking --seeds 1 2 3 --max-usd 0.3
uv run python experiments/report.py runs/screening/binpacking --csv runs/screening.csv
```

Every run appends events to `runs/<id>/ledger.jsonl` (append-only, no DB): config,
every LLM call with cost, every candidate with scores/novelty/acceptance, periodic
public-split reports with a render payload, and a final summary incl. private score
and generalization gap. The API's SSE endpoint and `experiments/report.py` are both
just readers of this file. Budgets (`max_usd` / `max_calls` / `max_seconds`) are hard
stops. Candidate code runs sandboxed: `unshare -rn` (no network) + rlimits + kill on
timeout, falling back to a socket-stub preamble where user namespaces are unavailable.

## Docker (optional)

```bash
docker compose up --build   # serves the UI on :8000
```

`DECISIONS.md` logs deviations from the briefing; `NOTICE.md` carries attributions
(FunSearch, ReEvo, EoH, OpenEvolve, PyVRP patterns). `.md/Plan.md` in the parent
directory is the original briefing.
