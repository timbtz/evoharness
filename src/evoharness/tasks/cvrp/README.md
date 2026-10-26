# CVRP optimization: source map and workflow

This is the implemented **capacity-constrained, distance-minimizing CVRP** task.
It has no time windows or lexicographic fleet objective. The proposed VRPTW
extension and evidence assessment live in [the routing report](../../../../reports/cvrp.md).

## Where to work

| Responsibility | Source | Contract |
|---|---|---|
| Dataset, split budgets, host scoring | [task.py](task.py) | Fixed references; return negative mean percentage gap |
| Instance download | [fetch.py](fetch.py) | Download every active split; runtime data stays outside source |
| Editable Python search policy | [seed/solver.py](seed/solver.py) | Construction, C calls, best-so-far routes |
| Editable native search engine | [seed/kernel.c](seed/kernel.c) | Granular moves, ruin/recreate, annealing |
| Candidate assembly | [seed.py](seed.py) | Embed C into one self-contained Python module |
| Model-facing contract and knowledge | [resources/](resources/) | Solver signature, constraints, curated mechanisms |
| Evaluation environment | [Dockerfile.eval](Dockerfile.eval) | Python, NumPy and GCC; one CPU and 1 GiB at runtime |
| Frozen-program experiment runner | [analysis/cvrp_benchmark.py](../../analysis/cvrp_benchmark.py) | Sequential repeats, exact costs, hashes and failure records |
| General program evolution | [engine/loop.py](../../engine/loop.py), [strategies/](../../strategies/) | Writer, acceptance, parent selection, reviewed memory |
| Historical configuration | [configs/cvrp.json](../../../../configs/cvrp.json) | Existing defaults, preserved for compatibility |
| Reviewed evidence | [reports/data/cvrp/](../../../../reports/data/cvrp/) | Candidate snapshots and new regression measurements |

The reusable engine stays outside the task package. Keep future time-window
parsing, feasibility and fleet scoring in a separate `vrptw` task; do not silently
change the meaning of historical CVRP scores. No VRPTW task is registered today.

## Solver interface and scoring

```python
def solve(coords, dist, demand, capacity, deadline, compile_c) -> list[list[int]]:
    ...
```

Node 0 is the depot; return only customer indices. Distances use CVRPLib
`floor(euclidean_distance + 0.5)`. The candidate can compile C inside its worker.
Host validation checks coverage and capacity, then re-costs visited edges using
trusted coordinates. The number of routes is **not** the optimization objective.

| Split | Instances | Seconds per instance | Use |
|---|---|---:|---|
| `train` | X-n101, X-n110, X-n125, X-n186 | 5 | Candidate search |
| `val` | X-n153, X-n228, X-n242 | 5 | Optional holdout gate |
| `private` | X-n200, X-n251, X-n303 | 6 | Historical run-end audit; now exposed regression data |
| `val30` | Same instances as `val` | 30 | Longer-budget diagnostic, not new instances |
| `public` | Small A/B/P instances | 3 | Progress display; largely saturated |

Full instance names, reference costs and split membership are in `task.py`.
CVRPLib names count the depot, so X-n303 has 302 customers.

## Run an offline comparison

From the repository root:

```bash
uv sync --group dev
uv run evoharness fetch cvrp
docker build -t evoharness-cvrp-eval \
  -f src/evoharness/tasks/cvrp/Dockerfile.eval src/evoharness/tasks/cvrp
uv run evoharness cvrp benchmark --splits val private --repeats 3 \
  --candidate score_only=reports/data/cvrp/candidates/score_only.py \
  --candidate memory=reports/data/cvrp/candidates/memory.py \
  --output .local/runs/cvrp-recheck.json
```

The assembled seed is always included. Omit `--candidate` for a seed-only run;
use `--dry-run` to inspect cost without building or evaluating. `--splits val30`
requests the longer-budget probe. Existing output files cannot be overwritten.
The equivalent installed module is `python -m evoharness.analysis.cvrp_benchmark`.

Keep other CPU experiments idle during measurement. Three programs × three
repeats on `val private` allocate 297 instance seconds, including distance setup
and compilation, plus container/import overhead. Compilation is cached across
instances within one evaluation and consumes part of the first instance's budget.
Repeats share each program's RNG policy and measure timing variation; they are
not independent evolution or solver seeds. Errors remain explicit and suppress
headline medians rather than becoming zero-gap successes.

## Run new LLM evolution

The original config defaults to greedy, score-only, train-gated search. A model
run needs the configured provider credentials; inspect it first:

```bash
uv run evoharness run --config configs/cvrp.json --execution require_container \
  --set switches.gate=holdout --set switches.search=staged \
  --max-calls 8 --max-usd 0.50 --max-seconds 900 --dry-run
```

Remove `--dry-run` to execute. Provider budgets are stop guards between operations,
not hard limits on an in-flight API call or evaluation. `val30` probes add substantial
evaluation time even to short evolution runs.

For a feedback ablation, use a separate `EVOHARNESS_HOME` per arm and seed, fetch
the same instances into each home, and initialize memory from the same frozen
snapshot (or empty state). Change only `switches.feedback` between `score_only`
and `memory`. Match actual model resources **and** evaluator CPU consumption;
the old equal-dollar comparison did not do this. Freeze a new final test set
before research begins.

## Runtime state and trust boundary

Downloaded data, live memory, and new ledgers are under `.local/`, or the directory
specified by `EVOHARNESS_HOME`. They are ignored by Git. Reviewed candidate copies
and reports belong under `reports/data/cvrp/`; do not edit the historical snapshots.

The host checks solution quality and total elapsed wall time. Per-instance timing
shares Python globals with candidate code and remains best-effort. Docker removes
network access and bounds CPU/memory; the explicit fallback is only a diagnostic
convenience, not equivalent isolation. The seed still has quadratic storage and
construction, and some inner loops do not poll deadlines. Use small benchmarks
until those mechanisms are redesigned.

## Verify changes

```bash
uv run pytest -q tests/tasks/test_cvrp.py tests/analysis/test_cvrp_benchmark.py
```

The downloaded-data integration test executes timed seed solves. Do not run it
alongside timing experiments. Without downloaded instances, that integration
test skips; route-validation and experiment-accounting tests still run.
