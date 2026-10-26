"""Re-evaluate frozen CVRP programs sequentially, without any model calls.

Example::

    uv run python -m evoharness.analysis.cvrp_benchmark \
        --candidate evolved=path/to/best.py --splits val private --repeats 3 \
        --output .local/runs/cvrp-recheck.json

The assembled seed is always included. Repeats measure wall-clock variability;
they are not independent evolution seeds. The old private split has already been
examined and is a regression set, not a fresh held-out test. Do not run competing
CPU jobs during a benchmark. Container construction occurs before measurement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
from typing import Callable

import numpy as np

from evoharness.engine.candidate import EvalResult
from evoharness.execution.policy import execution_policy
from evoharness.paths import REPO_ROOT
from evoharness.tasks.cvrp import task as cvrp


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(value: object) -> str:
    return _sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30,
                                cwd=REPO_ROOT)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def load_candidates(specifications: list[str]) -> list[dict]:
    """Read code as text; only the task sandbox may execute it."""
    candidates = [{"name": "seed", "source": "assembled seed/kernel.c + seed/solver.py",
                   "code": cvrp.TASK.seed_code()}]
    names = {"seed"}
    for spec in specifications:
        name, sep, source = spec.partition("=")
        if not sep or not name or not source or name in names:
            raise ValueError("each --candidate must be a unique NAME=PATH (seed is reserved)")
        path = Path(source).expanduser()
        candidates.append({"name": name, "source": str(path), "code": path.read_text()})
        names.add(name)
    for candidate in candidates:
        candidate["sha256"] = _sha256(candidate["code"].encode())
    return candidates


def evaluation_record(result: EvalResult, names: list[str]) -> dict:
    """Keep failures explicit and derive full-precision gaps from verified costs."""
    record = {"status": "error", "error": result.error, "score": None,
              "mean_gap_pct": None, "host_seconds": _finite_seconds(result.seconds),
              "instances": []}
    if result.error or not math.isfinite(result.score):
        record["error"] = result.error or "non-finite score"
        return record
    costs = result.metrics.get("costs")
    if not isinstance(costs, list) or len(costs) != len(names) or any(
        type(cost) is not int or cost < 0 for cost in costs
    ):
        record["error"] = "missing or malformed host-verified costs"
        return record
    seconds = result.metrics.get("seconds", [])
    if not isinstance(seconds, list) or len(seconds) != len(names):
        seconds = [None] * len(names)
    for name, cost, elapsed in zip(names, costs, seconds):
        bks = cvrp._BKS[name]
        record["instances"].append({
            "name": name, "cost": cost, "bks": bks,
            "gap_pct": 100.0 * (cost - bks) / bks,
            "reported_solver_seconds": _finite_seconds(elapsed),
        })
    gap = statistics.mean(item["gap_pct"] for item in record["instances"])
    if not math.isclose(-result.score, gap, rel_tol=1e-9, abs_tol=1e-9):
        record["error"] = "score does not match host-verified costs"
        record["instances"] = []
        return record
    record.update(status="ok", error=None, score=result.score, mean_gap_pct=gap)
    return record


def _finite_seconds(value: object) -> float | None:
    return (float(value) if type(value) in (int, float) and math.isfinite(value)
            and value >= 0 else None)


def summarize(records: list[dict], candidates: list[dict], splits: list[str],
              repeats: int) -> list[dict]:
    summaries = []
    for split in splits:
        for candidate in candidates:
            rows = [r for r in records if r["split"] == split
                    and r["candidate"] == candidate["name"]]
            values = [r["mean_gap_pct"] for r in rows if r["status"] == "ok"]
            complete = len(rows) == repeats and len(values) == repeats
            summaries.append({
                "candidate": candidate["name"], "split": split,
                "expected_repeats": repeats, "attempts": len(rows),
                "successful_repeats": len(values), "failed_repeats": len(rows) - len(values),
                "complete": complete,
                # Withhold the headline if a repeat failed: no survivor-only winner.
                "median_gap_pct": statistics.median(values) if complete else None,
                "min_gap_pct": min(values) if complete else None,
                "max_gap_pct": max(values) if complete else None,
            })
    return summaries


def _save(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_benchmark(candidates: list[dict], splits: list[str], repeats: int,
                  output: Path, metadata: dict,
                  evaluate: Callable[[str, str], EvalResult] | None = None) -> dict:
    """Run one evaluator at a time and checkpoint every result to disk."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark evidence: {output}")
    if repeats < 1 or not candidates or not splits or len(splits) != len(set(splits)):
        raise ValueError("positive repeats, candidates, and distinct splits are required")
    if any(split not in cvrp._SPLITS for split in splits):
        raise ValueError("unknown CVRP split")
    if len({candidate["name"] for candidate in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")
    evaluate = evaluate or cvrp.TASK.evaluate
    config = {
        "splits": {split: {"instances": cvrp._SPLITS[split],
                           "seconds_per_instance": cvrp._BUDGET[split]}
                   for split in splits},
        "bks": {name: cvrp._BKS[name] for split in splits for name in cvrp._SPLITS[split]},
        "repeats": repeats, "order": "split, repeat, cyclically rotated candidates",
        "script_timeout_slack_seconds": cvrp._SLACK,
        "host_wall_grace_seconds": cvrp._WALL_GRACE,
    }
    report = {
        "schema_version": 1, "task": "cvrp", "experiment": "frozen-program re-evaluation",
        "started_at": _now(), "finished_at": None, "status": "running",
        "model_calls": 0, "config": config, "config_sha256": _canonical_hash(config),
        "candidates": [{k: v for k, v in candidate.items() if k != "code"}
                       for candidate in candidates],
        "environment": metadata, "records": [], "summary": [],
        "limitations": [
            "Repeats share solver code and its internal RNG policy; these are timing reruns.",
            "Previously inspected private instances are a regression set, not fresh holdout evidence.",
            "Short-budget CVRP gaps do not measure VRPTW or XLTW challenge readiness.",
            "Candidate-reported solver seconds are informational; the host bounds total elapsed time.",
        ],
    }
    _save(output, report)
    for split in splits:
        for repeat in range(repeats):
            offset = repeat % len(candidates)
            order = candidates[offset:] + candidates[:offset]
            for candidate in order:
                started = _now()
                try:
                    result = evaluate(candidate["code"], split)
                except Exception as exc:
                    result = EvalResult(float("-inf"), error=f"{type(exc).__name__}: {exc}")
                record = evaluation_record(result, cvrp._SPLITS[split])
                record.update(candidate=candidate["name"], split=split, repeat=repeat + 1,
                              started_at=started, finished_at=_now(),
                              evaluation_index=len(report["records"]) + 1)
                report["records"].append(record)
                report["summary"] = summarize(report["records"], candidates, splits, repeats)
                _save(output, report)
                metric = (f"gap={record['mean_gap_pct']:.4f}%" if record["status"] == "ok"
                          else f"ERROR: {record['error']}")
                print(f"{split} repeat={repeat + 1} {candidate['name']}: {metric}", flush=True)
    report["status"] = ("complete" if all(r["status"] == "ok" for r in report["records"])
                        else "completed_with_errors")
    report["finished_at"] = _now()
    _save(output, report)
    return report


def environment_metadata(splits: list[str], allow_fallback: bool) -> dict:
    """Preflight datasets and container before any timed evaluation begins."""
    files = {}
    for name in dict.fromkeys(name for split in splits for name in cvrp._SPLITS[split]):
        cvrp._inst(name)
        path = cvrp._INST_DIR / f"{name}.vrp"
        files[path.name] = _sha256(path.read_bytes())
    ready = cvrp.docker_image_ready(cvrp._IMAGE, cvrp._DOCKERFILE, str(cvrp._DIR))
    if not ready and not allow_fallback:
        raise RuntimeError("CVRP container unavailable; build Dockerfile.eval and check Docker. "
                           "--allow-fallback explicitly permits a weaker diagnostic run.")
    source_paths = [cvrp._DIR / "task.py", cvrp._DIR / "seed.py",
                    cvrp._DIR / "seed" / "solver.py", cvrp._DIR / "seed" / "kernel.c",
                    Path(cvrp._DOCKERFILE), Path(__file__),
                    cvrp._DIR.parent / "common.py",
                    cvrp._DIR.parent.parent / "execution" / "sandbox.py",
                    cvrp._DIR.parent.parent / "execution" / "policy.py"]
    from evoharness.execution.sandbox import _HAVE_UNSHARE
    return {
        "python": platform.python_version(), "host_numpy": np.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "host_cpu_count": os.cpu_count(),
        "host_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "backend": "docker" if ready else "unshare" if _HAVE_UNSHARE else "socket-stub fallback",
        "container_image": cvrp._IMAGE if ready else None,
        "container_image_id": _command(["docker", "image", "inspect", cvrp._IMAGE,
                                         "--format", "{{.Id}}"] ) if ready else None,
        "cpuset": cvrp._CPUSET if ready else None,
        "memory_mb": 1024, "container_cpus": 1 if ready else None,
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "working_tree_dirty": bool(_command(["git", "status", "--porcelain"])),
        "instance_sha256": files,
        "source_sha256": {str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT)
                          else str(path): _sha256(path.read_bytes()) for path in source_paths},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--splits", nargs="+", choices=list(cvrp._SPLITS), default=["val", "private"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="show plan without evaluating or building")
    args = parser.parse_args(argv)
    try:
        if args.repeats < 1 or len(args.splits) != len(set(args.splits)):
            raise ValueError("--repeats must be positive and --splits must be distinct")
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite benchmark evidence: {args.output}")
        candidates = load_candidates(args.candidate)
        seconds = sum(len(cvrp._SPLITS[s]) * cvrp._BUDGET[s] for s in args.splits)
        print(f"{len(candidates)} frozen programs × {args.repeats} repeats × "
              f"{len(args.splits)} splits; nominal instance budget "
              f"{seconds * len(candidates) * args.repeats:.0f}s including compilation, "
              "plus container/import overhead; "
              "0 model calls", flush=True)
        if args.dry_run:
            return 0
        metadata = environment_metadata(args.splits, args.allow_fallback)
        selected_policy = "task_default" if args.allow_fallback else "require_container"
        metadata["execution_policy"] = selected_policy
        with execution_policy(selected_policy, "cvrp"):
            report = run_benchmark(candidates, args.splits, args.repeats, args.output, metadata)
        return 0 if report["status"] == "complete" else 1
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
