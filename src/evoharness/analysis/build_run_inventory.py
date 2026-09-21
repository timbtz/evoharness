#!/usr/bin/env python3
"""Read local run evidence and export a prompt-free, strict-JSON inventory.

No model calls, evaluators, writes to runs/, or third-party dependencies.
Every JSONL is scanned; specialized metrics still require task-specific analysis.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

from evoharness.paths import REPO_ROOT, RUNS_DIR

ROOT = REPO_ROOT
TASKS = ("stellar_p2", "binpacking", "circles", "matmul_c", "cvrp", "tsp")
SUMMARY_KEYS = ("train", "val", "private", "usd", "calls", "seconds", "best_id", "stop_reason")


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def scan_jsonl(path: Path, root: Path) -> dict:
    events, models = Counter(), Counter()
    digest = hashlib.sha256()
    result = {"source": path.relative_to(root).as_posix(), "rows": 0,
              "malformed_rows": 0, "non_object_rows": 0, "accepted_candidates": 0}
    timestamps = []
    with path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            if not raw.strip():
                continue
            result["rows"] += 1
            try:
                row = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                result["malformed_rows"] += 1
                continue
            if not isinstance(row, dict):
                result["non_object_rows"] += 1
                continue
            kind = str(row.get("type", row.get("event", "untyped")))
            events[kind] += 1
            ts = row.get("ts")
            if isinstance(ts, (float, int)) and math.isfinite(ts):
                timestamps.append(ts)
            if kind == "run_start" and isinstance(row.get("config"), dict):
                cfg = row["config"]
                result["config"] = {k: cfg[k] for k in ("task", "seed", "switches", "models", "budget") if k in cfg}
                result["has_experiment_spec"] = "experiment_spec" in row
            if kind == "llm_call" and isinstance(row.get("model"), str):
                models[row["model"]] += 1
            if kind == "candidate" and row.get("accepted") is True:
                result["accepted_candidates"] += 1
            if kind == "run_end":
                result["summary"] = {k: row[k] for k in SUMMARY_KEYS if k in row}
    result.update(events=dict(sorted(events.items())), models=dict(sorted(models.items())), sha256=digest.hexdigest())
    result["has_run_end"] = bool(events["run_end"])
    if timestamps:
        for name, ts in (("first_event_utc", min(timestamps)), ("last_event_utc", max(timestamps))):
            try:
                result[name] = datetime.fromtimestamp(ts, timezone.utc).isoformat()
            except (ValueError, OverflowError, OSError):
                pass
    task = result.get("config", {}).get("task")
    result["task"] = task or next((t for t in TASKS if t in str(path)),
                                  "stellar_p2" if "stellar-" in str(path) else "other")
    return clean(result)


def build_inventory(root: Path, runs: Path, excluded_families: tuple[str, ...] = ()) -> dict:
    excluded_families = tuple(dict.fromkeys(excluded_families))
    files = sorted(p for p in runs.rglob("*") if p.is_file() and not p.is_symlink()
                   and p.relative_to(runs).parts[0] not in excluded_families) if runs.exists() else []
    families = {}
    logs = []
    for path in files:
        relative = path.relative_to(runs)
        family = relative.parts[0] if len(relative.parts) > 1 else "(root files)"
        info = families.setdefault(family, {"files": 0, "bytes": 0, "extensions": Counter(), "jsonl_files": 0})
        info["files"] += 1
        info["bytes"] += path.stat().st_size
        info["extensions"][path.suffix or "(none)"] += 1
        if path.suffix == ".jsonl":
            info["jsonl_files"] += 1
            logs.append(scan_jsonl(path, root))
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": head,
        "excluded_families": list(excluded_families),
        "scope": "Local files present at scan time; all JSONL parsed. Text logs and JSON artifacts counted, not semantically validated. Working-tree code may differ from source_commit.",
        "caveats": ["A run_end event indicates a recorded end, not scientific success.",
                    "Missing run_end can mean interruption, a live run, or a different log schema.",
                    "Scores across tasks, fidelities and shaping versions are not comparable.",
                    "Dollar totals in individual ledgers need not include external roles or compute.",
                    "Raw runs are gitignored; absent runs are not evidence of zero experiments."],
        "totals": {"files": len(files), "bytes": sum(x["bytes"] for x in families.values()),
                   "families": len(families), "jsonl_files": len(logs),
                   "jsonl_rows": sum(x["rows"] for x in logs),
                   "malformed_rows": sum(x["malformed_rows"] for x in logs),
                   "ledgers_with_run_end": sum(x["has_run_end"] for x in logs)},
        "families": [{"name": name, **info, "extensions": dict(sorted(info["extensions"].items()))} for name, info in sorted(families.items())],
        "logs": logs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS_DIR)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/data/run-inventory.json")
    parser.add_argument("--exclude-family", action="append", default=["docs-smoke-check"],
                        help="omit a top-level family; default excludes the documentation-only validation run")
    args = parser.parse_args()
    if not args.runs.is_dir():
        parser.error("local runs directory is absent; retain the committed snapshot or provide --runs")
    # Local paths in the report must remain meaningful relative to this checkout.
    try:
        args.runs.resolve().relative_to(ROOT)
    except ValueError:
        parser.error("--runs must be within this checkout")
    data = build_inventory(ROOT, args.runs.resolve(), tuple(args.exclude_family))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    args.output.with_suffix(".js").write_text("window.EVO_RUN_INVENTORY = " + payload + ";\n")
    print(json.dumps(data["totals"], indent=2))


if __name__ == "__main__":
    main()
