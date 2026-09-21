"""Trusted physics boundary for open-ended research agents.

Agents may write an arbitrary solve(fm, rng) method, but this host tool owns the
bank blackout, evaluator configuration, serialization, provenance, and fidelity
promotion. It never accepts an inherited source boundary for the independent arm.
"""
from __future__ import annotations

import argparse
import ast
import atexit
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import time
from pathlib import Path

from evoharness.paths import REPO_ROOT, RUNS_DIR

_TASK_DIR = Path(__file__).resolve().parents[1]

from evoharness.engine.campaign_lock import CampaignLock
from evoharness.research.store import ResearchStore
from evoharness.tasks.stellar_p2.discovery.boundaries import (BasinRecord, Provenance, boundary_bank_kin,
                                       describe, enrich_metrics, resolution_signature,
                                       stable_hash)
from evoharness.tasks.stellar_p2.discovery.census import atomic_json


FORBIDDEN_IMPORTS = {"os", "sys", "pathlib", "subprocess", "socket", "requests",
                     "urllib", "http", "shutil", "pickle", "builtins", "io",
                     "importlib", "codecs"}
FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "getattr"}
FORBIDDEN_ATTRIBUTE_CALLS = FORBIDDEN_CALLS | {"import_module", "read_text", "read_bytes"}


def audit_candidate(code: str) -> list[str]:
    """Cheap fail-closed provenance/capability audit before sandbox execution."""
    errors = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
               node.name == "solve" for node in ast.walk(tree)):
        errors.append("candidate must define solve(fm, rng)")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    errors.append(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in FORBIDDEN_IMPORTS:
                errors.append(f"forbidden import: {node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                errors.append(f"forbidden call: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                    "seed_bank", "seed_bank_info"}:
                errors.append(f"public-bank access forbidden: {node.func.attr}")
            if (isinstance(node.func, ast.Attribute) and
                    node.func.attr in FORBIDDEN_ATTRIBUTE_CALLS):
                errors.append(f"forbidden attribute call: {node.func.attr}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append(f"dunder capability access forbidden: {node.attr}")
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant) and key.value in {"r_cos", "z_sin"}
                        and isinstance(value, (ast.List, ast.Tuple))):
                    errors.append("embedded raw boundary coefficients are forbidden in the independent arm")
        elif isinstance(node, (ast.List, ast.Tuple)):
            numeric = sum(isinstance(value, ast.Constant) and
                          isinstance(value.value, (int, float)) for value in ast.walk(node))
            if numeric >= 24:
                errors.append("large embedded numeric blocks are forbidden in the independent arm")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            numeric_tokens = re.findall(
                r"(?<![A-Za-z_])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", node.value)
            if len(numeric_tokens) >= 24:
                errors.append("large numeric strings are forbidden in the independent arm")
    return sorted(set(errors))


def _artifact(run_dir: Path, candidate_hash: str, fidelity: str) -> Path:
    return run_dir / "evaluations" / candidate_hash / f"{fidelity}.json"


def _artifact_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluator_state(task) -> tuple[str, str, dict]:
    specification = task.experiment_spec()
    if specification["seed_bank"]["mode"] != "disabled":
        raise RuntimeError("independent evaluator imported with public bank enabled")
    contract = {key: specification[key]
                for key in ("evaluator", "fidelities", "scoring", "seed_bank")}
    return stable_hash(contract), stable_hash(specification), specification


def _provenance_key(run_dir: Path, *, create: bool) -> bytes:
    path = run_dir / ".provenance-key"
    if create and not path.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(secrets.token_bytes(32))
        except FileExistsError:
            pass
    if not path.exists():
        raise RuntimeError("run provenance key is missing")
    return path.read_bytes()


def _signed(record: dict, run_dir: Path) -> dict:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return {**record, "artifact_signature": hmac.new(
        _provenance_key(run_dir, create=True), payload, hashlib.sha256).hexdigest()}


def _verify_signature(record: dict, run_dir: Path) -> None:
    signature = record.get("artifact_signature", "")
    unsigned = {key: value for key, value in record.items() if key != "artifact_signature"}
    payload = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    expected = hmac.new(_provenance_key(run_dir, create=False), payload,
                        hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise RuntimeError("artifact signature mismatch")


def _novelty_record(boundary: dict) -> dict:
    bank_file = _TASK_DIR / "seed_bank.json"
    bank_payload = json.loads(bank_file.read_text())
    distance, cosine = boundary_bank_kin(boundary, bank_payload["seeds"])
    robust = bool(distance is not None and cosine is not None and
                  distance >= .003 and cosine < .9999)
    return {"bank_sha256": hashlib.sha256(bank_file.read_bytes()).hexdigest(),
            "bank_distance": distance, "bank_cosine": cosine,
            "robust_novelty": robust,
            "criterion": "measured distance>=0.003 and measured cosine<0.9999",
            "error": "no same-NFP public reference" if distance is None else None}


def _publish_observation(args, record: dict, artifact: Path) -> None:
    """Connect trusted physics to the campaign belief state when explicitly requested."""
    if not args.research_run:
        return
    state_path = args.research_run.resolve() / "state.json"
    if not state_path.exists():
        raise RuntimeError(f"research state does not exist: {state_path}")
    state = json.loads(state_path.read_text())
    store = ResearchStore(args.research_run, state["objective"])
    experiment = store.state.get("experiments", {}).get(args.experiment_id)
    if experiment is None:
        raise RuntimeError(f"unknown research experiment: {args.experiment_id}")
    metric = experiment["metric"]
    if metric == "score":
        value = record.get("score")
    else:
        value = (record.get("metrics") or {}).get(metric)
    if value is None or not math.isfinite(float(value)):
        raise RuntimeError(f"physics artifact does not contain finite metric {metric!r}")
    try:
        artifact_reference = str(artifact.resolve().relative_to(REPO_ROOT))
    except ValueError:
        artifact_reference = str(artifact.resolve())
    store.record_observation(
        args.experiment_id, float(value), actual_cost=float(record.get("seconds", 0)),
        evaluator_hash=record["evaluator_fingerprint"], artifact=artifact_reference,
        held_out=not args.steering_observation)


def _screen(args) -> int:
    code = args.candidate.read_text()
    errors = audit_candidate(code)
    if errors:
        print(json.dumps({"status": "rejected", "errors": errors}))
        return 3
    candidate_hash = hashlib.sha256(code.encode()).hexdigest()
    os.environ["STELLAR_NO_BANK"] = "1"       # must precede task import
    os.environ["STELLAR_TRAIN_OVERRIDES"] = json.dumps({
        "max_evals": args.max_evals, "cpu_budget": args.cpu_seconds,
        "collect_top": 3, "workers": 1, "eval_timeout": args.eval_timeout})
    from evoharness.tasks.stellar_p2.task import TASK
    evaluator_fingerprint, execution_fingerprint, evaluator_spec = _evaluator_state(TASK)
    lock = CampaignLock(RUNS_DIR / ".stellar-physics.lock")
    lock.__enter__(); atexit.register(lock.__exit__, None, None, None)
    started = time.time()
    result = TASK.evaluate(code, "train")
    finished = time.time()
    metrics = enrich_metrics(result.metrics)
    record = {"schema_version": 2, "arm": "independent", "bank_disabled": True,
              "candidate_hash": candidate_hash, "candidate_path": str(args.candidate.resolve()),
              "candidate_source": code, "evaluator_fingerprint": evaluator_fingerprint,
              "execution_fingerprint": execution_fingerprint,
              "evaluator_spec": evaluator_spec,
              "fidelity": "very_low_fidelity", "started_at": started,
              "finished_at": finished, "seconds": finished - started, "error": result.error,
              "shaped_score": result.score, "p2_score": metrics.get("p2_score"),
              "honest_score": metrics.get("honest_score"), "metrics": metrics}
    if not result.error:
        key = hashlib.sha1(code.encode()).hexdigest()[:12]
        record["boundary"] = TASK._bcache[key]
        record["boundary_hash"] = stable_hash(record["boundary"])
        record["novelty_audit"] = _novelty_record(record["boundary"])
    record = _signed(record, args.run_dir)
    path = _artifact(args.run_dir, candidate_hash, "very_low_fidelity")
    atomic_json(path, record)
    _publish_observation(args, record, path)
    print(json.dumps({k: v for k, v in record.items()
                      if k not in {"candidate_source", "boundary"}}, sort_keys=True))
    return 0 if not result.error else 2


def _confirm(args, official: bool) -> int:
    source = _artifact(args.run_dir, args.candidate_hash, "very_low_fidelity")
    if not source.exists():
        raise RuntimeError("fresh independent VLF artifact is required")
    vlf = json.loads(source.read_text())
    _verify_signature(vlf, args.run_dir)
    if vlf.get("arm") != "independent" or not vlf.get("bank_disabled") or not vlf.get("boundary"):
        raise RuntimeError("invalid independent VLF provenance")
    if hashlib.sha256(vlf.get("candidate_source", "").encode()).hexdigest() != args.candidate_hash:
        raise RuntimeError("VLF candidate source hash mismatch")
    if stable_hash(vlf["boundary"]) != vlf.get("boundary_hash"):
        raise RuntimeError("VLF boundary hash mismatch")
    source_errors = audit_candidate(vlf["candidate_source"])
    if source_errors:
        raise RuntimeError(f"stored candidate fails current audit: {source_errors}")
    novelty = _novelty_record(vlf["boundary"])
    if not novelty["robust_novelty"]:
        raise RuntimeError(f"independence is not certifiable: {novelty}")
    if official:
        lf_path = _artifact(args.run_dir, args.candidate_hash, "low_fidelity")
        if not lf_path.exists():
            raise RuntimeError("LF confirmation is required before official evaluation")
        lf = json.loads(lf_path.read_text())
        _verify_signature(lf, args.run_dir)
        if (lf.get("parent_artifact_sha256") != _artifact_hash(source) or
                lf.get("boundary_hash") != vlf["boundary_hash"] or
                lf.get("candidate_hash") != args.candidate_hash):
            raise RuntimeError("LF artifact chain does not match the VLF artifact")
        metrics = lf.get("metrics") or {}
        lf_record = BasinRecord(args.candidate_hash, vlf["boundary"], metrics,
                                Provenance(args.candidate_hash, 0),
                                vlf["evaluator_fingerprint"], "low_fidelity")
        describe(lf_record)
        if (lf.get("error") or metrics.get("feasibility", float("inf")) > .1 or
                (metrics.get("feasibility", float("inf")) > .01 and
                 (resolution_signature(lf_record) != (0, 0) or
                  (metrics.get("selection_score", float("-inf")) < .66 and
                   metrics.get("objective_L", float("-inf")) < 13.2)))):
            raise RuntimeError("LF candidate has not entered the official corridor")
    os.environ["STELLAR_NO_BANK"] = "1"
    from evoharness.tasks.stellar_p2.task import TASK, verify_boundary
    evaluator_fingerprint, execution_fingerprint, evaluator_spec = _evaluator_state(TASK)
    if evaluator_fingerprint != vlf["evaluator_fingerprint"]:
        raise RuntimeError("evaluator fingerprint changed; re-screen in a new campaign")
    lock = CampaignLock(RUNS_DIR / ".stellar-physics.lock")
    lock.__enter__(); atexit.register(lock.__exit__, None, None, None)
    fidelity = "official" if official else "low_fidelity"
    result = verify_boundary(vlf["boundary"], official=official,
                             fidelity="low_fidelity")
    metrics = enrich_metrics(result.metrics)
    parent_path = (_artifact(args.run_dir, args.candidate_hash, "low_fidelity")
                   if official else source)
    record = {"schema_version": 2, "arm": "independent", "bank_disabled": True,
              "candidate_hash": args.candidate_hash,
              "boundary_hash": vlf["boundary_hash"],
              "evaluator_fingerprint": evaluator_fingerprint,
              "execution_fingerprint": execution_fingerprint,
              "evaluator_spec": evaluator_spec,
              "parent_artifact_sha256": _artifact_hash(parent_path),
              "fidelity": fidelity, "error": result.error,
              "official_score" if official else "low_fidelity_score": result.score,
              "p2_score": metrics.get("p2_score"),
              "honest_score": metrics.get("honest_score"), "metrics": metrics,
              "novelty_audit": novelty, "seconds": result.seconds}
    record = _signed(record, args.run_dir)
    path = _artifact(args.run_dir, args.candidate_hash, fidelity)
    atomic_json(path, record)
    _publish_observation(args, record, path)
    print(json.dumps(record, sort_keys=True))
    return 0 if not result.error else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path,
                    default=RUNS_DIR / "stellar-evaluate-1")
    sub = ap.add_subparsers(dest="command", required=True)
    screen = sub.add_parser("screen")
    screen.add_argument("candidate", type=Path)
    screen.add_argument("--max-evals", type=int, default=16)
    screen.add_argument("--cpu-seconds", type=float, default=600.0)
    screen.add_argument("--eval-timeout", type=float, default=150.0)
    for name in ("confirm", "official"):
        parser = sub.add_parser(name); parser.add_argument("candidate_hash")
    for parser in (screen, *[sub.choices[name] for name in ("confirm", "official")]):
        parser.add_argument("--research-run", type=Path,
                            help="campaign state receiving this trusted observation")
        parser.add_argument("--experiment-id",
                            help="pre-registered experiment whose metric this evaluates")
        parser.add_argument("--steering-observation", action="store_true",
                            help="mark as used for steering rather than held-out validation")
    args = ap.parse_args()
    args.run_dir = args.run_dir.resolve()
    if bool(args.research_run) != bool(args.experiment_id):
        ap.error("--research-run and --experiment-id must be supplied together")
    if args.command == "screen":
        if not (1 <= args.max_evals <= 400 and 60 <= args.cpu_seconds <= 7200 and
                30 <= args.eval_timeout <= 1800):
            ap.error("screen budgets outside host limits")
        return _screen(args)
    return _confirm(args, official=args.command == "official")


if __name__ == "__main__":
    raise SystemExit(main())
