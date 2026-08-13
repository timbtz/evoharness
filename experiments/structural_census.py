"""Structural Census 1: resumable, bank-disabled NAE/ellipse basin discovery."""
from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.structural_discovery import (BasinRecord, OperatorSpec, Provenance,
                                       census_promotable, describe, enrich_metrics, stable_hash,
                                       successive_halving)
from core.campaign_lock import CampaignLock


def design_axis_repair(seed: int, samples: int) -> list[OperatorSpec]:
    """Focused NFP-3 axis/boundary DOE from measured mirror+QI failures."""
    import numpy as np
    rng = np.random.default_rng(seed)
    jobs = []
    for i in range(samples):
        u = (i + rng.random()) / samples
        params = {"aspect_ratio": round(8.0 + 2.0 * u, 6),
            "max_elongation": round(1.8 + 1.2 * ((u + .31) % 1), 6),
            "rotational_transform": round(.5 + .4 * ((u + .67) % 1), 6),
            "mirror_ratio": round(.08 * ((u + .47) % 1), 6),
            "torsion": round(.04 + .10 * ((u + .79) % 1), 6),
            "n_field_periods": 3, "max_poloidal_mode": 3, "max_toroidal_mode": 3}
        jobs.append(OperatorSpec("nae_axis", "axis-repair1", params,
            "reduce measured mirror and QI violations through low-mirror low-torsion axis co-design",
            .10, "retire if no sample materially reduces max(mirror, qi) violation"))
    return jobs


def design_axis_radius(seed: int, samples: int) -> list[OperatorSpec]:
    """Continue toward near-axis validity by shrinking finite boundary radius."""
    import numpy as np
    rng = np.random.default_rng(seed)
    jobs = []
    for i in range(samples):
        u = (i + rng.random()) / samples
        params = {"aspect_ratio": round(9.5 + 4.5 * u, 6),
            "max_elongation": round(2.0 + 1.0 * ((u + .37) % 1), 6),
            "rotational_transform": round(.65 + .55 * ((u + .71) % 1), 6),
            "mirror_ratio": round(.02 + .10 * ((u + .23) % 1), 6),
            "torsion": round(.03 + .12 * ((u + .61) % 1), 6),
            "n_field_periods": 3, "max_poloidal_mode": 3, "max_toroidal_mode": 3}
        jobs.append(OperatorSpec("nae_axis", "axis-radius1", params,
            "shrink finite boundary radius toward the first-order QI-valid near-axis limit",
            .10, "reject exact aspect >14 before solve; retire if QI/mirror do not improve"))
    return jobs


def design(seed: int, per_arm: int, nae_mode: int = 2,
           profile: str = "broad") -> list[OperatorSpec]:
    """Deterministic stratified design over physically meaningful coordinates."""
    if profile == "axis_repair":
        return design_axis_repair(seed, per_arm)
    if profile == "axis_radius":
        return design_axis_radius(seed, per_arm)
    if profile != "broad":
        raise ValueError(f"unknown design profile: {profile}")
    import numpy as np
    rng = np.random.default_rng(seed)
    jobs: list[OperatorSpec] = []
    for nfp in (2, 3):
        for family in ("nae", "nae_axis", "ellipse"):
            for i in range(per_arm):
                u = (i + rng.random()) / per_arm
                if family in {"nae", "nae_axis"}:
                    params = {
                        "aspect_ratio": round(6.5 + 3.0 * u, 6),
                        "max_elongation": round(2.0 + 2.5 * ((u + .37) % 1), 6),
                        "rotational_transform": round(0.35 + 0.9 * ((u + .71) % 1), 6),
                        "mirror_ratio": round(0.05 + 0.15 * ((u + .19) % 1), 6),
                        "n_field_periods": nfp, "max_poloidal_mode": nae_mode,
                        "max_toroidal_mode": nae_mode,
                    }
                    if family == "nae_axis":
                        params["torsion"] = round(.03 + .32 * ((u + .53) % 1), 6)
                        mechanism = "near-axis magnetic-axis torsion and finite-radius boundary co-design"
                    else:
                        mechanism = "independent near-axis QI construction across topology and shape"
                else:
                    params = {
                        "aspect_ratio": round(6.5 + 3.0 * u, 6),
                        "elongation": round(1.5 + 2.5 * ((u + .43) % 1), 6),
                        "rotational_transform": round(0.35 + 0.9 * ((u + .77) % 1), 6),
                        "n_field_periods": nfp,
                    }
                    mechanism = "rotating-ellipse topology pivot with low-dimensional shape control"
                jobs.append(OperatorSpec(
                    family=family, version="census1", parameters=params,
                    mechanism=mechanism, expected_l_gain_fraction=0.10,
                    kill_criterion="retire region after bounded non-convergence or catastrophic feasibility"))
    return jobs


def candidate_code(spec: OperatorSpec) -> str:
    methods = {"nae": "seed_nae", "nae_axis": "seed_nae_axis", "ellipse": "seed_ellipse"}
    if spec.family not in methods:
        raise ValueError(f"construction executor does not support {spec.family!r}")
    required = ({"aspect_ratio", "max_elongation", "rotational_transform", "mirror_ratio",
                 "n_field_periods", "max_poloidal_mode", "max_toroidal_mode"}
                | ({"torsion"} if spec.family == "nae_axis" else set())
                if spec.family in {"nae", "nae_axis"} else
                {"aspect_ratio", "elongation", "rotational_transform", "n_field_periods"})
    if set(spec.parameters) != required:
        raise ValueError(f"{spec.family} parameters must be exactly {sorted(required)}")
    p = spec.parameters
    if not (4 <= float(p["aspect_ratio"]) <= 14 and int(p["n_field_periods"]) in (2, 3)):
        raise ValueError("construction aspect/NFP outside census envelope")
    elong = float(p.get("max_elongation", p.get("elongation")))
    if not (1.0 <= elong <= 6.0) or not (.1 <= abs(float(p["rotational_transform"])) <= 2.0):
        raise ValueError("construction elongation/transform outside safe envelope")
    if spec.family in {"nae", "nae_axis"} and not (0 <= float(p["mirror_ratio"]) <= .4 and
            int(p["max_poloidal_mode"]) in (1, 2, 3) and
            int(p["max_toroidal_mode"]) in (1, 2, 3)):
        raise ValueError("NAE mode/mirror parameters outside safe envelope")
    if spec.family == "nae_axis" and not (.03 <= float(p["torsion"]) <= .35):
        raise ValueError("NAE axis torsion outside safe envelope")
    method = methods[spec.family]
    params = repr(spec.parameters)
    return f'''def solve(fm, rng):
    # One start, one fresh VLF screen, no incumbent fallback.
    boundary = fm.{method}(**{params})
    aspect = fm.aspect(boundary)
    if not (4.0 <= aspect <= 14.0):
        raise RuntimeError("geometry screen rejected aspect %.6g" % aspect)
    metrics = fm.eval(boundary)
    if metrics is None or metrics.get("soft_fail"):
        raise RuntimeError("fresh VLF screen failed: %s" % fm.last_error)
    return boundary
'''


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def file_sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retryable(error: str | None) -> bool:
    """Retry infrastructure transients, never deterministic invalid physics."""
    text = (error or "").lower()
    return any(marker in text for marker in ("timeout", "timed out", "docker",
                                               "connection reset", "oom"))


def load_operator_specs(path: Path) -> tuple[list[OperatorSpec], list[dict]]:
    payload = json.loads(path.read_text())
    rows = payload.get("accepted")
    if not isinstance(rows, list):
        raise ValueError("operator file must contain an accepted list")
    specs, rejected = [], []
    for row in rows:
        raw = row.get("operator") if isinstance(row, dict) else None
        if not isinstance(raw, dict):
            raise ValueError("accepted entry lacks operator")
        try:
            spec = OperatorSpec(**raw)
            candidate_code(spec)  # fail before any physics if it is not executable
            specs.append(spec)
        except (TypeError, ValueError) as exc:
            rejected.append({"entry": row, "error": str(exc)})
    if not specs:
        raise ValueError(f"operator queue has no executable entries; rejected={rejected}")
    return specs, rejected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7001)
    ap.add_argument("--per-arm", type=int, default=2)
    ap.add_argument("--nae-mode", type=int, choices=(1, 2, 3), default=2)
    ap.add_argument("--families", default="nae,ellipse")
    ap.add_argument("--profile", choices=("broad", "axis_repair", "axis_radius"), default="broad")
    ap.add_argument("--max-jobs", type=int)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--promote", type=int, default=3)
    ap.add_argument("--lf", action="store_true", help="LF-confirm promoted leaders")
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--operators", type=Path,
                    help="accepted.json from structural_operator_campaign")
    args = ap.parse_args()
    if args.operators:
        args.operators = args.operators.resolve()
    if args.retries < 0 or args.retries > 2:
        ap.error("--retries must be between 0 and 2")

    # Must precede task import: this is a technical isolation boundary.
    os.environ["STELLAR_NO_BANK"] = "1"
    os.environ.setdefault("STELLAR_TRAIN_OVERRIDES", json.dumps({
        "max_evals": 2, "cpu_budget": 150.0, "collect_top": 1, "workers": 1,
        "eval_timeout": 75.0}))
    from tasks.stellar_p2.task import TASK, verify_boundary

    physics_lock = CampaignLock(ROOT / "runs/.stellar-physics.lock")
    physics_lock.__enter__()
    atexit.register(physics_lock.__exit__, None, None, None)

    run_dir = args.run_dir or ROOT / "runs" / f"structural-census1-s{args.seed}"
    frozen_spec = {"schema_version": 1, "seed": args.seed,
        "per_arm": args.per_arm, "nae_mode": args.nae_mode, "families": args.families,
        "profile": args.profile,
        "max_jobs": args.max_jobs, "retries": args.retries,
        "promote": args.promote, "lf": args.lf,
        "operators": (str(args.operators.relative_to(ROOT.resolve())) if args.operators else None),
        "operators_sha256": file_sha256(args.operators) if args.operators else None,
        "source_sha256": {p: file_sha256(ROOT / p) for p in (
            "experiments/structural_census.py", "core/structural_discovery.py",
            "core/stellar_operators.py", "tasks/stellar_p2/task.py")}}
    spec_path = run_dir / "experiment.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != frozen_spec:
        raise RuntimeError("frozen experiment specification differs; use a new run directory")
    if not spec_path.exists():
        atomic_json(spec_path, frozen_spec)
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema_version": 1, "seed": args.seed, "started_at": time.time(), "jobs": {}}
    evaluator = TASK.experiment_spec()
    fingerprint = stable_hash(evaluator["evaluator"])
    if state.get("seed") != args.seed:
        raise RuntimeError("refusing to resume state created with a different seed")
    previous_fp = state.get("evaluator_fingerprint")
    if previous_fp is not None and previous_fp != fingerprint:
        raise RuntimeError("stale evaluator fingerprint: use a new run directory")
    state["evaluator_fingerprint"] = fingerprint
    rejected_queue = []
    if args.operators:
        specs, rejected_queue = load_operator_specs(args.operators)
    else:
        specs = design(args.seed, args.per_arm, args.nae_mode, args.profile)
    allowed_families = {x.strip() for x in args.families.split(",") if x.strip()}
    if not allowed_families <= {"nae", "nae_axis", "ellipse"}:
        raise ValueError("--families supports only nae,nae_axis,ellipse")
    specs = [s for s in specs if s.family in allowed_families]
    if args.max_jobs is not None:
        specs = specs[:args.max_jobs]

    for spec in specs:
        previous = state["jobs"].get(spec.id, {})
        if previous.get("status") == "evaluated":
            continue
        if previous.get("status") == "failed":
            prior_errors = previous.get("errors") or []
            if (prior_errors and not retryable(prior_errors[-1])) or \
                    int(previous.get("attempts", 0)) >= args.retries + 1:
                continue
        result = None
        errors = []
        for attempt in range(args.retries + 1):
            result = TASK.evaluate(candidate_code(spec), "train")
            if not result.error:
                break
            errors.append(result.error)
            if not retryable(result.error):
                break
        entry = {"operator": asdict(spec), "operator_id": spec.id,
                 "attempts": len(errors) + int(result is not None and not result.error),
                 "errors": errors, "status": "failed"}
        if result is not None and not result.error:
            # The task cache is the authoritative artifact returned by this code.
            import hashlib
            code_key = hashlib.sha1(candidate_code(spec).encode()).hexdigest()[:12]
            boundary = TASK._bcache[code_key]
            provenance = Provenance(spec.id, args.seed, independent=True,
                                    public_bank_enabled=False)
            record = BasinRecord(spec.id, boundary, enrich_metrics(result.metrics), provenance,
                                 fingerprint, "very_low_fidelity", solves=2)
            entry.update({"status": "evaluated", "boundary": boundary,
                          "metrics": record.metrics,
                          "descriptor": asdict(describe(record)),
                          "provenance": asdict(provenance), "solves": record.solves})
        state["jobs"][spec.id] = entry
        atomic_json(state_path, state)
        print(json.dumps({"job": spec.id, "status": entry["status"],
                          "score": entry.get("metrics", {}).get(
                              "honest_score", entry.get("metrics", {}).get("shaped_score")),
                          "error": errors[-1] if errors else None}), flush=True)

    records = []
    for entry in state["jobs"].values():
        if entry.get("status") != "evaluated":
            continue
        prov = Provenance(**entry["provenance"])
        record = BasinRecord(entry["operator_id"], entry["boundary"], entry["metrics"],
                             prov, fingerprint, "very_low_fidelity",
                             solves=entry.get("solves", 2))
        describe(record); records.append(record)
    leaders = successive_halving(filter(census_promotable, records), args.promote)
    report = {"schema_version": 1, "report_kind": "structural_census", "seed": args.seed, "bank_disabled": True,
              "evaluator_fingerprint": fingerprint, "jobs": len(specs),
              "evaluated": len(records), "failed": len(specs) - len(records),
              "distinct_cells": len({r.descriptor.cell for r in records}),
              "promotable": sum(census_promotable(r) for r in records),
              "queue_rejected": len(rejected_queue),
              "queue_rejections": rejected_queue,
              "best_L_by_feasibility_band": {band: max(
                  r.objective_l for r in records if r.descriptor.feasibility_bin == band)
                  for band in sorted({r.descriptor.feasibility_bin for r in records})},
              "leaders": []}
    for family in sorted({s.family for s in specs}):
        family_entries = [e for e in state["jobs"].values()
                          if e["operator"]["family"] == family]
        family_metrics = [e["metrics"] for e in family_entries if e.get("metrics")]
        report.setdefault("families", {})[family] = {
            "jobs": len(family_entries),
            "converged": sum(e["status"] == "evaluated" for e in family_entries),
            "best_L": max((m["objective_L"] for m in family_metrics
                           if m.get("objective_L") is not None), default=None),
            "best_feasibility": min((m["feasibility"] for m in family_metrics
                                     if m.get("feasibility") is not None), default=None),
        }
    for leader in leaders:
        row = {"job_id": leader.job_id, "vlf_metrics": leader.metrics,
               "descriptor": asdict(leader.descriptor), "boundary": leader.boundary}
        if args.lf:
            lf = verify_boundary(leader.boundary, fidelity="low_fidelity")
            lf_metrics = enrich_metrics(lf.metrics) if not lf.error else lf.metrics
            row["lf"] = {"score": lf.score, "metrics": lf_metrics, "error": lf.error,
                         "seconds": lf.seconds}
        report["leaders"].append(row)
    if args.lf:
        vlf_order = [r["job_id"] for r in report["leaders"]]
        lf_valid = [r for r in report["leaders"] if not r.get("lf", {}).get("error")]
        lf_order = [r["job_id"] for r in sorted(lf_valid,
            key=lambda x: x["lf"]["metrics"].get("selection_score", float("-inf")),
            reverse=True)]
        positions = {job: i for i, job in enumerate(vlf_order)}
        report["tournament"] = {"vlf_order": vlf_order, "lf_order": lf_order,
            "rank_inversions": sum(positions[a] > positions[b]
                for i, a in enumerate(lf_order) for b in lf_order[i + 1:]),
            "lf_required": True, "official_bypass_allowed": False}
    atomic_json(run_dir / "report.json", report)
    print("CENSUS_REPORT " + json.dumps({k: v for k, v in report.items()
          if k != "leaders"}, sort_keys=True))
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
