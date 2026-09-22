"""Post-hoc public-bank novelty audit for independently generated leaders.

Generation remains bank-disabled. This separate read-only stage loads the bank
only after the campaign is frozen and cannot influence candidate construction.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable

from evoharness.tasks.stellar_p2.discovery.boundaries import stable_hash
from evoharness.tasks.stellar_p2.discovery.census import atomic_json


def audit_leaders(leaders: list[dict], metric: Callable[[dict], tuple[float | None, float | None]]) -> list[dict]:
    rows = []
    for leader in leaders:
        boundary = leader["boundary"]
        distance, cosine = metric(boundary)
        robust = bool(distance is not None and cosine is not None and
                      distance >= .003 and cosine < .9999)
        rows.append({"job_id": leader.get("job_id") or leader.get("operator_id"),
                     "boundary_hash": stable_hash(boundary),
                     "bank_distance": distance, "bank_cosine": cosine,
                     "robust_novelty": robust,
                     "criterion": "measured distance>=0.003 and measured cosine<0.9999",
                     "audit_error": "no same-NFP public reference" if distance is None else None})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parent_report", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = json.loads(args.parent_report.read_text())
    if report.get("arm") != "independent" or not report.get("bank_disabled"):
        raise RuntimeError("novelty audit requires a frozen bank-disabled independent report")
    # Import only now, explicitly with the public bank enabled. No evaluator is
    # called and no boundary is returned to the generating campaign.
    os.environ.pop("STELLAR_NO_BANK", None)
    from evoharness.tasks.stellar_p2.task import bank_kin
    rows = audit_leaders(report.get("leaders", []), bank_kin)
    out = {"schema_version": 1, "report_kind": "structural_novelty_audit",
           "parent_report_hash": stable_hash(report), "leaders": rows,
           "robust": sum(r["robust_novelty"] for r in rows), "audited": len(rows)}
    atomic_json(args.output, out)
    print("NOVELTY_AUDIT " + json.dumps({k: v for k, v in out.items() if k != "leaders"}))
    return 0 if all(r["robust_novelty"] for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
