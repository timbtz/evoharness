"""Fail-closed requirement audit for the Stellar structural-discovery plan."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "plans/stellar-070-implementation.json"
DONE = {"tested", "validated"}


def audit(manifest: Path = MANIFEST) -> dict:
    data = json.loads(manifest.read_text())
    rows = []
    ids = {r["id"] for r in data["requirements"]}
    for req in data["requirements"]:
        missing = [p for p in req.get("evidence", []) if not (ROOT / p).exists()]
        bad_blockers = [b for b in req.get("blocked_by", []) if b not in ids]
        effective = req["status"]
        if missing or bad_blockers:
            effective = "invalid"
        rows.append({**req, "effective_status": effective,
                     "missing_evidence": missing, "unknown_blockers": bad_blockers})
    lookup = {r["id"]: r for r in rows}
    for row in rows:
        blockers = row.get("blocked_by", [])
        row["blockers_remaining"] = [b for b in blockers
                                     if lookup[b]["effective_status"] not in DONE]
        if row["status"] == "blocked" and not row["blockers_remaining"]:
            row["effective_status"] = "ready"
    return {"schema_version": 1, "requirements": rows,
            "counts": {s: sum(r["effective_status"] == s for r in rows)
                       for s in sorted({r["effective_status"] for r in rows})},
            "large_run_ready": (not lookup["large-live"]["blockers_remaining"] and lookup["large-live"]["effective_status"] in {"ready", "validated"})}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--require-ready", action="store_true")
    args = ap.parse_args()
    result = audit()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("PLAN_AUDIT " + json.dumps({"counts": result["counts"],
                                          "large_run_ready": result["large_run_ready"]}))
        for row in result["requirements"]:
            if row["effective_status"] not in DONE:
                print(f"{row['id']}: {row['effective_status']} — {row['description']}")
    return 2 if args.require_ready and not result["large_run_ready"] else 0


if __name__ == "__main__":
    sys.exit(main())
