"""Aggregate structural campaign reports into the plan's headline metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _llm_summary(paths: list[Path]) -> dict:
    """Aggregate z.ai control-plane evidence without conflating it with physics."""
    proposal_dirs = {p.parent for p in paths if p.name == "accepted.json"}
    calls = proposals = accepted = compiled = rejected = 0
    usd = prompt_tokens = completion_tokens = 0.0
    for directory in sorted(proposal_dirs):
        accepted_path = directory / "accepted.json"
        try:
            payload = json.loads(accepted_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        proposals += int(payload.get("proposals_requested", 0))
        accepted += len(payload.get("accepted", []))
        ledger = directory / "ledger.jsonl"
        if not ledger.exists():
            continue
        for line in ledger.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") == "llm_call":
                calls += 1
                usd += float(row.get("usd", 0.0) or 0.0)
                prompt_tokens += float(row.get("prompt_tokens", 0) or 0)
                completion_tokens += float(row.get("completion_tokens", 0) or 0)
            elif row.get("type") == "structural_proposal_rejected":
                rejected += 1
    return {"proposals_requested": proposals, "operators_accepted": accepted,
            "operators_rejected": rejected, "llm_calls": calls,
            "llm_usd": round(usd, 8), "llm_prompt_tokens": int(prompt_tokens),
            "llm_completion_tokens": int(completion_tokens)}


def build(paths: list[Path]) -> dict:
    loaded = [(p, json.loads(p.read_text())) for p in paths]
    loaded = [(p, r) for p, r in loaded
              if r.get("report_kind") == "structural_census"]
    paths, reports = [x[0] for x in loaded], [x[1] for x in loaded]
    leaders = [x for r in reports for x in r.get("leaders", [])]
    l_values = [float(v) for r in reports for v in
                [*(r.get("best_L_by_feasibility_band") or {}).values(),
                 *(f.get("best_L") for f in r.get("families", {}).values())]
                if v is not None]
    llm = _llm_summary(sorted(ROOT.glob("runs/**/accepted.json")))
    return {"schema_version": 1, "campaigns": len(reports),
        "starts_generated": sum(r.get("jobs", 0) for r in reports),
        "converged_vlf": sum(r.get("evaluated", 0) for r in reports),
        "failed": sum(r.get("failed", 0) for r in reports),
        "distinct_physical_cells": sum(r.get("distinct_cells", 0) for r in reports),
        "independent_cells": sum(r.get("distinct_cells", 0) for r in reports
                                 if r.get("bank_disabled")),
        "promotable": sum(r.get("promotable", 0) for r in reports),
        "lf_confirmed": sum(not x.get("lf", {}).get("error") for x in leaders if "lf" in x),
        "fidelity_rank_inversions": sum(r.get("tournament", {}).get("rank_inversions", 0)
                                        for r in reports),
        "best_vlf_L": max(l_values, default=None),
        "fallback_returns": sum(r.get("fallback_returns", 0) for r in reports),
        "material_gain_candidates": sum(r.get("material_gain_candidates", 0) for r in reports),
        **llm,
        "reports": [str(p) for p in paths]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="*", type=Path)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "runs/structural-supervisor/dashboard.json")
    args = ap.parse_args()
    paths = args.reports or sorted((ROOT / "runs").glob("structural-*/report.json"))
    data = build(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True))
    print("STRUCTURAL_DASHBOARD " + json.dumps(data, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
