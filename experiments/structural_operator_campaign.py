"""Bounded, resumable z.ai structural operator design campaign (no physics)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.ledger import BudgetGuard, Ledger
from core.llm import LLM
from core.structural_llm import (CONSTRUCTION_FAMILIES, TRANSFORM_FAMILIES,
                                 OperatorDesigner)


class DryRunLLM:
    """End-to-end control-plane validation without network calls or fake physics."""
    def __init__(self): self.i = 0
    def chat(self, *args, role=None, **kwargs):
        if role == "structural_critic":
            return json.dumps({"accept": True, "reasons": ["dry-run schema accepted"],
                               "risk": "medium"})
        self.i += 1
        family = "nae" if self.i % 2 else "ellipse"
        params = ({"aspect_ratio": 8.0, "max_elongation": 3.0,
                   "rotational_transform": .7, "mirror_ratio": .12,
                   "n_field_periods": 2, "max_poloidal_mode": 2,
                   "max_toroidal_mode": 2} if family == "nae" else
                  {"aspect_ratio": 8.0, "elongation": 2.5,
                   "rotational_transform": .7, "n_field_periods": 3})
        return json.dumps({"family": family, "version": f"dry-{self.i}",
            "parameters": params, "mechanism": "topology-scale structural construction",
            "expected_l_gain_fraction": .1,
            "expected_constraint_effects": {k: "must be measured" for k in
                ("aspect", "iota", "qi", "mirror", "elongation")},
            "novelty_mechanism": "independent analytic construction",
            "kill_criterion": "stop after one bounded fresh screen",
            "falsifiable_prediction": "improves L by ten percent at fixed feasibility band"})


def evidence_text(paths: list[Path]) -> str:
    blocks = []
    for path in paths:
        if path.exists():
            blocks.append(f"## {path}\n{path.read_text()[:12000]}")
    return "\n\n".join(blocks) or "No completed structural evidence yet."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "structural-operators")
    ap.add_argument("--proposals", type=int, default=4)
    ap.add_argument("--usd", type=float, default=2.0)
    ap.add_argument("--seconds", type=int, default=1800)
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--critic-model", default="glm-5.2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--balanced", action="store_true",
                    help="alternate NAE and ellipse proposal rounds")
    ap.add_argument("--families", default="nae,ellipse")
    args = ap.parse_args()
    ledger = Ledger(args.run_dir)
    guard = BudgetGuard(args.usd, args.proposals * 2, args.seconds)
    llm = DryRunLLM() if args.dry_run else LLM(ledger, guard)
    designer = OperatorDesigner(llm, args.model, args.critic_model, ledger)
    requested_families = {x.strip() for x in args.families.split(",") if x.strip()}
    executable = CONSTRUCTION_FAMILIES | TRANSFORM_FAMILIES
    if not requested_families <= executable or not requested_families:
        ap.error(f"--families must be a non-empty subset of {','.join(sorted(executable))}")
    if requested_families & TRANSFORM_FAMILIES and requested_families - TRANSFORM_FAMILIES:
        # The two arms have different prompts, critics and provenance; mixing
        # them in one campaign would hand the writer a contradictory contract.
        ap.error("transform families must be campaigned separately from construction")
    designer.allowed_families = requested_families
    evidence_paths = [ROOT / "plans/stellar-070-structural-discovery-plan.md",
                      *sorted((ROOT / "runs").glob("structural-*/report.json"))]
    evidence = evidence_text(evidence_paths)
    accepted = []
    for i in range(args.proposals):
        if args.balanced:
            ordered = sorted(requested_families)
            designer.allowed_families = {ordered[i % len(ordered)]}
        try:
            proposal, verdict = designer.propose(evidence + "\n\nAlready accepted:\n" +
                json.dumps([p.to_dict() for p in accepted], sort_keys=True))
        except Exception as exc:
            ledger.append({"type": "structural_campaign_halt", "error": str(exc)[:500]})
            break
        if proposal:
            accepted.append(proposal)
        print(json.dumps({"round": i + 1, "accepted": proposal is not None,
                          "reasons": verdict["reasons"]}), flush=True)
    output = {"schema_version": 1, "accepted": [p.to_dict() for p in accepted],
              "proposals_requested": args.proposals, "llm_calls": guard.calls,
              "usd": guard.usd}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "accepted.json").write_text(json.dumps(output, indent=2, sort_keys=True))
    print("OPERATOR_REPORT " + json.dumps({k: v for k, v in output.items() if k != "accepted"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
