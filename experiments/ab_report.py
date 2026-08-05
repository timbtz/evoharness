"""Arm-vs-arm report for the gradient A/B campaign.

The campaign's whole point is one comparison: with the exact boundary-margin
gradient tools (fm.aspect / fm.margin_grad / fm.margin_step) versus without
them, at matched seed, resume point, budget and stall limit. This reads the
ledgers and answers it.

Two things it refuses to fudge:

* It ranks by HONEST score (p2 discounted to the 0.002 margin target at the
  official 0.92 slope), not by `train` (which also carries the novelty penalty)
  and not by `p2_score` (which rewards camping the 1e-2 tolerance). A gradient
  tool that lands the aspect wall to 11 digits is a better camping instrument
  than the ladder was, so the camped number would flatter the treated arm.
* It reports whether the treated arm ACTUALLY CALLED the tools. An A/B where
  the writer ignored the treatment is a null about prompt engineering, not
  about gradients, and the two must not be confused.

  .venv/bin/python experiments/ab_report.py            # all pairs
  .venv/bin/python experiments/ab_report.py --pair p1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
CAMP = _ROOT / "runs" / "grad_campaign"
MARGIN_TARGET, MARGIN_SLOPE = 0.002, 0.92
TOOL_RE = re.compile(r"fm\.(margin_grad|margin_step|aspect)\b")


def honest(m: dict) -> float | None:
    """p2 discounted to the margin target — the number that is not camping."""
    if m.get("honest_score") is not None:
        return float(m["honest_score"])
    p2, feas = m.get("p2_score"), m.get("feasibility")
    if p2 is None or feas is None:
        return None
    return float(p2) - MARGIN_SLOPE * max(0.0, float(feas) - MARGIN_TARGET)


def read_arm(run_dir: Path) -> dict | None:
    led = run_dir / "ledger.jsonl"
    if not led.exists():
        return None
    rows, tool_calls = [], 0
    for line in led.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(r)
        tool_calls += len(TOOL_RE.findall(line))

    cands = [r for r in rows if r.get("type") == "candidate"]
    live = [r for r in cands
            if (r["scores"].get("train") or -9e9) > -1e8]
    end = next((r for r in rows if r.get("type") == "run_end"), None)

    scored = []
    for r in live:
        m = (r.get("meta") or {}).get("metrics") or {}
        h = honest(m)
        if h is not None:
            scored.append((h, r, m))
    scored.sort(key=lambda t: -t[0])

    out = {
        "run": run_dir.name,
        "candidates": len(cands),
        "dead": len(cands) - len(live),
        "tool_calls": tool_calls,
        "finished": end is not None,
        "stop_reason": (end or {}).get("stop_reason"),
        "usd": (end or {}).get("usd"),
        "wall_min": round(((end or {}).get("seconds") or 0) / 60, 1),
        "private": (end or {}).get("private"),
        "train_of_best": (end or {}).get("train"),
    }
    if scored:
        h, r, m = scored[0]
        out.update(best_honest=round(h, 6), best_id=r.get("id"),
                   best_p2=m.get("p2_score"),
                   best_feasibility=m.get("feasibility"),
                   best_bank_dist=m.get("bank_dist"),
                   evals_used=m.get("evals_used"),
                   evals_warm=m.get("evals_warm"), evals_soft=m.get("evals_soft"))
        # how many candidates cleared the resumed seed's honest score
        seed_h = None
        for hh, rr, _mm in scored:
            if rr.get("id") == "c0000":
                seed_h = hh
        if seed_h is not None:
            out["seed_honest"] = round(seed_h, 6)
            out["beat_seed"] = sum(1 for hh, _r, _m in scored if hh > seed_h + 1e-9)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", help="p1 | p2 | p3 | p4 (default: all found)")
    a = ap.parse_args()

    pairs = [a.pair] if a.pair else ["p1", "p2", "p3", "p4"]
    report = []
    for pair in pairs:
        on, off = read_arm(CAMP / f"{pair}-on"), read_arm(CAMP / f"{pair}-off")
        if on is None and off is None:
            continue
        row = {"pair": pair, "on": on, "off": off}
        if on and off and "best_honest" in on and "best_honest" in off:
            row["delta_honest_on_minus_off"] = round(
                on["best_honest"] - off["best_honest"], 6)
            row["both_finished"] = bool(on["finished"] and off["finished"])
        report.append(row)

        print(f"\n=== {pair} ===")
        hdr = f"{'':22s} {'ON (tools)':>16s} {'OFF (control)':>16s}"
        print(hdr)
        for k in ("candidates", "dead", "tool_calls", "best_honest",
                  "best_p2", "best_feasibility", "best_bank_dist", "seed_honest",
                  "beat_seed", "evals_used", "evals_warm", "evals_soft",
                  "private", "usd", "wall_min", "finished", "stop_reason"):
            vo = (on or {}).get(k, "-")
            vf = (off or {}).get(k, "-")
            if isinstance(vo, float):
                vo = round(vo, 6)
            if isinstance(vf, float):
                vf = round(vf, 6)
            print(f"{k:22s} {str(vo)[:16]:>16s} {str(vf)[:16]:>16s}")
        if "delta_honest_on_minus_off" in row:
            d = row["delta_honest_on_minus_off"]
            verdict = ("tools ahead" if d > 1e-6 else
                       "control ahead" if d < -1e-6 else "tie")
            done = "" if row["both_finished"] else "  (INCOMPLETE — both arms not finished)"
            print(f"\n  delta honest (ON - OFF) = {d:+.6f}   -> {verdict}{done}")
        if on and on.get("tool_calls", 0) == 0:
            print("  WARNING: treated arm never called the tools — this is a "
                  "null about prompting, not about gradients")
        if off and off.get("tool_calls", 0) > 0:
            print("  WARNING: control arm shows tool calls — the kill-switch or "
                  "the doc-stripping leaked")

    (CAMP / "ab_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {CAMP / 'ab_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
