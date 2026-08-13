"""Progress and result report for the STELLAR_FULL_GRAD campaign (runs/gradrun/g*).

Answers the three questions the campaign exists to settle, per run:
  1. Did the writers actually buy a gradient, and at what step cap?
  2. Did anything beat the resumed seed on the SUBMITTABLE frontier?
  3. What did it cost?

Point 2 is the one that needs care. A candidate's honest_score can look excellent
while its bank_dist sits inside the 1e-3 export guard — that is a near-copy of a
public submission, worth nothing on the leaderboard (g1's top scorer, c0023, was
exactly this: honest 0.6299 at bank_dist 8.9e-4). So the headline number here is
the best honest score among boundaries that are actually submittable, and campers
are reported separately rather than mixed in.

  .venv/bin/python experiments/gradrun_report.py            # human report
  .venv/bin/python experiments/gradrun_report.py --oneline  # monitor line
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import re

SUBMITTABLE = 1e-3


def _meta(e: dict) -> dict:
    """meta is a dict on some records and a repr string on others."""
    m = e.get("meta")
    if isinstance(m, dict):
        return m
    try:
        return ast.literal_eval(m or "{}")
    except Exception:
        return {}


def scan(path: str) -> dict:
    r = {"cands": 0, "grad": 0, "step": 0, "caps": {}, "dead": 0, "usd": 0.0,
         "best_sub": None, "best_sub_id": None, "best_camp": None,
         "best_camp_id": None, "seed_honest": None, "accepted": 0}
    for ln in open(path, errors="ignore"):
        try:
            e = json.loads(ln)
        except Exception:
            continue
        if e.get("type") != "candidate":
            continue
        code = e.get("code") or ""
        if not code:
            continue
        r["cands"] += 1
        r["accepted"] += str(e.get("accepted")) == "True"
        if "metric_grad" in code:
            r["grad"] += 1
        if "grad_step" in code:
            r["step"] += 1
            for c in re.findall(r"cap\s*=\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", code):
                try:
                    v = float(c)
                except ValueError:
                    continue
                if v < 1.0:                      # a cap, not a loop bound
                    r["caps"][v] = r["caps"].get(v, 0) + 1
        meta = _meta(e)
        if meta.get("error"):
            r["dead"] += 1
        m = meta.get("metrics") or {}
        h, bd = m.get("honest_score"), m.get("bank_dist")
        if e.get("id") == "c0000":
            r["seed_honest"] = h
        if not isinstance(h, float) or not isinstance(bd, float):
            continue
        if bd >= SUBMITTABLE:
            if r["best_sub"] is None or h > r["best_sub"]:
                r["best_sub"], r["best_sub_id"] = h, e.get("id")
        elif r["best_camp"] is None or h > r["best_camp"]:
            r["best_camp"], r["best_camp_id"] = h, e.get("id")
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oneline", action="store_true")
    ap.add_argument("--key", action="store_true",
                    help="trigger key only (run, candidate decade, best submittable). "
                         "A watcher should compare THIS and display --digest: the "
                         "gradient count increments ~18x per run and is worth "
                         "showing but not worth waking for.")
    ap.add_argument("--digest", action="store_true",
                    help="oneline with candidate counts floored to 10s, so a "
                         "watcher fires on a moving frontier rather than on "
                         "every candidate (~100 events per campaign otherwise)")
    a = ap.parse_args()
    if a.digest or a.key:
        a.oneline = True

    out = []
    for path in sorted(glob.glob("runs/gradrun/g[123]/ledger.jsonl")):
        name = path.split("/")[2]
        r = scan(path)
        if not r["cands"]:
            continue
        caps = ", ".join(f"{v:g}x{n}" for v, n in
                         sorted(r["caps"].items())[:5]) or "none"
        sub = f"{r['best_sub']:.6f}" if r["best_sub"] else "none"
        seed = f"{r['seed_honest']:.6f}" if r["seed_honest"] else "?"
        if a.key:
            out.append(f"{name}:{r['cands'] // 10 * 10}:{sub}")
        elif a.digest:
            # Only the candidate count is coarsened. grad stays exact: it is the
            # signal being watched, and flooring it to 5s renders real low-but-
            # nonzero adoption as "grad=0", which reads as the opposite result.
            out.append(f"{name}: ~{r['cands'] // 10 * 10}c grad={r['grad']} "
                       f"best_sub={sub} (seed {seed})")
        elif a.oneline:
            out.append(f"{name}: {r['cands']}c grad={r['grad']} step={r['step']} "
                       f"caps[{caps}] best_sub={sub} (seed {seed})")
        else:
            print(f"=== {name} ===")
            print(f"  candidates {r['cands']} ({r['accepted']} accepted, {r['dead']} dead)")
            print(f"  metric_grad {r['grad']}  grad_step {r['step']}  caps: {caps}")
            print(f"  seed honest          {seed}")
            print(f"  best SUBMITTABLE     {sub}  ({r['best_sub_id']})")
            if r["best_camp"]:
                print(f"  best camper (bank_dist < 1e-3, unsubmittable): "
                      f"{r['best_camp']:.6f}  ({r['best_camp_id']})")
    if a.oneline and out:
        print(" | ".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
