"""Export the best archive boundary as a leaderboard-ready submission JSON.

Scans memory/stellar_p2/archive.jsonl, re-verifies the top candidates with the
OFFICIAL high-fidelity evaluator (the only trusted number), writes
experiments/submissions/p2-<key>-<score>.json containing the boundary in
SurfaceRZFourier JSON form plus a provenance block. Actual upload to the HF
space is manual (Tim's account).

Usage: uv run python experiments/submit_export.py [--top N] [--min-shaped S]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.stellar_p2.task import (  # noqa: E402
    _ARCHIVE, bank_distance as _nearest_bank_distance, verify_boundary)

OUT = Path(__file__).parent / "submissions"
_TOL = 0.01          # constellaration problems._DEFAULT_RELATIVE_TOLERANCE
_FEAS_SLOPE = 0.92   # official score bought per unit feasibility spent, measured
# on this task's archive (see memory/stellar_p2/performance-analysis/
# feasibility-tolerance-economics.md): paired official evals give 0.92, a
# 307-boundary regression gives 0.91. Used only to report what a submission would
# score at its source seed's feasibility margin — an estimate, flagged as such.
_BANK = json.loads((Path(__file__).resolve().parent.parent / "tasks" /
                    "stellar_p2" / "seed_bank.json").read_text())["seeds"] \
    if (Path(__file__).resolve().parent.parent / "tasks" / "stellar_p2" /
        "seed_bank.json").exists() else []


def nearest_seed(boundary: dict) -> dict | None:
    """The public submission this boundary most resembles, with similarity that a
    distance threshold alone hides (cosine, relative norm) and its own margin."""
    import numpy as np

    def vec(b):
        return np.concatenate([np.asarray(b["r_cos"], float).ravel(),
                               np.asarray(b["z_sin"], float).ravel()])
    x, best = vec(boundary), None
    for s in _BANK:
        sb = s["boundary"]
        if sb.get("n_field_periods") != boundary.get("n_field_periods"):
            continue
        if np.shape(sb["r_cos"]) != np.shape(boundary["r_cos"]):
            continue
        y = vec(sb)
        rel = float(np.linalg.norm(x - y) / np.linalg.norm(y))
        if best is None or rel < best["relative_norm"]:
            best = {"user": s["provenance"]["user"],
                    "submitted": s["provenance"].get("submitted"),
                    "official_score": s["official_score"],
                    "official_feasibility": s["official_feasibility"],
                    "relative_norm": round(rel, 6),
                    "cosine": round(float(x @ y / (np.linalg.norm(x)
                                                   * np.linalg.norm(y))), 8)}
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=3,
                    help="verify the N best archive entries by shaped score")
    ap.add_argument("--min-shaped", type=float, default=-0.5,
                    help="ignore archive entries below this shaped score")
    ap.add_argument("--key", action="append", default=[],
                    help="export these archive keys instead of the shaped-best N "
                         "(repeatable). Needed because `shaped` carries the novelty "
                         "penalty, so out-of-ball winners rank below in-ball campers.")
    args = ap.parse_args()

    if not _ARCHIVE.exists():
        sys.exit(f"no archive at {_ARCHIVE}")
    entries, seen = [], set()
    for line in _ARCHIVE.read_text().splitlines():
        e = json.loads(line)
        if e.get("shaped", -9e9) >= args.min_shaped and e["key"] not in seen:
            seen.add(e["key"])
            entries.append(e)
    if args.key:
        by_key = {e["key"]: e for e in entries}
        missing = [k for k in args.key if k not in by_key]
        if missing:
            sys.exit(f"archive has no entry for key(s): {', '.join(missing)}")
        entries = [by_key[k] for k in args.key]
        args.top = len(entries)
    else:
        entries.sort(key=lambda e: -e["shaped"])
    if not entries:
        sys.exit("archive has no entries above --min-shaped")

    OUT.mkdir(exist_ok=True)
    best = None
    for e in entries[: args.top]:
        print(f"verifying {e['key']} (shaped {e['shaped']:.4f}, "
              f"{e['fidelity']}) at official high fidelity ...", flush=True)
        r = verify_boundary(e["boundary"], official=True)
        if r.error:
            print(f"  FAILED: {r.error[:200]}")
            continue
        score = r.metrics["p2_score"]
        print(f"  official score {score:.4f}, feasibility "
              f"{r.metrics['feasibility']:.4f}")
        rec = {"official_score": score, "metrics": r.metrics, "entry": e}
        if best is None or score > best["official_score"]:
            best = rec

    if best is None:
        sys.exit("no entry survived official verification")
    e = best["entry"]
    bank_dist = _nearest_bank_distance(e["boundary"])
    if bank_dist is not None and bank_dist < 1e-3:
        sys.exit(f"REFUSING export: boundary is a near-copy of a public seed-bank "
                 f"submission (max coefficient distance {bank_dist:.2e} < 1e-3). "
                 "Submitting it would plagiarize another user's result.")
    feas = best["metrics"]["feasibility"]
    kin = nearest_seed(e["boundary"])
    path = OUT / f"p2-{e['key']}-{best['official_score']:.4f}.json"
    path.write_text(json.dumps({
        "boundary": e["boundary"],                 # submission payload
        "provenance": {
            "task": "stellar_p2", "archive_key": e["key"],
            "code_sha": e.get("code_sha"), "archived_at": e.get("ts"),
            "official_score": best["official_score"],
            "official_feasibility": feas,
            "exported_at": round(time.time(), 1),
            "package": "constellaration==0.2.6",
            "nearest_public_seed_distance": bank_dist,
            # 2026-07-27: a distance number alone reads as "novel enough"; the
            # audit found 2.6e-3 co-existing with cosine 0.999989 to the #1 public
            # entry. Every export now carries the similarity AND the tolerance
            # position, so a submission can never be quoted without them.
            "nearest_public_seed": kin,
            "feasibility_tolerance": _TOL,
            "tolerance_used_fraction": round(feas / _TOL, 4),
            "score_at_equal_margin_estimate": None if not kin else round(
                best["official_score"] - _FEAS_SLOPE
                * (feas - kin["official_feasibility"]), 4),
            "disclosure": None if bank_dist is None else
                "REFINED FROM PUBLIC LEADERBOARD SUBMISSIONS "
                "(tasks/stellar_p2/seed_bank.json), not from scratch. Compare "
                "score_at_equal_margin_estimate against the nearest seed's score "
                "before claiming an improvement; see NOTICE.md and memory/"
                "stellar_p2/performance-analysis/."},
    }, indent=2))
    print(f"\nwrote {path}")
    if kin:
        print(f"  nearest public seed: {kin['user']} official {kin['official_score']:.4f} "
              f"at feasibility {kin['official_feasibility']:.5f}\n"
              f"  similarity to it: cosine {kin['cosine']:.6f}, "
              f"||delta||/||seed|| {kin['relative_norm']:.4%}\n"
              f"  tolerance used: {feas / _TOL:.0%} (seed used "
              f"{kin['official_feasibility'] / _TOL:.0%})")
    print("submission payload = the \"boundary\" object "
          "(SurfaceRZFourier JSON). Upload manually to the HF space.")


if __name__ == "__main__":
    main()
