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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=3,
                    help="verify the N best archive entries by shaped score")
    ap.add_argument("--min-shaped", type=float, default=-0.5,
                    help="ignore archive entries below this shaped score")
    args = ap.parse_args()

    if not _ARCHIVE.exists():
        sys.exit(f"no archive at {_ARCHIVE}")
    entries, seen = [], set()
    for line in _ARCHIVE.read_text().splitlines():
        e = json.loads(line)
        if e.get("shaped", -9e9) >= args.min_shaped and e["key"] not in seen:
            seen.add(e["key"])
            entries.append(e)
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
    path = OUT / f"p2-{e['key']}-{best['official_score']:.4f}.json"
    path.write_text(json.dumps({
        "boundary": e["boundary"],                 # submission payload
        "provenance": {
            "task": "stellar_p2", "archive_key": e["key"],
            "code_sha": e.get("code_sha"), "archived_at": e.get("ts"),
            "official_score": best["official_score"],
            "official_feasibility": best["metrics"]["feasibility"],
            "exported_at": round(time.time(), 1),
            "package": "constellaration==0.2.6",
            "nearest_public_seed_distance": bank_dist,
            "disclosure": None if bank_dist is None else
                "search may be seeded from public leaderboard submissions "
                "(see tasks/stellar_p2/seed_bank.json) — refined, not from scratch"},
    }, indent=2))
    print(f"\nwrote {path}\nsubmission payload = the \"boundary\" object "
          "(SurfaceRZFourier JSON). Upload manually to the HF space.")


if __name__ == "__main__":
    main()
