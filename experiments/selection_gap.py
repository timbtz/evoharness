"""Selection-gap diagnostic (Rekursiv AI: generation vs selection).

For each run: the best candidate the run EVALUATED (oracle over all scored
candidates, accepted or not) vs the one its gate/selection actually KEPT
(best accepted, by the run-end combined train+val metric). A large gap means
the writers generate winners the selection throws away — invest in selection
(gate, noise handling). A near-zero gap means generation is the bottleneck —
invest in diversity/feedback.

Usage: uv run python experiments/selection_gap.py [run_id ...]  (default: all cvrp runs)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def combined(scores: dict) -> float:
    t = scores.get("train", float("-inf"))
    v = scores.get("val")
    return (t + v) / 2 if v is not None else t


def analyze(run_dir: Path) -> None:
    ledger = run_dir / "ledger.jsonl"
    if not ledger.exists():
        return
    cands = [json.loads(l) for l in ledger.read_text().splitlines()
             if '"candidate"' in l]
    cands = [c for c in cands if c.get("type") == "candidate"
             and c.get("scores", {}).get("train", float("-inf")) > float("-inf")]
    if not cands:
        return
    oracle = max(cands, key=lambda c: combined(c["scores"]))
    kept = [c for c in cands if c.get("accepted")]
    if not kept:
        print(f"{run_dir.name}: n={len(cands)} accepted=0 "
              f"oracle={oracle['id']} {combined(oracle['scores']):.4f}")
        return
    selected = max(kept, key=lambda c: combined(c["scores"]))
    gap = combined(oracle["scores"]) - combined(selected["scores"])
    print(f"{run_dir.name}: n={len(cands)} accepted={len(kept)} "
          f"oracle={oracle['id']} {combined(oracle['scores']):.4f} "
          f"selected={selected['id']} {combined(selected['scores']):.4f} "
          f"gap={gap:.4f}{'  <-- selection loses value' if gap > 0.02 else ''}")


if __name__ == "__main__":
    runs = sys.argv[1:] or sorted(
        d.name for d in (ROOT / "runs").iterdir()
        if d.is_dir() and d.name.startswith("cvrp") and (d / "ledger.jsonl").exists())
    for r in runs:
        analyze(ROOT / "runs" / r)
