"""Evaluate nae2 batch candidates at very_low_fidelity (host-side, sequential).

  python -m evoharness.tasks.stellar_p2.proposals.evaluate_near_axis \
      --batch .local/runs/nae2-batch1 \
      [--names a,b,c] [--limit 8]

One clean-room container at a time (cpus=1) — box courtesy: a polish loop
(2 cpus) and the zai persistent loop (1 cpu) are usually live.
Records per candidate: score, feasibility, violations, objective metric and
the nfp-comparable product bookkeeping (metric, metric_per_nfp).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evoharness.paths import RUNS_DIR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, default=RUNS_DIR / "nae2-batch1")
    ap.add_argument("--names", default="")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--fidelity", default="very_low_fidelity")
    a = ap.parse_args()

    from evoharness.tasks.stellar_p2.task import verify_boundary

    batch = a.batch.resolve()
    files = sorted((batch / "candidates").glob("*.json"))
    if a.names:
        wanted = set(a.names.split(","))
        files = [f for f in files if f.stem in wanted]
    files = files[: a.limit]

    out_p = batch / "results.json"
    results = json.loads(out_p.read_text()) if out_p.exists() else {}
    for f in files:
        if f.stem in results:
            continue
        payload = json.loads(f.read_text())
        t0 = time.time()
        res = verify_boundary(payload["boundary"], official=False,
                              fidelity=a.fidelity)
        m = res.metrics or {}
        nfp = payload["boundary"]["n_field_periods"]
        metric = m.get("minimum_normalized_magnetic_gradient_scale_length")
        row = {
            "meta": payload.get("meta", {}),
            "score": res.score,
            "error": res.error,
            "seconds": round(time.time() - t0, 1),
            "feasibility": m.get("feasibility"),
            "violations": m.get("violations"),
            "objective_metric": metric,
            "metric_per_nfp": (metric / nfp) if metric is not None else None,
            "qi": m.get("qi"),
            "aspect": m.get("aspect_ratio"),
            "elongation": m.get("max_elongation"),
            "mirror": m.get("edge_magnetic_mirror_ratio"),
            "iota_edge_per_nfp": m.get(
                "edge_rotational_transform_over_n_field_periods"),
        }
        results[f.stem] = row
        out_p.write_text(json.dumps(results, indent=2))
        print(f"{f.stem}: feas={row['feasibility']} score={res.score:.4f} "
              f"viol={row['violations']} [{row['seconds']}s]", flush=True)
    print(f"done, {len(results)} rows in {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
