"""In-image one-shot: evaluate a boundary JSON with the pinned forward model
and print the full metric set as JSON (works for infeasible boundaries too,
unlike verify_boundary's sparse clean-room metrics).

  docker run --rm --cpus 1 --memory 3g --user $(id -u):$(id -g) -e HOME=/tmp \
    -v $ROOT:/work -w /work evoharness-stellar-eval \
    python -m evoharness.tasks.stellar_p2.physics.oracle <boundary.json> [fidelity]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("boundary", type=Path)
    parser.add_argument("fidelity", nargs="?", default="very_low_fidelity")
    args = parser.parse_args()
    payload = json.loads(args.boundary.read_text())
    boundary = payload.get("boundary", payload)
    from evoharness.tasks.stellar_p2.physics.metrics.difftest import run_oracle
    try:
        m, _ = run_oracle(boundary, args.fidelity)
        print(json.dumps({
            "worst": m["feasibility"],
            "L": m["minimum_normalized_magnetic_gradient_scale_length"],
            "qi": m["qi"], "aspect": m["aspect_ratio"],
            "violations": list(map(float, m["violations"])),
        }))
    except Exception as e:
        print(json.dumps({"worst": None, "error": f"{type(e).__name__}: {e}"[:200]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
