"""VMEC `input.*` (INDATA namelist) -> our boundary dict convention.

Used to import PUBLIC stellarator boundaries (Zenodo supplementary data,
simsopt test files) into the P2 evaluator. Provenance is mandatory: every
boundary written by this script carries the source URL/DOI and the original
file name — these configurations are other people's published work.

Convention (matches experiments/desc_converter.py and the constellaration
SurfaceRZFourier JSON):
  r_cos/z_sin are (mpol+1) x (2*ntor+1); column j holds toroidal mode
  n = j - ntor. VMEC `RBC(n,m)` -> `r_cos[m][ntor+n]`, `ZBS(n,m)` ->
  `z_sin[m][ntor+n]`. Stellarator-symmetric surfaces set r_sin=z_cos=None
  and pin r_cos[0, n<0] = z_sin[0, n<=0] = 0.

A uniform scaling of R and Z is physics-null for every P2 metric (aspect,
elongation, mirror, edge iota, qi residual and the *normalized* magnetic
gradient scale length are all scale-invariant), so boundaries are rescaled
to R(0,0) = 1 by default to match the archive's gauge.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

_COEF = re.compile(
    r"\b(RBC|ZBS|RBS|ZBC)\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*=\s*"
    r"([+-]?\d*\.?\d+(?:[eEdD][+-]?\d+)?)",
    re.IGNORECASE,
)
# [^,!\n] and [ \t] (never \s): \s spans newlines, so a greedy value class
# would swallow the following lines and hide every later key.
_SCALAR = re.compile(r"^[ \t]*([A-Za-z_]+)[ \t]*=[ \t]*([^,!\n]+)",
                     re.MULTILINE)


def parse_indata(text: str) -> dict:
    """Pull NFP, LASYM and the boundary coefficient table out of an INDATA."""
    # strip Fortran comments so commented-out coefficients are never read
    lines = []
    for raw in text.splitlines():
        line = raw.split("!")[0]
        if line.strip():
            lines.append(line)
    body = "\n".join(lines)

    nfp = None
    lasym = False
    for key, val in _SCALAR.findall(body):
        k = key.upper()
        if k == "NFP":
            nfp = int(float(val.strip().split()[0]))
        elif k == "LASYM":
            lasym = val.strip().upper().lstrip(".")[:1] == "T"

    coeffs = {"RBC": {}, "ZBS": {}, "RBS": {}, "ZBC": {}}
    for kind, n, m, val in _COEF.findall(body):
        v = float(val.replace("d", "e").replace("D", "e"))
        coeffs[kind.upper()][(int(n), int(m))] = v
    if nfp is None:
        raise ValueError("NFP not found in INDATA")
    return {"nfp": nfp, "lasym": lasym, "coeffs": coeffs}


def to_boundary(text: str, normalize_r00: bool = True,
                max_m: int | None = None, max_n: int | None = None) -> dict:
    p = parse_indata(text)
    rbc, zbs = p["coeffs"]["RBC"], p["coeffs"]["ZBS"]
    if p["lasym"] or p["coeffs"]["RBS"] or p["coeffs"]["ZBC"]:
        raise NotImplementedError(
            "non-stellarator-symmetric input (LASYM=T or RBS/ZBC present)")
    if not rbc:
        raise ValueError("no RBC coefficients found")

    keys = set(rbc) | set(zbs)
    if max_m is not None:
        keys = {(n, m) for n, m in keys if m <= max_m}
    if max_n is not None:
        keys = {(n, m) for n, m in keys if abs(n) <= max_n}
    mpol = max(m for _, m in keys)
    ntor = max(abs(n) for n, _ in keys)

    rc = np.zeros((mpol + 1, 2 * ntor + 1))
    zs = np.zeros((mpol + 1, 2 * ntor + 1))
    for (n, m) in keys:
        rc[m, ntor + n] = rbc.get((n, m), 0.0)
        zs[m, ntor + n] = zbs.get((n, m), 0.0)

    scale = 1.0
    if normalize_r00:
        r00 = rc[0, ntor]
        if r00 == 0:
            raise ValueError("RBC(0,0) is zero — cannot normalize")
        scale = 1.0 / r00
        rc *= scale
        zs *= scale

    # stellarator symmetry pins these; VMEC files sometimes carry -0.0 there
    rc[0, :ntor] = 0.0
    zs[0, :ntor + 1] = 0.0

    return {
        "boundary": {
            "r_cos": rc.tolist(), "z_sin": zs.tolist(),
            "r_sin": None, "z_cos": None,
            "n_field_periods": int(p["nfp"]),
            "is_stellarator_symmetric": True,
        },
        "mpol": int(mpol), "ntor": int(ntor),
        "n_modes": int(np.count_nonzero(rc) + np.count_nonzero(zs)),
        "r00_original": float(rc[0, ntor] / scale),
        "scale_applied": float(scale),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_file")
    ap.add_argument("--out", required=True, help="boundary JSON to write")
    ap.add_argument("--source", default="", help="URL or DOI of the source")
    ap.add_argument("--name", default="", help="short config name")
    ap.add_argument("--note", default="", help="free-text provenance note")
    ap.add_argument("--max-m", type=int, default=None)
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--no-normalize", action="store_true")
    a = ap.parse_args()

    src = Path(a.input_file)
    res = to_boundary(src.read_text(errors="replace"),
                      normalize_r00=not a.no_normalize,
                      max_m=a.max_m, max_n=a.max_n)
    payload = {
        "boundary": res["boundary"],
        "provenance": {
            "name": a.name or src.name,
            "source": a.source,
            "original_file": src.name,
            "note": a.note,
            "license": "see source record; public scientific data — NOT ours",
            "converter": "experiments/vmec_input_to_boundary.py",
            "mpol": res["mpol"], "ntor": res["ntor"],
            "n_nonzero_modes": res["n_modes"],
            "r00_original": res["r00_original"],
            "scale_applied": res["scale_applied"],
        },
    }
    Path(a.out).write_text(json.dumps(payload, indent=1))
    print(json.dumps({"out": a.out, "nfp": res["boundary"]["n_field_periods"],
                      "mpol": res["mpol"], "ntor": res["ntor"],
                      "modes": res["n_modes"],
                      "r00_original": res["r00_original"]}))


if __name__ == "__main__":
    main()
