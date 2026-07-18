"""Download the 15 CVRPLIB instances used by the cvrp task (all four splits).

Sets A/B/P (Augerat et al., 1995, proven optima) come from the CVRPLIB
mirror at galgos.inf.puc-rio.br, addressed by numeric instance id (the
site has no stable name-based URL). Set X (Uchoa et al., 2017) comes from
the PyVRP/Instances GitHub mirror, addressed by name.

Re-runnable: instances/ is gitignored, so run this after a fresh clone:
    uv run python tasks/cvrp/fetch.py
"""
from pathlib import Path
from urllib.request import urlopen

GALGOS = "https://galgos.inf.puc-rio.br/cvrplib/en/download/{}/{}"
# name -> CVRPLIB instance id (scraped from the galgos instance listing page)
GALGOS_IDS = {
    "A-n32-k5": 4, "A-n37-k6": 11, "A-n45-k6": 16, "A-n60-k9": 23,
    "B-n45-k5": 39, "B-n50-k7": 41,
    "P-n55-k10": 90, "P-n65-k10": 94,
}
PYVRP_URL = "https://raw.githubusercontent.com/PyVRP/Instances/main/CVRP/{}.{}"
PYVRP_NAMES = [
    "X-n101-k25", "X-n110-k13", "X-n125-k30", "X-n153-k22",
    "X-n200-k36", "X-n251-k28", "X-n303-k21",
]


def _fetch(url: str, dest: Path) -> None:
    data = urlopen(url, timeout=30).read()
    dest.write_bytes(data)
    print(f"fetched {dest} ({len(data)} bytes)")


def _sanity_check(vrp_path: Path) -> None:
    text = vrp_path.read_text()
    dim = next(l for l in text.splitlines() if l.startswith("DIMENSION"))
    ewt = next(l for l in text.splitlines() if l.startswith("EDGE_WEIGHT_TYPE"))
    assert int(dim.split(":")[1].strip()) > 0, f"bad DIMENSION in {vrp_path}"
    assert "EUC_2D" in ewt, f"expected EUC_2D in {vrp_path}, got {ewt!r}"


def main() -> None:
    out = Path(__file__).parent / "instances"
    out.mkdir(exist_ok=True)

    for name, iid in GALGOS_IDS.items():
        _fetch(GALGOS.format("instance", iid), out / f"{name}.vrp")
        _fetch(GALGOS.format("bks", iid), out / f"{name}.sol")

    for name in PYVRP_NAMES:
        _fetch(PYVRP_URL.format(name, "vrp"), out / f"{name}.vrp")
        _fetch(PYVRP_URL.format(name, "sol"), out / f"{name}.sol")

    for name in list(GALGOS_IDS) + PYVRP_NAMES:
        _sanity_check(out / f"{name}.vrp")
    print(f"sanity check OK for all {len(GALGOS_IDS) + len(PYVRP_NAMES)} instances")


if __name__ == "__main__":
    main()
