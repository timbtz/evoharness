"""Download the five TSPLIB instances used by the tsp task's private split.

Re-runnable: instances/ is gitignored, so run this after a fresh clone:
    uv run python tasks/tsp/fetch.py
"""
from pathlib import Path
from urllib.request import urlopen

NAMES = ["eil51", "berlin52", "st70", "kroA100", "ch150"]
URL = "https://raw.githubusercontent.com/mastqe/tsplib/master/{}.tsp"


def main() -> None:
    out = Path(__file__).parent / "instances"
    out.mkdir(exist_ok=True)
    for name in NAMES:
        data = urlopen(URL.format(name), timeout=30).read()
        dest = out / f"{name}.tsp"
        dest.write_bytes(data)
        print(f"fetched {dest} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
