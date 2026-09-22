"""Download the five TSPLIB instances used by the tsp task's private split.

Re-runnable: Downloaded data under .local/data/ is gitignored, so run this after a fresh clone:
    uv run evoharness fetch tsp
"""
from pathlib import Path
from evoharness.paths import task_data
from urllib.request import urlopen

NAMES = ["eil51", "berlin52", "st70", "kroA100", "ch150"]
URL = "https://raw.githubusercontent.com/mastqe/tsplib/master/{}.tsp"


def main() -> None:
    out = task_data("tsp")
    out.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        data = urlopen(URL.format(name), timeout=30).read()
        dest = out / f"{name}.tsp"
        dest.write_bytes(data)
        print(f"fetched {dest} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
