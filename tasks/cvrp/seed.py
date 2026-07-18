"""Assembler for the CVRP seed artifact. The seed's real sources live in
seed/kernel.c (C local-search + LNS engine) and seed/solver.py (Python policy);
seed_code() stitches them into the single self-contained module that evolution
operates on: solver.py with the C source injected as its C_KERNEL string."""
from pathlib import Path

_SRC = Path(__file__).parent / "seed"


def seed_code() -> str:
    c = (_SRC / "kernel.c").read_text()
    py = (_SRC / "solver.py").read_text()
    assert '"""' not in c and not c.rstrip().endswith("\\"), "kernel.c breaks r-string embedding"
    assert py.count("C_KERNEL = __KERNEL_C__") == 1, "solver.py placeholder missing"
    return py.replace("C_KERNEL = __KERNEL_C__", 'C_KERNEL = r"""\n' + c + '"""', 1)
