"""Seed assembler: the evolved artifact is the single module seed/optimizer.py."""
from pathlib import Path


def seed_code() -> str:
    return (Path(__file__).parent / "seed" / "optimizer.py").read_text()
