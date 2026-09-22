"""Locations used by EvoHarness at runtime.

The repository contains a reviewed, immutable-ish record of earlier work. New
runs and downloaded task data belong under ``EVOHARNESS_HOME`` so installing
the package does not make the checkout itself a mutable application directory.
"""
from __future__ import annotations

import os
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def _checkout_root() -> Path | None:
    """Return a source checkout root when this package is running from one."""
    source_root = PACKAGE_ROOT.parent.parent
    if (source_root / "pyproject.toml").is_file() and (source_root / "src" / "evoharness").is_dir():
        return source_root
    current = Path.cwd().resolve()
    if (current / "pyproject.toml").is_file() and (current / "src" / "evoharness").is_dir():
        return current
    return None


_CHECKOUT_ROOT = _checkout_root()
REPO_ROOT = _CHECKOUT_ROOT or Path.cwd().resolve()
"""Source checkout root when available; otherwise the caller's working directory."""

_home = os.environ.get("EVOHARNESS_HOME")
_default_state = (REPO_ROOT / ".local" if _CHECKOUT_ROOT else
                  Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) /
                  "evoharness")
STATE_ROOT = (Path(_home).expanduser() if _home else _default_state).resolve()
RUNS_DIR = STATE_ROOT / "runs"
MEMORY_DIR = STATE_ROOT / "memory"
DATA_DIR = STATE_ROOT / "data"
ARCHIVE_DIR = STATE_ROOT / "archive"


def _safe_name(name: str) -> str:
    """Validate the single task-name component accepted by task helpers."""
    path = Path(name)
    if not name or path.name != name or name in {".", ".."}:
        raise ValueError(f"task name must be one path component, got {name!r}")
    return name


def task_data(name: str) -> Path:
    """Return the private runtime data directory for *name* without creating it."""
    return DATA_DIR / _safe_name(name)


def task_memory(name: str) -> Path:
    """Return the private runtime memory directory for *name* without creating it."""
    return MEMORY_DIR / _safe_name(name)


_RUNTIME_ROOTS = {
    "runs": RUNS_DIR,
    "memory": MEMORY_DIR,
    "data": DATA_DIR,
    "archive": ARCHIVE_DIR,
}
_SOURCE_ROOTS = {
    "wiki": STATE_ROOT / "notebooks",
    "notebooks": STATE_ROOT / "notebooks",
    "plans": STATE_ROOT / "plans",
    "reference": STATE_ROOT / "reference",
    "experiments": ARCHIVE_DIR / "experiments",
    "docs": ARCHIVE_DIR / "docs",
    "external": STATE_ROOT / "external",
    "hf_refresh": STATE_ROOT / "leaderboard",
}


def _migrated_module_path(candidate: Path) -> Path | None:
    """Map one legacy Python module to its active packaged implementation."""
    if candidate.suffix != ".py" or candidate.parts[0] not in {"core", "axes", "tasks", "experiments", "scripts"}:
        return None
    module = ".".join(candidate.with_suffix("").parts)
    migration = ARCHIVE_DIR / "module-migration.json"
    try:
        destinations = json.loads(migration.read_text())
        destination = destinations.get(module)
    except (OSError, json.JSONDecodeError):
        destination = None
    if destination and destination.startswith("evoharness."):
        mapped = PACKAGE_ROOT.joinpath(*destination.split(".")[1:]).with_suffix(".py")
        if mapped.is_file():
            return mapped
    if candidate.parts[0] == "tasks":
        mapped = PACKAGE_ROOT / candidate
        if mapped.is_file():
            return mapped
    if candidate.parts[0] == "axes":
        mapped = PACKAGE_ROOT / "strategies" / Path(*candidate.parts[1:])
        if mapped.is_file():
            return mapped
    return None


def resolve_artifact(path: str | Path) -> Path:
    """Resolve a runtime artifact or a checked-in historical/source artifact.

    Runtime paths use the state root. Historical notebooks, plans, references,
    and archived experiment scripts are mapped to their preserved locations.
    A packaged task/resource is preferred over a checkout path. The function
    never creates directories; use :func:`task_data` or :func:`task_memory`.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(REPO_ROOT)
        except ValueError:
            return candidate.resolve()
    if any(part == ".." for part in candidate.parts):
        raise ValueError(f"artifact path must not escape its root: {path!r}")
    if not candidate.parts:
        raise ValueError("artifact path must not be empty")

    head, *tail = candidate.parts
    remainder = Path(*tail)
    if head in _RUNTIME_ROOTS:
        runtime = _RUNTIME_ROOTS[head] / remainder
        archived = ARCHIVE_DIR / candidate
        return runtime if runtime.exists() or not archived.exists() else archived
    migrated = _migrated_module_path(candidate)
    if migrated is not None:
        return migrated
    if head in _SOURCE_ROOTS:
        source = _SOURCE_ROOTS[head] / remainder
        return source if source.exists() else ARCHIVE_DIR / "source" / candidate

    archived_source = ARCHIVE_DIR / "source" / candidate
    if archived_source.exists():
        return archived_source

    packaged = PACKAGE_ROOT / candidate
    return packaged if packaged.exists() else REPO_ROOT / candidate
