"""Versioned, JSON-safe description of the effective inputs to a run."""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evoharness.engine.config import Config


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree_hash(path: Path) -> str | None:
    """Content hash of a tree, including relative names but not mtimes."""
    if not path.exists():
        return None
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    for file in files:
        rel = file.name if path.is_file() else file.relative_to(path).as_posix()
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_state(root: Path) -> dict[str, Any]:
    """Capture the commit and a content-based fingerprint of local changes."""
    def git(*args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, timeout=10)

    try:
        commit_result = git("rev-parse", "HEAD")
        if commit_result.returncode:
            return {"commit": None, "dirty": None, "dirty_fingerprint": None}
        commit = commit_result.stdout.decode().strip()
        status = git("status", "--porcelain=v1", "--untracked-files=all")
        diff = git("diff", "--binary", "HEAD", "--")
        untracked = git("ls-files", "--others", "--exclude-standard", "-z")
        status_bytes = status.stdout if status.returncode == 0 else b""
        diff_bytes = diff.stdout if diff.returncode == 0 else b""
        untracked_bytes = b""
        if untracked.returncode == 0:
            for raw_name in sorted(filter(None, untracked.stdout.split(b"\0"))):
                file = root / raw_name.decode(errors="surrogateescape")
                if file.is_file():
                    untracked_bytes += raw_name + b"\0" + file.read_bytes() + b"\0"
        return {
            "commit": commit,
            "dirty": bool(status_bytes),
            "dirty_fingerprint": _sha256(
                status_bytes + b"\0" + diff_bytes + b"\0" + untracked_bytes),
        }
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "dirty": None, "dirty_fingerprint": None}


@dataclass(frozen=True)
class ExperimentSpec:
    """Effective run inputs stored in the first ledger event."""

    schema_version: int
    code: dict[str, Any]
    config: dict[str, Any]
    task: dict[str, Any]
    memory: dict[str, Any]

    @classmethod
    def capture(cls, cfg: Config, task: Any, root: Path) -> "ExperimentSpec":
        task_spec = getattr(task, "experiment_spec", lambda: {})()
        from evoharness.paths import task_memory
        memory_dir = task_memory(cfg.task)
        return cls(
            schema_version=1,
            code=_git_state(root),
            config=cfg.to_dict(),
            task={"name": cfg.task, **task_spec},
            memory={
                "path": str(memory_dir),
                "snapshot_hash": _tree_hash(memory_dir),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "code": self.code,
            "config": self.config,
            "task": self.task,
            "memory": self.memory,
        }
