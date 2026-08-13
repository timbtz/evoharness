"""Cross-process lock ensuring one memory-heavy Stellar evaluator per host."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path


class CampaignLock:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close(); self.handle = None
            raise RuntimeError(f"another Stellar physics campaign holds {self.path}")
        self.handle.seek(0); self.handle.truncate()
        self.handle.write(str(os.getpid())); self.handle.flush()
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close(); self.handle = None

