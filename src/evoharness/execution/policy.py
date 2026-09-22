"""Per-run execution policy, scoped safely across concurrent API threads."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

POLICIES = ("task_default", "require_container")
BENCHMARK_IMAGE = "evoharness-benchmark-eval"
_CURRENT = ContextVar("evoharness_execution", default="task_default")


def current_policy() -> str:
    return _CURRENT.get()


def benchmark_image() -> str:
    from .sandbox import docker_image_ready
    dockerfile = Path(__file__).with_name("Dockerfile.eval")
    if not docker_image_ready(BENCHMARK_IMAGE, str(dockerfile)):
        raise RuntimeError("Container execution required: build evoharness-benchmark-eval "
                           "with src/evoharness/execution/Dockerfile.eval and check Docker access.")
    return BENCHMARK_IMAGE


@contextmanager
def execution_policy(policy: str, task: str | None = None):
    if policy not in POLICIES:
        raise ValueError(f"unknown execution policy {policy!r}; expected {POLICIES}")
    if task == "stellar_p2":
        from evoharness.paths import PACKAGE_ROOT
        from .sandbox import docker_image_ready
        dockerfile = PACKAGE_ROOT / "tasks" / "stellar_p2" / "Dockerfile.eval"
        if not docker_image_ready("evoharness-stellar-eval", str(dockerfile)):
            raise RuntimeError("Container execution required: stellar_p2 needs the "
                               "evoharness-stellar-eval image and Docker daemon access.")
    elif policy == "require_container":
        benchmark_image()  # Fail before any model call if the required backend is absent.
    token = _CURRENT.set(policy)
    try:
        yield
    finally:
        _CURRENT.reset(token)
