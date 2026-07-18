"""Subprocess sandbox: timeout, memory cap, temp dir, best-effort network block.

Network isolation uses `unshare -rn` (user+net namespace) when the kernel allows it;
otherwise (Python only) a socket-stubbing preamble is prepended (good enough for the
PoC threat model: accidents, not adversaries).
"""
from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

_NO_NET_PREAMBLE = (
    "import socket as _s\n"
    "def _blocked(*a, **k): raise OSError('network disabled in sandbox')\n"
    "_s.socket = _s.create_connection = _s.getaddrinfo = _blocked\n"
)


def _unshare_works() -> bool:
    try:
        return subprocess.run(
            ["unshare", "-rn", "true"], capture_output=True, timeout=5
        ).returncode == 0
    except Exception:
        return False


_HAVE_UNSHARE = _unshare_works()
_ENV = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.environ.get("PYTHONPATH", "")}


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    returncode: int
    seconds: float
    timed_out: bool


def _run(cmd: list[str], cwd: str, timeout: float, mem_mb: int,
         unshare: bool = True) -> SandboxResult:
    if unshare and _HAVE_UNSHARE:
        cmd = ["unshare", "-rn"] + cmd

    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2**20,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 5,) * 2)

    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, preexec_fn=limits, start_new_session=True, env=_ENV,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        timed_out = True
    return SandboxResult(out, err, proc.returncode, time.monotonic() - t0, timed_out)


def run_python(code: str, timeout: float = 30.0, mem_mb: int = 2048) -> SandboxResult:
    """Run `code` as a python script in an isolated temp dir; return captured output."""
    tmp = tempfile.mkdtemp(prefix="evoh_")
    try:
        script = os.path.join(tmp, "main.py")
        with open(script, "w") as f:
            if not _HAVE_UNSHARE:
                f.write(_NO_NET_PREAMBLE)
            f.write(code)
        return _run([sys.executable, "-I", script], tmp, timeout, mem_mb)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_DOCKER_READY: dict[str, bool] = {}


def docker_image_ready(image: str, dockerfile: str | None = None,
                       context: str | None = None) -> bool:
    """True iff docker works and `image` exists locally (built lazily from
    `dockerfile` on first miss). Cached per image; False => caller should fall back."""
    if image not in _DOCKER_READY:
        ok = False
        try:
            ok = subprocess.run(["docker", "image", "inspect", image],
                                capture_output=True, timeout=30).returncode == 0
            if not ok and dockerfile:
                ok = subprocess.run(
                    ["docker", "build", "-q", "-t", image, "-f", dockerfile,
                     context or os.path.dirname(dockerfile)],
                    capture_output=True, timeout=900).returncode == 0
        except Exception:
            ok = False
        _DOCKER_READY[image] = ok
    return _DOCKER_READY[image]


def run_python_docker(code: str, timeout: float = 30.0, mem_mb: int = 1024,
                      image: str = "python:3.13-slim",
                      cpuset: str | None = None) -> SandboxResult:
    """Run `code` as a python script in a throwaway container: no network, 1 CPU
    (optionally pinned via `cpuset` for stable wall-clock budgets), memory cap.
    Check docker_image_ready() first; a docker-level failure surfaces as rc != 0."""
    tmp = tempfile.mkdtemp(prefix="evoh_")
    name = f"evoh-{os.path.basename(tmp)}"
    try:
        with open(os.path.join(tmp, "main.py"), "w") as f:
            f.write(code)
        os.chmod(tmp, 0o755)
        cmd = ["docker", "run", "--rm", "--name", name, "--network", "none",
               "--cpus", "1", "--memory", f"{mem_mb}m", "--memory-swap", f"{mem_mb}m",
               "--pids-limit", "256", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges",
               "-v", f"{tmp}:/work:ro", "-w", "/work"]
        if cpuset is not None:
            cmd += ["--cpuset-cpus", cpuset]
        cmd += [image, "python3", "-I", "main.py"]
        t0 = time.monotonic()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        try:  # +15s: container start/stop overhead is outside the script's budget
            out, err = proc.communicate(timeout=timeout + 15.0)
            timed_out = False
        except subprocess.TimeoutExpired:
            try:
                subprocess.run(["docker", "kill", name], capture_output=True, timeout=30)
            except Exception:
                pass  # wedged daemon: fall through to killing the client
            try:
                out, err = proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
            timed_out = True
        return SandboxResult(out, err, proc.returncode, time.monotonic() - t0, timed_out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_c(code: str, timeout: float = 30.0, mem_mb: int = 2048,
          cflags: tuple[str, ...] = ("-O3", "-march=native")) -> SandboxResult:
    """Compile `code` with gcc (fixed flags) and run the binary in the jail.
    A compile failure is returned as the compiler's SandboxResult (rc != 0)."""
    tmp = tempfile.mkdtemp(prefix="evoh_")
    try:
        with open(os.path.join(tmp, "main.c"), "w") as f:
            f.write(code)
        cc = _run(["gcc", *cflags, "-o", "main", "main.c", "-lm"], tmp, 60.0, mem_mb)
        if cc.returncode != 0 or cc.timed_out:
            return cc
        return _run(["./main"], tmp, timeout, mem_mb)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
