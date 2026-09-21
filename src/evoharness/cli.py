"""The public command-line interface for EvoHarness.

Workflow modules are deliberately imported only after their command has been
selected.  This keeps ``evoharness --help``, task configuration, and doctor
usable on a lightweight machine that does not have the stellarator stack.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import runpy
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .paths import DATA_DIR, MEMORY_DIR, PACKAGE_ROOT, REPO_ROOT, RUNS_DIR, STATE_ROOT


_WORKFLOWS: dict[str, tuple[str, ...]] = {
    "repair": ("evoharness.tasks.stellar_p2.workflows.repair",),
    "discover": ("evoharness.tasks.stellar_p2.workflows.discover",),
    "research": ("evoharness.tasks.stellar_p2.workflows.research",),
    "evaluate": ("evoharness.tasks.stellar_p2.workflows.evaluate",),
    "factory": ("evoharness.tasks.stellar_p2.workflows.factory",),
    "polish": ("evoharness.tasks.stellar_p2.physics.polish",),
    "export": ("evoharness.tasks.stellar_p2.workflows.export",),
}
_ANALYSIS: dict[str, tuple[str, ...]] = {
    "report": ("evoharness.analysis.report",),
    "screen": ("evoharness.analysis.screening",),
    "inventory": ("evoharness.analysis.build_run_inventory",),
    "figures": ("evoharness.analysis.render_figures",),
}


def _parse_value(raw: str) -> Any:
    """Accept JSON literals while leaving ordinary model names as strings."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _set_value(config: dict[str, Any], assignment: str) -> None:
    if "=" not in assignment:
        raise ValueError(f"override must use key=value: {assignment!r}")
    dotted, raw = assignment.split("=", 1)
    parts = dotted.split(".")
    if not all(parts):
        raise ValueError(f"override has an empty key component: {assignment!r}")
    target: dict[str, Any] = config
    for part in parts[:-1]:
        child = target.get(part)
        if child is None:
            child = {}
            target[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"cannot set {dotted!r}: {part!r} is not an object")
        target = child
    target[parts[-1]] = _parse_value(raw)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        content = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"config file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(content, dict):
        raise ValueError("config must contain one JSON object")
    return content


def _default_config_path(task: str) -> Path:
    """Find a public config from a checkout or from an installed wheel."""
    source_config = REPO_ROOT / "configs" / f"{task}.json"
    if source_config.is_file():
        return source_config
    bundled = files("evoharness").joinpath("resources", "configs", f"{task}.json")
    return Path(str(bundled))


def _engine() -> SimpleNamespace:
    """Load the public engine pieces without relying on package re-exports."""
    try:
        from evoharness.engine.config import Config, OBJECTIVES, SWITCHES, TASKS
        from evoharness.engine.loop import run
        return SimpleNamespace(Config=Config, OBJECTIVES=OBJECTIVES, SWITCHES=SWITCHES,
                               TASKS=TASKS, run=run)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("evoharness"):
            raise RuntimeError("the installed package is missing evoharness.engine") from exc
        raise


def _run_module(candidates: Sequence[str], forwarded: Sequence[str]) -> int:
    """Run a workflow module with its original argv and help behavior."""
    module = next((name for name in candidates if importlib.util.find_spec(name)), None)
    if module is None:
        raise RuntimeError("workflow is not installed; expected one of: " + ", ".join(candidates))
    previous_argv = sys.argv
    sys.argv = [module, *forwarded]
    try:
        try:
            runpy.run_module(module, run_name="__main__", alter_sys=False)
        except SystemExit as exc:
            return int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        sys.argv = previous_argv
    return 0


def _command_fetch(args: argparse.Namespace) -> int:
    """Dispatch a task-data fetcher without running it merely to show help."""
    if any(value in {"-h", "--help"} for value in args.args):
        print(f"usage: evoharness fetch {args.task} [fetcher options]")
        return 0
    return _run_module((f"evoharness.tasks.{args.task}.fetch",), args.args)


def _delegated_tail(argv: Sequence[str], args: argparse.Namespace) -> list[str]:
    """Return workflow argv exactly as the user supplied it.

    ``argparse.REMAINDER`` and ``parse_known_args`` separate unknown options,
    which corrupts ordered workflow arguments such as ``--exclude-family x``.
    The public command grammar has a fixed prefix, so retain the raw suffix.
    """
    if args.command in {"fetch", "stellarator", "physics"}:
        return list(argv[2:])
    return list(argv[1:])


def _task_names() -> list[str]:
    engine = _engine()
    names = getattr(engine, "TASKS", ())
    return list(names) if names else ["binpacking", "circles", "tsp", "matmul_c", "cvrp", "stellar_p2"]


def _command_tasks(args: argparse.Namespace) -> int:
    names = _task_names()
    if args.json:
        print(json.dumps({"tasks": names}, indent=2))
    else:
        print("\n".join(names))
    return 0


def _command_axes(args: argparse.Namespace) -> int:
    engine = _engine()
    payload = {
        "switches": getattr(engine, "SWITCHES", {}),
        "objectives": getattr(engine, "OBJECTIVES", ()),
        "execution": ("task_default", "require_container"),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    for name, values in payload["switches"].items():
        print(f"{name}: {', '.join(values)}")
    print("objectives: " + ", ".join(payload["objectives"]))
    print("execution: task_default, require_container (orthogonal runtime policy)")
    return 0


def _command_doctor(args: argparse.Namespace) -> int:
    """Perform only local availability checks; do not instantiate providers."""
    from evoharness.engine.llm import load_env
    load_env()
    task_specs = {}
    for name in ("binpacking", "circles", "tsp", "matmul_c", "cvrp", "stellar_p2"):
        module = f"evoharness.tasks.{name}.task"
        task_specs[name] = importlib.util.find_spec(module) is not None
    images = {}
    for image in ("evoharness-benchmark-eval", "evoharness-cvrp-eval", "evoharness-stellar-eval"):
        try:
            images[image] = subprocess.run(["docker", "image", "inspect", image],
                capture_output=True, timeout=5).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            images[image] = False
    checks: dict[str, Any] = {
        "evaluator_images": images,
        "gcc": shutil.which("gcc") is not None,
        "downloaded_data": {name: len(list((DATA_DIR / name).glob(pattern)))
                            for name, pattern in (("tsp", "*.tsp"), ("cvrp", "*.vrp"))},
        "package": PACKAGE_ROOT.is_dir(),
        "repository": REPO_ROOT.is_dir(),
        "state_root": str(STATE_ROOT),
        "runtime_directories": {"runs": str(RUNS_DIR), "memory": str(MEMORY_DIR),
                                "data": str(DATA_DIR)},
        "docker": shutil.which("docker") is not None,
        "uvicorn": importlib.util.find_spec("uvicorn") is not None,
        "zai_key_configured": bool(os.environ.get("ZAI_API_KEY")),
        "tasks": task_specs,
    }
    if args.json:
        print(json.dumps(checks, indent=2, sort_keys=True))
    else:
        print(f"package: {'ok' if checks['package'] else 'missing'} ({PACKAGE_ROOT})")
        print(f"state root: {STATE_ROOT} (created on first run)")
        print(f"docker: {'found' if checks['docker'] else 'not found'}")
        print(f"uvicorn: {'found' if checks['uvicorn'] else 'not found'}")
        print("model key: " + ("configured" if checks["zai_key_configured"] else "not configured"))
        print("evaluator images: " + ", ".join(f"{name}={'ready' if ready else 'absent'}"
                                                for name, ready in images.items()))
        print("downloaded instances: " + ", ".join(f"{name}={count}"
                                                    for name, count in checks["downloaded_data"].items()))
        print("task modules: " + ", ".join(
            f"{name}={'ok' if available else 'missing'}" for name, available in task_specs.items()))
    return 0 if checks["package"] else 1


def _command_run(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else _default_config_path(args.task or "binpacking")
    payload = _load_config(config_path)
    for assignment in args.set:
        _set_value(payload, assignment)
    if args.task is not None:
        payload["task"] = args.task
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.execution is not None:
        payload["execution"] = args.execution
    for key, value in (("max_usd", args.max_usd), ("max_calls", args.max_calls),
                       ("max_seconds", args.max_seconds)):
        if value is not None:
            payload.setdefault("budget", {})[key] = value

    engine = _engine()
    config_type = getattr(engine, "Config", None)
    runner = getattr(engine, "run", None)
    if config_type is None or runner is None:
        raise RuntimeError("evoharness.engine must export Config and run")
    config = config_type.from_dict(payload)
    run_dir = Path(args.run_dir).expanduser() if args.run_dir else (
        RUNS_DIR / f"{config.task}-s{config.seed}-{int(time.time() * 1000) % 10**8}")
    if args.dry_run:
        print(json.dumps({"config": config.to_dict(), "run_dir": str(run_dir)}, indent=2))
        return 0
    run_dir.mkdir(parents=True, exist_ok=False)
    result = runner(config, run_dir=run_dir)
    print(json.dumps(result, indent=2, default=str))
    return 0


def _command_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise RuntimeError("serve requires the uvicorn dependency") from exc
    uvicorn.run("evoharness.app.server:app", host=args.host, port=args.port,
                reload=args.reload, log_level=args.log_level)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evoharness", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    tasks = sub.add_parser("tasks", help="list available optimization tasks")
    tasks.add_argument("--json", action="store_true")
    tasks.set_defaults(handler=_command_tasks)

    axes = sub.add_parser("axes", help="show configurable search axes")
    axes.add_argument("--json", action="store_true")
    axes.set_defaults(handler=_command_axes)

    doctor = sub.add_parser("doctor", help="check local installation without calling providers")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(handler=_command_doctor)

    run = sub.add_parser("run", help="run a JSON configuration")
    run.add_argument("--config", help="configuration JSON (defaults to configs/<task>.json)")
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                     help="override a JSON value; nested keys use dots")
    run.add_argument("--task", help="override config task")
    run.add_argument("--seed", type=int, help="override config seed")
    run.add_argument("--execution", choices=("task_default", "require_container"),
                     help="override the task execution policy")
    run.add_argument("--max-usd", type=float, help="override budget.max_usd")
    run.add_argument("--max-calls", type=int, help="override budget.max_calls")
    run.add_argument("--max-seconds", type=float, help="override budget.max_seconds")
    run.add_argument("--run-dir", help="new directory for this run")
    run.add_argument("--dry-run", action="store_true", help="validate and print the resolved run")
    run.set_defaults(handler=_command_run)

    serve = sub.add_parser("serve", help="serve the local web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(handler=_command_serve)

    fetch = sub.add_parser("fetch", help="download task data", add_help=False)
    fetch.add_argument("task", choices=("tsp", "cvrp"))
    fetch.add_argument("args", nargs=argparse.REMAINDER)
    fetch.set_defaults(handler=_command_fetch)

    stellar = sub.add_parser("stellarator", help="run stellarator workflows")
    stellar_sub = stellar.add_subparsers(dest="stellar_command", required=True)
    for name in _WORKFLOWS:
        workflow = stellar_sub.add_parser(name, help=f"delegate to stellarator {name} workflow",
                                          add_help=False)
        workflow.add_argument("args", nargs=argparse.REMAINDER)
        workflow.set_defaults(handler=lambda a, n=name: _run_module(_WORKFLOWS[n], a.args))

    for name in ("report", "screen", "inventory", "figures"):
        analysis = sub.add_parser(name, help=f"delegate to {name} analysis", add_help=False)
        analysis.add_argument("args", nargs=argparse.REMAINDER)
        analysis.set_defaults(handler=lambda a, n=name: _run_module(_ANALYSIS[n], a.args))

    physics = sub.add_parser("physics", help="physics workflow aliases")
    physics_sub = physics.add_subparsers(dest="physics_command", required=True)
    polish = physics_sub.add_parser("polish", help="run the stellarator polish campaign",
                                    add_help=False)
    polish.add_argument("args", nargs=argparse.REMAINDER)
    polish.set_defaults(handler=lambda a: _run_module(_WORKFLOWS["polish"], a.args))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args, unknown = parser.parse_known_args(raw_argv)
        if unknown:
            if not hasattr(args, "args"):
                parser.error("unrecognized arguments: " + " ".join(unknown))
        if hasattr(args, "args"):
            args.args = _delegated_tail(raw_argv, args)
        return int(args.handler(args))
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2  # argparse.error raises SystemExit; this is only for type checkers.


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
