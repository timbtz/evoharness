"""Small persistent multi-agent research harness for structural discovery.

This intentionally does not encode stellarator hypotheses. Agents may inspect
the whole repository, create experiments in isolated worktrees, and change the
research direction. The harness owns only persistence, portfolio diversity,
process isolation, and evidence/provenance contracts.
"""
from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


ROLES = {"explorer", "experimentalist", "analyst", "referee", "replicator"}
BACKENDS = {"claude", "codex", "zai"}
from evoharness.paths import REPO_ROOT, resolve_artifact

ARTIFACT_ROOT = REPO_ROOT
_ARTIFACT_REF = re.compile(r"(?<![\w.-])((?:src|reports|configs|\.local|core|experiments|runs|memory|plans|tests)/[^\s`'\";,\)]+)")


def parse_backend_minimums(value: str) -> dict[str, int]:
    """Parse an enforceable worker allocation such as ``zai=4,claude=1``."""
    result: dict[str, int] = {}
    for item in (part.strip() for part in value.split(",")):
        if not item:
            continue
        if "=" not in item:
            raise ValueError("backend minimums must use backend=count")
        backend, count_text = (part.strip() for part in item.split("=", 1))
        if backend not in BACKENDS or backend in result:
            raise ValueError(f"invalid or duplicate backend minimum: {backend}")
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(f"invalid backend count: {count_text}") from exc
        if count < 1:
            raise ValueError("backend minimum counts must be positive")
        result[backend] = count
    return result


def validate_backend_minimums(tasks: Iterable["AgentTask"],
                              minimums: Mapping[str, int]) -> None:
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.backend] = counts.get(task.backend, 0) + 1
    missing = {backend: count - counts.get(backend, 0)
               for backend, count in minimums.items()
               if counts.get(backend, 0) < count}
    if missing:
        raise ValueError(f"coordinator missed backend worker minimums: {missing}")


def artifact_claims(text: str) -> tuple[list[str], list[str]]:
    references = []
    for match in _ARTIFACT_REF.findall(text):
        if any(char in match for char in "*?[]{}"):
            continue
        path = match.rstrip(".:]")
        path = re.sub(r":\d+(?:-\d+)?$", "", path)
        path = re.sub(r"(?<=\.py):[A-Za-z_][\w.]*$", "", path)
        references.append(path)
    references = sorted(set(references))
    verified = [path for path in references if resolve_artifact(ARTIFACT_ROOT / path).is_file()]
    # Existing directories are navigation hints, not immutable evidence and
    # not missing claims. A report needs a concrete file to earn verification.
    missing = [path for path in references if not resolve_artifact(ARTIFACT_ROOT / path).exists()]
    return verified, missing


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class AgentTask:
    hypothesis_id: str
    title: str
    role: str
    backend: str
    instructions: str
    read_only: bool = True
    depends_on: tuple[str, ...] = ()
    requires_repository: bool = True
    task_id: str = ""

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown research role: {self.role}")
        if self.backend not in BACKENDS:
            raise ValueError(f"unknown agent backend: {self.backend}")
        if not self.hypothesis_id.strip() or not self.title.strip() or not self.instructions.strip():
            raise ValueError("task hypothesis, title, and instructions are required")

    @classmethod
    def parse(cls, row: dict[str, Any]) -> "AgentTask":
        allowed = {"hypothesis_id", "title", "role", "backend", "instructions",
                   "read_only", "depends_on", "requires_repository"}
        if set(row) - allowed:
            raise ValueError(f"unexpected task fields: {sorted(set(row) - allowed)}")
        return cls(str(row["hypothesis_id"]), str(row["title"]), str(row["role"]),
                   str(row["backend"]), str(row["instructions"]),
                   bool(row.get("read_only", True)),
                   tuple(str(x) for x in row.get("depends_on", [])),
                   bool(row.get("requires_repository", True)))


def validate_portfolio(tasks: list[AgentTask], maximum_branch_fraction: float = .5) -> None:
    """Prevent a coordinator from spending the whole round on one local idea."""
    if not tasks:
        raise ValueError("coordinator returned no tasks")
    if not (0 < maximum_branch_fraction <= 1):
        raise ValueError("maximum branch fraction must lie in (0, 1]")
    counts: dict[str, int] = {}
    for task in tasks:
        if task.backend == "zai" and task.requires_repository:
            raise ValueError("z.ai has no repository tools; assign it context-only work")
        if task.role in {"referee", "replicator"}:
            continue
        counts[task.hypothesis_id] = counts.get(task.hypothesis_id, 0) + 1
    research_count = sum(counts.values())
    if research_count >= 3 and counts and max(counts.values()) > max(
            1, math.ceil(research_count * maximum_branch_fraction)):
        raise ValueError("one hypothesis exceeds the round's portfolio allocation")
    if research_count >= 3 and len(counts) < 2:
        raise ValueError("a research round with 3+ workers needs multiple hypotheses")
    if len(tasks) >= 4 and not any(task.role in {"referee", "replicator"} for task in tasks):
        raise ValueError("a 4+ task round needs an adversarial review or replication task")


@dataclass
class ResearchStore:
    run_dir: Path
    objective: str
    state: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.run_dir = self.run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "state.json"
        if path.exists():
            self.state = json.loads(path.read_text())
            if self.state.get("objective") != self.objective:
                raise RuntimeError("research objective changed; use a new run directory")
        else:
            self.state = {"schema_version": 2, "objective": self.objective,
                          "created_at": time.time(), "round": 0,
                          "tasks": {}, "hypotheses": {}, "backend_health": {},
                          "experiments": {}, "observations": [], "events": 0}
            self.save()

        # In-place, backwards-compatible migration for early shadow runs.
        self.state.setdefault("experiments", {})
        self.state.setdefault("observations", [])
        self.state["schema_version"] = max(2, int(self.state.get("schema_version", 1)))

    def save(self) -> None:
        """Initialize/persist single-writer state. Concurrent updates use mutate()."""
        lock_path = self.run_dir / ".state.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            path = self.run_dir / "state.json"
            if path.exists():
                current = json.loads(path.read_text())
                # Merge append-only maps and monotonic counters; scalar objective
                # and creation metadata are immutable.
                current["tasks"].update(self.state.get("tasks", {}))
                current["hypotheses"].update(self.state.get("hypotheses", {}))
                current.setdefault("backend_health", {}).update(
                    self.state.get("backend_health", {}))
                current.setdefault("experiments", {}).update(
                    self.state.get("experiments", {}))
                current.setdefault("observations", [])
                known_observations = {
                    row.get("observation_id") for row in current["observations"]
                }
                current["observations"].extend(
                    row for row in self.state.get("observations", [])
                    if row.get("observation_id") not in known_observations
                )
                current["schema_version"] = max(
                    2, int(current.get("schema_version", 1)))
                current["round"] = max(int(current.get("round", 0)),
                                       int(self.state.get("round", 0)))
                current["events"] = max(int(current.get("events", 0)),
                                        int(self.state.get("events", 0)))
                self.state = current
            _atomic_json(path, self.state)

    def mutate(self, update) -> None:
        """Locked reload-modify-write; safe across parallel worker threads."""
        lock_path = self.run_dir / ".state.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            path = self.run_dir / "state.json"
            fresh = json.loads(path.read_text())
            update(fresh)
            _atomic_json(path, fresh)
            self.state = fresh

    def event(self, kind: str, **payload: Any) -> None:
        row = {"ts": round(time.time(), 3), "type": kind, **payload}
        lock_path = self.run_dir / ".events.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            with (self.run_dir / "events.jsonl").open("a") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
        self.mutate(lambda state: state.__setitem__(
            "events", int(state.get("events", 0)) + 1))

    def add_tasks(self, tasks: list[AgentTask]) -> list[AgentTask]:
        validate_portfolio(tasks)
        created = []
        for task in tasks:
            number = len(self.state["tasks"]) + 1
            task_id = f"r{self.state['round']:03d}-t{number:04d}"
            concrete = AgentTask(**{**asdict(task), "task_id": task_id})
            directory = self.run_dir / "tasks" / task_id
            directory.mkdir(parents=True, exist_ok=False)
            (directory / "brief.md").write_text(concrete.instructions + "\n")
            self.state["tasks"][task_id] = {**asdict(concrete), "status": "queued",
                "brief_sha256": _hash_text(concrete.instructions),
                "created_at": time.time()}
            hypothesis = self.state["hypotheses"].setdefault(task.hypothesis_id, {
                "title": task.title, "tasks": [], "status": "active"})
            hypothesis["tasks"].append(task_id)
            created.append(concrete)
        self.save()
        return created

    def set_status(self, task_id: str, status: str, **fields: Any) -> None:
        def update(state):
            if task_id not in state["tasks"]:
                raise KeyError(task_id)
            state["tasks"][task_id].update({"status": status, **fields})
        self.mutate(update)

    def set_backend_health(self, backend: str, status: str, reason: str = "") -> None:
        self.mutate(lambda state: state.setdefault("backend_health", {}).__setitem__(
            backend, {"status": status, "reason": reason, "ts": time.time()}))

    def prepare_handoff(self, task: AgentTask, context: str) -> dict[str, Any]:
        """Durably prepare one idempotent, AHP-inspired local worker transfer.

        This is a research-worker profile, not an AHP v1 network claim: AHP
        hands a user between complete HTTPS applications and returns a URL.
        We retain its objective/thread/transfer/resource semantics for CLIs.
        """
        directory = self.run_dir / "tasks" / task.task_id
        context_path = directory / "evidence.md"
        context_hash = _hash_text(context)
        if context_path.exists() and _hash_text(context_path.read_text()) != context_hash:
            raise RuntimeError("an idempotent handoff retry cannot change its evidence resource")
        if not context_path.exists():
            context_path.write_text(context)
        thread_id = str(uuid.uuid5(uuid.NAMESPACE_URL,
                                   f"{self.objective}\n{task.hypothesis_id}"))
        logical = {
            "profile": "evoharness-research-handoff/v1",
            "thread_id": thread_id,
            "task_id": task.task_id,
            "hypothesis_id": task.hypothesis_id,
            "objective": task.instructions,
            "role": task.role,
            "backend": task.backend,
            "read_only": task.read_only,
            "requires_repository": task.requires_repository,
            "depends_on": list(task.depends_on),
            "resources": [
                {"name": "task brief", "path": f"tasks/{task.task_id}/brief.md",
                 "sha256": _hash_text(task.instructions + "\n"), "trust": "coordinator-objective"},
                {"name": "scientific evidence", "path": f"tasks/{task.task_id}/evidence.md",
                 "sha256": context_hash, "trust": "untrusted-context"},
            ],
        }
        fingerprint = _hash_text(json.dumps(logical, sort_keys=True, separators=(",", ":")))
        proposed_key = str(uuid.uuid4())
        created = False

        def update(state):
            nonlocal created
            row = state["tasks"].get(task.task_id)
            if row is None:
                raise KeyError(task.task_id)
            previous = row.get("handoff")
            if previous:
                if previous["request_fingerprint"] != fingerprint:
                    raise RuntimeError("handoff idempotency conflict")
                return
            row["handoff"] = {**logical, "idempotency_key": proposed_key,
                              "request_fingerprint": fingerprint,
                              "accepted_at": time.time()}
            created = True

        self.mutate(update)
        envelope = self.state["tasks"][task.task_id]["handoff"]
        _atomic_json(directory / "handoff.json", envelope)
        if created:
            self.event("handoff_accepted", task_id=task.task_id,
                       thread_id=envelope["thread_id"],
                       idempotency_key=envelope["idempotency_key"],
                       request_fingerprint=fingerprint)
        return envelope

    def claim_task(self, task_id: str) -> bool:
        """Ensure two concurrent retries cannot launch the same logical worker."""
        claimed = False

        def update(state):
            nonlocal claimed
            row = state["tasks"].get(task_id)
            if row is None:
                raise KeyError(task_id)
            if row.get("status") == "queued":
                row.update({"status": "running", "started_at": time.time()})
                claimed = True

        self.mutate(update)
        return claimed

    def register_hypothesis(self, hypothesis_id: str, title: str, mechanism: str,
                            *, prior_weight: float = 1.0,
                            lineage: Iterable[str] = ()) -> None:
        """Register a mechanism-level hypothesis, not merely a candidate program."""
        if not hypothesis_id.strip() or not mechanism.strip() or prior_weight <= 0:
            raise ValueError("hypothesis id, mechanism, and positive prior are required")

        def update(state):
            row = state["hypotheses"].setdefault(hypothesis_id, {})
            row.update({"title": title, "mechanism": mechanism,
                        "prior_weight": float(prior_weight),
                        "lineage": list(lineage), "status": row.get("status", "active")})
            row.setdefault("tasks", [])
            row.setdefault("predictions", {})
            row.setdefault("posterior_weight", 0.0)
            row.setdefault("residuals", {})
            self._update_beliefs(state)

        self.mutate(update)
        self.event("hypothesis_registered", hypothesis_id=hypothesis_id)

    def refresh_beliefs(self) -> None:
        """Recompute allocator weights after migration or administrative repair."""
        self.mutate(self._update_beliefs)

    def set_hypothesis_status(self, hypothesis_id: str, status: str,
                              reason: str) -> None:
        if status not in {"active", "retired", "revised"} or not reason.strip():
            raise ValueError("hypothesis status and reason are required")
        def update(state):
            if hypothesis_id not in state["hypotheses"]:
                raise KeyError(hypothesis_id)
            state["hypotheses"][hypothesis_id].update(
                {"status": status, "status_reason": reason, "status_at": time.time()})
            self._update_beliefs(state)
        self.mutate(update)
        self.event("hypothesis_status", hypothesis_id=hypothesis_id,
                   status=status, reason=reason)

    def revise_hypothesis(self, hypothesis_id: str, mechanism: str,
                          reason: str) -> None:
        if not mechanism.strip() or not reason.strip():
            raise ValueError("revised mechanism and reason are required")
        def update(state):
            row = state["hypotheses"].get(hypothesis_id)
            if row is None:
                raise KeyError(hypothesis_id)
            row.setdefault("revisions", []).append({
                "mechanism": row.get("mechanism"), "reason": reason,
                "revised_at": time.time()})
            row.update({"mechanism": mechanism, "status": "active",
                        "status_reason": reason, "status_at": time.time()})
            self._update_beliefs(state)
        self.mutate(update)
        self.event("hypothesis_revised", hypothesis_id=hypothesis_id,
                   reason=reason)

    def register_experiment(self, experiment_id: str, condition: str, metric: str,
                            *, estimated_cost: float = 1.0, fidelity: str = "vlf",
                            artifact: str = "", strict: bool = True) -> None:
        """Register a possible intervention before observing its result."""
        if not experiment_id.strip() or not condition.strip() or not metric.strip():
            raise ValueError("experiment id, condition, and metric are required")
        if estimated_cost <= 0:
            raise ValueError("estimated experiment cost must be positive")

        created = False

        def update(state):
            nonlocal created
            experiments = state.setdefault("experiments", {})
            proposed = {
                "condition": condition, "metric": metric,
                "estimated_cost": float(estimated_cost), "fidelity": fidelity,
                "artifact": artifact, "status": "proposed", "created_at": time.time(),
            }
            if experiment_id in experiments:
                existing = experiments[experiment_id]
                stable_keys = ("condition", "metric", "estimated_cost", "fidelity", "artifact")
                if any(existing.get(key) != proposed.get(key) for key in stable_keys):
                    if strict:
                        raise ValueError(f"experiment idempotency conflict: {experiment_id}")
                    # Coordinators may refine a recurring probe between rounds.
                    # Preserve the original pre-registration, record the
                    # revision, and keep the campaign alive rather than making
                    # a recoverable planning disagreement fatal.
                    revisions = state.setdefault("experiment_revisions", [])
                    revisions.append({"experiment_id": experiment_id,
                                      "previous": dict(existing),
                                      "proposed": proposed,
                                      "recorded_at": time.time()})
                return
            experiments[experiment_id] = proposed
            created = True

        self.mutate(update)
        if created:
            self.event("experiment_registered", experiment_id=experiment_id)

    def record_prediction(self, hypothesis_id: str, experiment_id: str, mean: float,
                          *, scale: float, rationale: str = "") -> None:
        """Pre-register a quantitative prediction used for falsification and VoI."""
        if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("prediction mean must be finite and scale positive")

        def update(state):
            if hypothesis_id not in state["hypotheses"]:
                raise KeyError(hypothesis_id)
            if experiment_id not in state.setdefault("experiments", {}):
                raise KeyError(experiment_id)
            state["hypotheses"][hypothesis_id].setdefault("predictions", {})[
                experiment_id] = {"mean": float(mean), "scale": float(scale),
                                  "rationale": rationale, "created_at": time.time()}

        self.mutate(update)
        self.event("prediction_registered", hypothesis_id=hypothesis_id,
                   experiment_id=experiment_id)

    def record_observation(self, experiment_id: str, value: float, *,
                           actual_cost: float, evaluator_hash: str,
                           artifact: str, held_out: bool = True) -> str:
        """Append trusted evidence and update approximate model weights.

        The evaluator remains authoritative. This lightweight Gaussian scoring is
        a budget allocator, not a replacement for the physics or promotion gates.
        """
        if not math.isfinite(value) or actual_cost < 0:
            raise ValueError("observation value must be finite and cost nonnegative")
        if not evaluator_hash.strip() or not artifact.strip():
            raise ValueError("evaluator hash and artifact are required")
        observation_id = f"obs-{int(time.time() * 1_000_000)}-{os.getpid()}"

        def update(state):
            experiment = state.setdefault("experiments", {}).get(experiment_id)
            if experiment is None:
                raise KeyError(experiment_id)
            row = {"observation_id": observation_id, "experiment_id": experiment_id,
                   "metric": experiment["metric"], "value": float(value),
                   "actual_cost": float(actual_cost), "evaluator_hash": evaluator_hash,
                   "artifact": artifact, "held_out": bool(held_out), "ts": time.time()}
            state.setdefault("observations", []).append(row)
            experiment.update({"status": "observed", "last_observation": observation_id})
            self._update_beliefs(state)

        self.mutate(update)
        self.event("observation_recorded", observation_id=observation_id,
                   experiment_id=experiment_id)
        return observation_id

    @staticmethod
    def _update_beliefs(state: dict[str, Any]) -> None:
        """Compute transparent pseudo-posteriors from pre-registered predictions."""
        scored: list[tuple[str, float]] = []
        observations = state.get("observations", [])
        for hypothesis_id, hypothesis in state.get("hypotheses", {}).items():
            if (hypothesis.get("status", "active") != "active" or
                    not hypothesis.get("mechanism")):
                # A retired/revised-out mechanism must not retain a stale visible
                # allocation weight from the previous normalization.
                hypothesis["posterior_weight"] = 0.0
                continue
            prior = max(float(hypothesis.get("prior_weight", 1.0)), 1e-300)
            log_weight = math.log(prior)
            residuals: dict[str, dict[str, float | bool]] = {}
            matches = 0
            for observation in observations:
                prediction = hypothesis.get("predictions", {}).get(
                    observation["experiment_id"])
                if prediction is None or observation.get("metric") != state["experiments"][
                        observation["experiment_id"]]["metric"]:
                    continue
                standardized = ((float(observation["value"]) - float(prediction["mean"])) /
                                float(prediction["scale"]))
                log_weight -= .5 * standardized * standardized
                residuals[observation["observation_id"]] = {
                    "raw": float(observation["value"]) - float(prediction["mean"]),
                    "standardized": standardized,
                    "held_out": bool(observation.get("held_out", False)),
                }
                matches += 1
            hypothesis["residuals"] = residuals
            hypothesis["evidence_count"] = matches
            scored.append((hypothesis_id, log_weight))
        if not scored:
            return
        ceiling = max(value for _, value in scored)
        normalizer = sum(math.exp(value - ceiling) for _, value in scored)
        for hypothesis_id, value in scored:
            state["hypotheses"][hypothesis_id]["posterior_weight"] = (
                math.exp(value - ceiling) / normalizer)

    def select_discriminating_experiment(self) -> dict[str, Any] | None:
        """Select an unobserved intervention by weighted disagreement per cost."""
        candidates: list[dict[str, Any]] = []
        for experiment_id, experiment in self.state.get("experiments", {}).items():
            if experiment.get("status") == "observed":
                continue
            predictions = []
            for hypothesis_id, hypothesis in self.state["hypotheses"].items():
                if (hypothesis.get("status", "active") != "active" or
                        not hypothesis.get("mechanism")):
                    continue
                prediction = hypothesis.get("predictions", {}).get(experiment_id)
                if prediction is not None:
                    predictions.append((hypothesis_id, float(prediction["mean"]),
                                        float(hypothesis.get("posterior_weight", 0))))
            if len(predictions) < 2:
                continue
            total_weight = sum(weight for _, _, weight in predictions)
            if total_weight <= 0:
                weights = [1 / len(predictions)] * len(predictions)
            else:
                weights = [weight / total_weight for _, _, weight in predictions]
            center = sum(weight * prediction[1]
                         for weight, prediction in zip(weights, predictions))
            disagreement = sum(weight * (prediction[1] - center) ** 2
                               for weight, prediction in zip(weights, predictions))
            cost = float(experiment["estimated_cost"])
            candidates.append({"experiment_id": experiment_id,
                               "score": disagreement / cost,
                               "disagreement": disagreement, "estimated_cost": cost,
                               "predictions": {hid: mean for hid, mean, _ in predictions}})
        return max(candidates, key=lambda row: row["score"], default=None)

    def pool_signal(self, *, residual_threshold: float = 2.5,
                    concentration_threshold: float = .85) -> dict[str, Any]:
        """Advise expand/retain/shrink from held-out predictive evidence."""
        active = [(hypothesis_id, row) for hypothesis_id, row in
                  self.state.get("hypotheses", {}).items()
                  if row.get("status", "active") == "active" and row.get("mechanism")]
        if not active:
            return {"action": "expand", "reason": "no active mechanism hypotheses"}
        best_id, best = max(active, key=lambda item: float(
            item[1].get("posterior_weight", 0)))
        held_out = [abs(float(row["standardized"]))
                    for row in best.get("residuals", {}).values()
                    if row.get("held_out")]
        worst = max(held_out, default=None)
        weight = float(best.get("posterior_weight", 0))
        if worst is not None and worst > residual_threshold:
            return {"action": "expand", "reason": "best hypothesis fails held-out prediction",
                    "hypothesis_id": best_id, "posterior_weight": weight,
                    "worst_standardized_residual": worst}
        if held_out and weight >= concentration_threshold and worst <= residual_threshold:
            return {"action": "shrink", "reason": "posterior concentrated with acceptable holdout",
                    "hypothesis_id": best_id, "posterior_weight": weight,
                    "worst_standardized_residual": worst}
        return {"action": "retain", "reason": "evidence remains inconclusive",
                "hypothesis_id": best_id, "posterior_weight": weight,
                "worst_standardized_residual": worst}

    def record_result(self, task_id: str, text: str, *, returncode: int,
                      seconds: float, command: list[str]) -> Path:
        directory = self.run_dir / "tasks" / task_id
        result = directory / "result.md"
        canonical_text = text.rstrip() + "\n"
        canonical_hash = _hash_text(canonical_text)
        result.write_text(canonical_text)
        verified, missing = artifact_claims(canonical_text)
        task_row = self.state["tasks"][task_id]
        if not task_row.get("requires_repository", True):
            evidence_quality = ("unverified-context-claims" if missing else
                                "context-citation-report" if verified else
                                "context-only-reasoning")
        else:
            evidence_quality = ("artifact-referencing-report" if verified and not missing else
                                "unverified-artifact-claims" if missing else
                                "reasoning-only")
        self.set_status(task_id, "completed" if returncode == 0 else "failed",
                        result_sha256=canonical_hash, returncode=returncode,
                        seconds=round(seconds, 2), command=command,
                        verified_artifacts=verified, missing_artifact_claims=missing,
                        evidence_quality=evidence_quality,
                        finished_at=time.time())
        self.event("agent_result", task_id=task_id, returncode=returncode,
                   result_sha256=canonical_hash)
        handoff = self.state["tasks"][task_id].get("handoff", {})
        _atomic_json(directory / "return-handoff.json", {
            "profile": "evoharness-research-result/v1",
            "thread_id": handoff.get("thread_id"), "task_id": task_id,
            "parent_idempotency_key": handoff.get("idempotency_key"),
            "result_sha256": canonical_hash, "returncode": returncode,
            "resources": [{"name": "worker result", "path": f"tasks/{task_id}/result.md",
                           "sha256": canonical_hash,
                           "trust": "untrusted-agent-report"}],
            "completed_at": time.time(),
        })
        return result

    def evidence_digest(self, maximum_chars: int = 50_000) -> str:
        """Compact artifact index plus recent conclusions; raw files stay on disk."""
        blocks = [f"# Objective\n{self.objective}", "# Persistent research evidence"]
        hypotheses = []
        for hypothesis_id, row in sorted(self.state["hypotheses"].items()):
            if not row.get("mechanism"):
                continue
            hypotheses.append({
                "id": hypothesis_id, "title": row.get("title"),
                "mechanism": row["mechanism"],
                "status": row.get("status", "active"),
                "posterior_weight": row.get("posterior_weight"),
                "evidence_count": row.get("evidence_count", 0),
                "contradictions": sorted(
                    (value for value in row.get("residuals", {}).values()
                     if abs(float(value.get("standardized", 0))) > 2),
                    key=lambda value: abs(float(value["standardized"])), reverse=True)[:3],
            })
        if hypotheses:
            blocks.append("# Mechanism belief state\n" +
                          json.dumps(hypotheses, indent=2, sort_keys=True))
            blocks.append("# Pool control signal\n" +
                          json.dumps(self.pool_signal(), sort_keys=True))
        next_experiment = self.select_discriminating_experiment()
        if next_experiment:
            blocks.append("# Highest-value unobserved experiment\n" +
                          json.dumps(next_experiment, indent=2, sort_keys=True))
        completed = [(task_id, row) for task_id, row in self.state["tasks"].items()
                     if row.get("status") in {"completed", "failed"} and
                     not str(row.get("hypothesis_id", "")).startswith("coordinator-r")]
        for task_id, row in completed[-30:]:
            result = self.run_dir / "tasks" / task_id / "result.md"
            excerpt = result.read_text()[-2500:] if result.exists() else "missing result"
            quality = row.get("evidence_quality", "legacy-unclassified")
            warning = ("\nWARNING: these claimed artifact paths do not exist: " +
                       json.dumps(row.get("missing_artifact_claims"))
                       if row.get("missing_artifact_claims") else "")
            trust = ("\nUNTRUSTED AGENT REPORT: treat every conclusion as a hypothesis until a "
                     "host observation with evaluator hash confirms it.")
            blocks.append(f"## {task_id}: {row['title']} [{row['status']}; {quality}]"
                          f"{warning}{trust}\n{excerpt}")
        blocks.append("# Artifact rule\nTreat claims without a path/hash/evaluator record as hypotheses, not facts.")
        return "\n\n".join(blocks)[-maximum_chars:]


@dataclass(frozen=True)
class AgentLimits:
    timeout_seconds: int = 3600
    claude_max_usd: float = 5.0
    model: str | None = None
    zai_max_usd: float = 2.0
    zai_model: str = "glm-5.2"
    claude_coordinator_model: str | None = None
    claude_worker_model: str | None = None
    total_budget_usd: float | None = None
    deadline_epoch: float | None = None


class BudgetLedger:
    """Crash-safe conservative reservation ledger for a campaign."""
    def __init__(self, path: Path, cap: float | None) -> None:
        self.path = path
        self.cap = cap
        self.lock = path.with_suffix(path.suffix + ".lock")
        if not path.exists():
            self._write({"cap_usd": cap, "reserved_usd": 0.0, "reservations": {}})

    def _write(self, data: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, self.path)

    def reserve(self, task_id: str, amount: float) -> bool:
        if self.cap is None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock.open("a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text()) if self.path.exists() else {
                "cap_usd": self.cap, "reserved_usd": 0.0, "reservations": {}}
            reservations = data.setdefault("reservations", {})
            if task_id in reservations:
                return True
            reserved = float(data.get("reserved_usd", 0.0))
            if reserved + amount > float(data.get("cap_usd", self.cap)) + 1e-9:
                return False
            reservations[task_id] = amount
            data["reserved_usd"] = reserved + amount
            self._write(data)
            return True


class CliAgentRunner:
    def __init__(self, repo: Path, store: ResearchStore,
                 limits: AgentLimits = AgentLimits()) -> None:
        self.repo = repo.resolve(); self.store = store; self.limits = limits
        self.budget = BudgetLedger(store.run_dir / "budget.json", limits.total_budget_usd)

    @staticmethod
    def available() -> set[str]:
        available = {name for name in {"claude", "codex"} if shutil.which(name)}
        if os.environ.get("ZAI_API_KEY"):
            available.add("zai")
        else:
            env = REPO_ROOT / ".env"
            if env.exists() and any(line.strip().startswith("ZAI_API_KEY=")
                                    for line in env.read_text().splitlines()):
                available.add("zai")
        return available

    def usable(self) -> set[str]:
        disabled = {backend for backend, row in
                    self.store.state.get("backend_health", {}).items()
                    if row.get("status") == "unavailable"}
        return self.available() - disabled

    def _workdir(self, task: AgentTask) -> Path:
        directory = self.store.run_dir / "tasks" / task.task_id
        if task.read_only:
            return self.repo
        worktree = directory / "worktree"
        if not worktree.exists():
            subprocess.run(["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
                           cwd=self.repo, check=True, capture_output=True, text=True)
        return worktree

    def command(self, task: AgentTask, workdir: Path, output: Path) -> list[str]:
        if task.backend == "claude":
            # Implementation agents may edit their detached worktree but cannot
            # invoke the host physics CLI or mutate campaign artifacts directly.
            tools = "Read,Glob,Grep" if task.read_only else "Read,Glob,Grep,Edit,Write"
            command = ["claude", "-p", "--output-format", "text",
                       "--permission-mode", "dontAsk", "--allowedTools", tools,
                       "--max-budget-usd", str(self.limits.claude_max_usd)]
            selected_model = self.limits.model
            if task.hypothesis_id.startswith("coordinator-"):
                selected_model = self.limits.claude_coordinator_model or selected_model
            elif self.limits.claude_worker_model:
                selected_model = self.limits.claude_worker_model
            if selected_model:
                command += ["--model", selected_model]
            return command
        if task.backend == "zai":
            return ["zai-api", self.limits.zai_model]
        command = ["codex", "exec", "-C", str(workdir), "-s",
                   "read-only" if task.read_only else "workspace-write",
                   "-c", 'approval_policy="never"',
                   "--output-last-message", str(output), "-"]
        if self.limits.model:
            command[2:2] = ["-m", self.limits.model]
        return command

    def run(self, task: AgentTask, context: str) -> Path:
        if task.backend not in self.available():
            raise RuntimeError(f"agent CLI not installed: {task.backend}")
        if self.limits.deadline_epoch is not None and time.time() >= self.limits.deadline_epoch:
            directory = self.store.run_dir / "tasks" / task.task_id
            directory.mkdir(parents=True, exist_ok=True)
            return self.store.record_result(task.task_id, "Campaign walltime deadline reached.",
                returncode=124, seconds=0.0, command=[])
        charge = (self.limits.claude_max_usd if task.backend == "claude" else
                  self.limits.zai_max_usd if task.backend == "zai" else 0.0)
        if charge and not self.budget.reserve(task.task_id, charge):
            directory = self.store.run_dir / "tasks" / task.task_id
            directory.mkdir(parents=True, exist_ok=True)
            return self.store.record_result(task.task_id, "Campaign budget cap reached.",
                returncode=125, seconds=0.0, command=[])
        workdir = self._workdir(task)
        directory = self.store.run_dir / "tasks" / task.task_id
        output = directory / "last-message.txt"
        handoff = self.store.prepare_handoff(task, context)
        prompt = ("You are one worker in a persistent scientific research portfolio.\n"
                  "Work only on the assigned hypothesis. Inspect the repository and evidence.\n"
                  "Report falsifiable claims, exact artifact paths, negative results, and next tests.\n"
                  "Do not call an inherited-boundary result independent. Do not claim a score without "
                  "a fresh evaluator artifact.\n\nASSIGNMENT:\n" + task.instructions +
                  "\n\nSHARED EVIDENCE:\n" + context)
        if task.backend == "zai":
            prompt = ("IMPORTANT: You have no filesystem or shell access. Use only the supplied "
                      "evidence. Never claim that you inspected, counted, or found repository files.\n\n" +
                      prompt)
        command = self.command(task, workdir, output)
        if not self.store.claim_task(task.task_id):
            existing = directory / "result.md"
            if existing.exists():
                return existing
            raise RuntimeError(f"handoff already claimed: {task.task_id}")
        self.store.set_status(task.task_id, "running", workdir=str(workdir),
                              thread_id=handoff["thread_id"],
                              idempotency_key=handoff["idempotency_key"])
        started = time.monotonic()
        if task.backend == "zai":
            from evoharness.engine.ledger import BudgetGuard, Ledger
            from evoharness.engine.llm import LLM
            ledger = Ledger(directory / "zai")
            guard = BudgetGuard(self.limits.zai_max_usd, 2,
                                self.limits.timeout_seconds)
            try:
                llm = LLM(ledger, guard)
                temperature = .2 if task.hypothesis_id.startswith("coordinator-") else .7
                text = llm.chat(
                    self.limits.zai_model,
                    [{"role": "user", "content": prompt}], temperature,
                    role=f"research_{task.role}", max_tokens=None)
                if not text.strip():
                    text = llm.chat(
                        self.limits.zai_model,
                        [{"role": "user", "content": prompt +
                          "\n\nYour previous attempt exhausted its completion budget without a "
                          "final answer. Return the complete requested final artifact now."}], .0,
                        role=f"research_{task.role}_empty_retry",
                        max_tokens=None)
                if not text.strip():
                    raise RuntimeError("z.ai returned empty content after one bounded retry")
                return self.store.record_result(task.task_id, text, returncode=0,
                    seconds=time.monotonic() - started,
                    command=["zai-api", self.limits.zai_model])
            except Exception as exc:
                return self.store.record_result(task.task_id, f"z.ai failed: {exc}",
                    returncode=2, seconds=time.monotonic() - started,
                    command=["zai-api", self.limits.zai_model])
        try:
            proc = subprocess.run(command, cwd=workdir, input=prompt,
                                  capture_output=True, text=True,
                                  timeout=self.limits.timeout_seconds)
            text = output.read_text() if output.exists() else proc.stdout
            if proc.stderr:
                (directory / "stderr.log").write_text(proc.stderr)
                if "Failed RTM_NEWADDR" in proc.stderr or "bwrap: loopback" in proc.stderr:
                    self.store.set_backend_health(
                        task.backend, "unavailable",
                        "nested sandbox cannot create its loopback network namespace")
                elif ("Failed to authenticate" in proc.stderr or
                      "OAuth session expired" in proc.stderr):
                    self.store.set_backend_health(
                        task.backend, "unavailable",
                        "provider authentication expired and requires user login")
            return self.store.record_result(task.task_id, text, returncode=proc.returncode,
                seconds=time.monotonic() - started, command=command)
        except subprocess.TimeoutExpired as exc:
            text = f"Agent timed out after {self.limits.timeout_seconds}s.\n{exc.stdout or ''}"
            return self.store.record_result(task.task_id, text, returncode=124,
                seconds=time.monotonic() - started, command=command)

    def run_parallel(self, tasks: Iterable[AgentTask], context: str,
                     workers: int) -> list[Path]:
        # Round-robin providers so a coordinator's task ordering cannot fill
        # every slot with one expensive backend while a required cheap backend
        # waits. Role order within each backend remains stable.
        grouped: dict[str, list[AgentTask]] = {}
        for task in tasks:
            grouped.setdefault(task.backend, []).append(task)
        tasks = []
        while any(grouped.values()):
            for backend in grouped:
                if grouped[backend]:
                    tasks.append(grouped[backend].pop(0))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self.run, task, context) for task in tasks]
            return [future.result() for future in futures]


def coordinator_prompt(objective: str, evidence: str, backends: Iterable[str],
                       task_count: int, allow_worktrees: bool = False,
                       required_backends: Iterable[str] = (),
                       backend_minimums: Mapping[str, int] | None = None) -> str:
    required = sorted(set(required_backends))
    minimums = dict(backend_minimums or {})
    for backend in required:
        minimums[backend] = max(1, minimums.get(backend, 0))
    return f"""You are the senior research coordinator, not a parameter tuner.
Objective: {objective}

Choose {task_count} bounded research tasks for independent workers using available backends
{sorted(backends)}. This round MUST satisfy these minimum WORKER counts by backend: {minimums}.
You may ask them to read broadly, derive a new construction, write numerical
experiments, analyze failures, or attack another worker's premise. Preserve a portfolio: for 3+
tasks use at least two genuinely different hypotheses. Do not allocate a majority to polishing the
current champion unless material independent evidence already justifies it. Every task needs a
falsifiable deliverable and must state its expected outcome before evaluation. Prefer a proposed
experiment where live mechanisms predict different outcomes. Obey the pool control signal: an
"expand" signal requires at least one genuinely new mechanism; a "shrink" signal permits pruning
near-duplicates but never deletes the immutable evidence. The host—not workers—controls official
physics evaluation and promotion. Posterior-like weights and information-per-cost scores allocate
budget only; they never override feasibility, high-fidelity, provenance, or official gates.

Pre-register mechanism predictions before the workers see any result. Each proposed experiment must
have predictions from at least two hypotheses in the same metric; `scale` is the expected one-sigma
prediction error. IDs must be stable slugs and new experiment IDs must be unique across rounds.

Return only JSON: {{"rationale":"...",
"hypotheses":[{{"id":"slug","title":"...","mechanism":"causal account, not a candidate",
"prior_weight":1.0,"lineage":[]}}],
"experiments":[{{"id":"probe-slug","condition":"host-controlled test definition",
"metric":"metric-name","estimated_cost":1.0,"fidelity":"vlf|lf|official"}}],
"predictions":[{{"hypothesis_id":"slug","experiment_id":"probe-slug","mean":0.0,
"scale":1.0,"rationale":"why this outcome follows"}}],
"tasks":[{{"hypothesis_id":"slug","title":"...",
"role":"explorer|experimentalist|analyst|referee|replicator","backend":"claude|codex|zai",
"instructions":"...","read_only":true|false,"depends_on":[],
"requires_repository":true|false}}]}}

z.ai is a context-only model call with no filesystem or shell. Set `requires_repository:false` and
use it only for breadth/synthesis from the supplied evidence. Claude and Codex can inspect the repo.

The persistent evidence is supplied as a separate content-addressed handoff resource and in the
SHARED EVIDENCE section of your prompt. Do not ask for it again.

Execution mode: {"implementation workers may edit isolated worktrees" if allow_worktrees else "SHADOW READ-ONLY: workers may inspect and reason but must not be assigned to write files, run physics, or browse unavailable external literature"}.
"""


def synthesis_prompt(objective: str, evidence: str) -> str:
    return f"""You are the senior scientific adjudicator taking the second turn after a
heterogeneous worker portfolio. Objective: {objective}

Cross-review the worker reports against one another and against their cited repository artifacts.
Your own synthesis task directory is currently running and therefore has no result by design; do
not count it as a missing worker.
Do not average claims and do not treat agent prose as an observation. Identify: (1) conclusions
supported directly by existing artifacts, (2) hypotheses falsified or materially revised, (3)
contradictions or correlated errors, (4) the single cheapest host-controlled next test with the
highest discriminatory value per cost, and (5) whether the next round should expand, retain, or
shrink the mechanism pool. Preserve negative results and say explicitly if no candidate is ready
for implementation or physics. You may inspect files but must not write or run physics.

Return a concise Markdown decision report with exact artifact paths and a final machine-readable
JSON object on one line:
{{"pool_action":"expand|retain|shrink","next_experiment_id":"existing-id-or-null",
"implementation_ready":false,"physics_ready":false,"accepted_findings":["..."],
"rejected_findings":["..."]}}

Persistent evidence is supplied once, separately, in the SHARED EVIDENCE section. Untrusted agent
reports are clearly marked there.
"""


def parse_coordinator_plan(text: str) -> tuple[str, list[AgentTask], dict[str, list[dict[str, Any]]]]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("coordinator returned no JSON object")
    value = json.loads(text[start:end + 1])
    allowed = {"rationale", "tasks", "hypotheses", "experiments", "predictions"}
    if set(value) - allowed or not {"rationale", "tasks"} <= set(value) or not isinstance(
            value["tasks"], list):
        raise ValueError("invalid coordinator response schema")
    tasks = [AgentTask.parse(row) for row in value["tasks"]]
    validate_portfolio(tasks)
    design = {key: value.get(key, [])
              for key in ("hypotheses", "experiments", "predictions")}
    if any(not isinstance(rows, list) for rows in design.values()):
        raise ValueError("coordinator research design fields must be lists")

    hypothesis_ids = set()
    for row in design["hypotheses"]:
        expected = {"id", "title", "mechanism", "prior_weight", "lineage"}
        if set(row) != expected or str(row["id"]) in hypothesis_ids:
            raise ValueError("invalid or duplicate hypothesis design")
        if float(row["prior_weight"]) <= 0 or not str(row["mechanism"]).strip():
            raise ValueError("hypothesis design needs mechanism and positive prior")
        hypothesis_ids.add(str(row["id"]))
    experiment_ids = set()
    for row in design["experiments"]:
        expected = {"id", "condition", "metric", "estimated_cost", "fidelity"}
        if set(row) != expected or str(row["id"]) in experiment_ids:
            raise ValueError("invalid or duplicate experiment design")
        if float(row["estimated_cost"]) <= 0:
            raise ValueError("experiment design cost must be positive")
        experiment_ids.add(str(row["id"]))
    predictions_per_experiment: dict[str, set[str]] = {}
    for row in design["predictions"]:
        expected = {"hypothesis_id", "experiment_id", "mean", "scale", "rationale"}
        if set(row) != expected or float(row["scale"]) <= 0:
            raise ValueError("invalid prediction design")
        hypothesis_id, experiment_id = str(row["hypothesis_id"]), str(row["experiment_id"])
        if hypothesis_id not in hypothesis_ids or experiment_id not in experiment_ids:
            raise ValueError("prediction references an unknown design id")
        predictions_per_experiment.setdefault(experiment_id, set()).add(hypothesis_id)
    if design["experiments"] and any(
            len(predictions_per_experiment.get(experiment_id, set())) < 2
            for experiment_id in experiment_ids):
        raise ValueError("each experiment needs predictions from two hypotheses")
    return str(value["rationale"]), tasks, design


def parse_coordinator_tasks(text: str) -> tuple[str, list[AgentTask]]:
    """Compatibility view for callers that only need worker evoharness.tasks."""
    rationale, tasks, _ = parse_coordinator_plan(text)
    return rationale, tasks
