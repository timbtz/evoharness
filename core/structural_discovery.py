"""Typed records and scheduling primitives for structural basin discovery.

This module is deliberately independent of the Stellar container.  It owns the
research bookkeeping and decisions; the physics evaluator remains the sole
source of measured metrics.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np

CONSTRAINT_KEYS = ("aspect", "iota", "qi", "mirror", "elongation")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


# Transform families act on an already-converged source basin, so their
# materiality bar is measured honest-score gain, not a 5% L claim. The flat 5%
# floor is what pushed every accepted operator 10-30x past the measured qi
# frontier: a mode-continuation band at amplitude 1e-4 gains 2.2% L and +0.0136
# honest at UNCHANGED feasibility, while the 5% claim forces amplitudes that
# drive feasibility to 0.12-0.36 (runs/transform-frontier/report.json).
TRANSFORM_FAMILIES = {"truncation_reconstruction", "structural_dilation",
                      "mode_continuation"}
L_GAIN_FLOOR = {"transform": .002, "construction": .05}


def min_l_gain(family: str) -> float:
    return L_GAIN_FLOOR["transform" if family in TRANSFORM_FAMILIES
                        else "construction"]


@dataclass(frozen=True)
class OperatorSpec:
    family: str
    version: str
    parameters: dict[str, Any]
    mechanism: str
    expected_l_gain_fraction: float
    kill_criterion: str
    independent: bool = True

    def __post_init__(self) -> None:
        if not self.family or not self.version:
            raise ValueError("operator family and version are required")
        floor = min_l_gain(self.family)
        if self.expected_l_gain_fraction < floor:
            raise ValueError(f"{self.family} operators must claim a "
                             f">={floor:.1%} plausible L gain")
        if not self.mechanism.strip() or not self.kill_criterion.strip():
            raise ValueError("mechanism and kill criterion are required")

    @property
    def id(self) -> str:
        return f"{self.family}@{self.version}:{stable_hash(self.parameters)[:12]}"


@dataclass(frozen=True)
class Provenance:
    operator_id: str
    experiment_seed: int
    source_boundary_hashes: tuple[str, ...] = ()
    independent: bool = True
    public_bank_enabled: bool = False

    def __post_init__(self) -> None:
        if self.independent and (self.public_bank_enabled or self.source_boundary_hashes):
            raise ValueError("independent provenance cannot depend on boundary artifacts")


@dataclass(frozen=True)
class BehaviorDescriptor:
    nfp: int
    max_poloidal_mode: int
    max_toroidal_mode: int
    support_bin: str
    l_bin: int
    feasibility_bin: str
    active_constraint: str
    novelty_bin: str
    convergence_class: str
    provenance_class: str

    @property
    def cell(self) -> tuple[Any, ...]:
        return tuple(asdict(self).values())


@dataclass
class BasinRecord:
    job_id: str
    boundary: dict[str, Any]
    metrics: dict[str, Any]
    provenance: Provenance
    evaluator_fingerprint: str
    fidelity: str
    solves: int = 1
    descriptor: BehaviorDescriptor | None = None
    status: str = "evaluated"
    errors: list[str] = field(default_factory=list)

    @property
    def honest_score(self) -> float:
        return float(self.metrics.get("honest_score", self.metrics.get("shaped_score", -math.inf)))

    @property
    def objective_l(self) -> float:
        value = self.metrics.get("objective_L")
        return float(value) if value is not None else -math.inf

    @property
    def feasibility(self) -> float:
        return float(self.metrics.get("feasibility", math.inf))


def constraint_violations(metrics: dict[str, Any]) -> dict[str, float]:
    """Reproduce the pinned P2 evaluator's five normalized violations."""
    required = ("aspect_ratio", "edge_iota_per_nfp", "log10_qi",
                "edge_mirror_ratio", "max_elongation")
    if any(metrics.get(k) is None for k in required):
        return {}
    values = (
        (float(metrics["aspect_ratio"]) - 10.0) / 10.0,
        (0.25 - abs(float(metrics["edge_iota_per_nfp"]))) / 0.25,
        (float(metrics["log10_qi"]) + 4.0) / 4.0,
        (float(metrics["edge_mirror_ratio"]) - 0.2) / 0.2,
        (float(metrics["max_elongation"]) - 5.0) / 5.0,
    )
    return dict(zip(CONSTRAINT_KEYS, values))


def enrich_metrics(metrics: dict[str, Any], margin_target: float = .002,
                   margin_slope: float = .92) -> dict[str, Any]:
    """Canonical score decomposition used by host records and reports."""
    out = dict(metrics)
    violations = constraint_violations(out)
    if violations:
        out["violations"] = violations
        out["active_violation"] = max(violations, key=violations.get)
        computed = max(0.0, *violations.values())
        out["computed_feasibility"] = computed
    p2, feasibility = out.get("p2_score"), out.get("feasibility")
    if p2 is not None and feasibility is not None and float(p2) > 0:
        penalty = margin_slope * max(0.0, float(feasibility) - margin_target)
        out["margin_penalty"] = penalty
        out["honest_score"] = float(p2) - penalty
    out["raw_score"] = p2
    out["selection_score"] = float(out.get("honest_score",
        out.get("shaped_score", -math.inf)))
    return out


def _support(boundary: dict[str, Any], tol: float = 1e-12) -> tuple[int, int, str]:
    rc = np.asarray(boundary["r_cos"], dtype=float)
    zs = np.asarray(boundary["z_sin"], dtype=float)
    if rc.shape != zs.shape or rc.ndim != 2 or rc.shape[1] % 2 != 1:
        raise ValueError("invalid Fourier coefficient matrices")
    active = (np.abs(rc) > tol) | (np.abs(zs) > tol)
    rows, cols = np.nonzero(active)
    if not len(rows):
        return 0, 0, "empty"
    ntor = (rc.shape[1] - 1) // 2
    max_m = int(rows.max())
    max_n = int(np.abs(cols - ntor).max())
    density = float(active.sum() / active.size)
    return max_m, max_n, "sparse" if density < 0.25 else "dense"


def describe(record: BasinRecord) -> BehaviorDescriptor:
    record.metrics = enrich_metrics(record.metrics)
    b, m = record.boundary, record.metrics
    max_m, max_n, support_bin = _support(b)
    feasibility = record.feasibility
    if feasibility <= 0.002:
        fbin = "margin"
    elif feasibility <= 0.01:
        fbin = "official-feasible"
    elif feasibility <= 0.1:
        fbin = "repairable"
    else:
        fbin = "catastrophic"
    distance = m.get("bank_dist")
    cosine = m.get("bank_cos")
    if distance is None:
        novelty = "no-same-nfp-bank"
    elif distance >= 0.003 and (cosine is None or cosine < 0.9999):
        novelty = "robust"
    elif distance >= 0.001:
        novelty = "guard-only"
    else:
        novelty = "public-near"
    violations = m.get("violations") or {}
    active = str(m.get("active_violation") or
                 (max(violations, key=violations.get) if violations else "unknown"))
    desc = BehaviorDescriptor(
        nfp=int(b["n_field_periods"]), max_poloidal_mode=max_m,
        max_toroidal_mode=max_n, support_bin=support_bin,
        l_bin=int(math.floor(max(record.objective_l, 0.0) * 2)),
        feasibility_bin=fbin, active_constraint=active,
        novelty_bin=novelty,
        convergence_class="converged" if not record.errors else "failed",
        provenance_class="independent" if record.provenance.independent else "inherited")
    record.descriptor = desc
    return desc


class BehaviorArchive:
    """One live champion per physical behavior cell; history is never erased."""

    def __init__(self) -> None:
        self.champions: dict[tuple[Any, ...], BasinRecord] = {}
        self.history: list[BasinRecord] = []

    def add(self, record: BasinRecord) -> bool:
        descriptor = record.descriptor or describe(record)
        self.history.append(record)
        old = self.champions.get(descriptor.cell)
        if old is None or rank_key(record) > rank_key(old):
            self.champions[descriptor.cell] = record
            return True
        return False


def rank_key(record: BasinRecord) -> tuple[float, ...]:
    """Feasibility-aware vector ranking, never a hidden raw-score scalar."""
    feasible = record.feasibility <= 0.01
    robust = record.descriptor is not None and record.descriptor.novelty_bin in {
        "robust", "no-same-nfp-bank"}
    return (float(feasible), record.honest_score, record.objective_l,
            float(robust), -record.feasibility, -float(record.solves))


def successive_halving(records: Iterable[BasinRecord], keep: int) -> list[BasinRecord]:
    """Retain diverse cell leaders, then take the strongest evidence vector."""
    if keep < 1:
        raise ValueError("keep must be positive")
    archive = BehaviorArchive()
    for record in records:
        if record.status == "evaluated" and not record.errors:
            archive.add(record)
    return sorted(archive.champions.values(), key=rank_key, reverse=True)[:keep]


def census_promotable(record: BasinRecord) -> bool:
    """Cheap Gate 1: repairable, or exceptional L that warrants repair study."""
    return record.feasibility <= 0.1 or record.objective_l >= 12.5


def material_gain(current: float, baseline: float, threshold: float = 0.01) -> bool:
    return math.isfinite(current) and math.isfinite(baseline) and current - baseline >= threshold


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    next_fidelity: str | None
    reasons: tuple[str, ...]


def promotion_decision(record: BasinRecord, stage: str,
                       lf_rank_reversed: bool = False) -> PromotionDecision:
    """Explicit multi-fidelity gates; private/official can never be skipped."""
    if record.errors or record.status != "evaluated":
        return PromotionDecision(False, None, ("evaluation failed",))
    if not census_promotable(record):
        return PromotionDecision(False, None, ("no repairable feasibility or exceptional L",))
    if stage == "vlf":
        return PromotionDecision(True, "low_fidelity", ("diverse VLF leader passed Gate 1",))
    if stage == "low_fidelity":
        if lf_rank_reversed:
            return PromotionDecision(False, None, ("LF reversed the VLF ranking",))
        projected = record.honest_score
        if projected < .66 and record.objective_l < 13.2:
            return PromotionDecision(False, None, ("credible ceiling below 0.66",))
        return PromotionDecision(True, "official", ("LF candidate entered target corridor",))
    if stage == "official":
        return PromotionDecision(False, None, ("terminal fidelity",))
    raise ValueError(f"unknown fidelity stage: {stage}")


@dataclass(frozen=True)
class SurrogatePrediction:
    """Uncertainty-aware prediction used only for cheap scheduling, never ranking alone."""
    objective_l: float
    feasibility: float
    uncertainty: float
    neighbors: int


class ConstructionSurrogate:
    """Deterministic local surrogate over typed construction parameters.

    This is deliberately conservative: it proposes screens and estimates a ceiling,
    but every selected point still requires a fresh evaluator run. Categorical
    NFP/family changes are separated by the distance metric instead of interpolated.
    """
    def __init__(self, keys: Iterable[str], categorical: Iterable[str] = ("n_field_periods",)):
        self.keys = tuple(keys)
        self.categorical = frozenset(categorical)
        if not self.keys:
            raise ValueError("surrogate requires at least one parameter")
        self._rows: list[tuple[dict[str, float], float, float]] = []

    def _distance(self, a: dict[str, Any], b: dict[str, Any]) -> float:
        total = 0.0
        for key in self.keys:
            if key not in a or key not in b:
                return float("inf")
            if key in self.categorical:
                total += 0.0 if a[key] == b[key] else 4.0
            else:
                # Unit-scale each coordinate locally; clipping prevents one
                # malformed proposal from dominating acquisition.
                try:
                    delta = float(a[key]) - float(b[key])
                except (TypeError, ValueError):
                    return float("inf")
                total += min(abs(delta), 10.0) ** 2
        return math.sqrt(total)

    def add(self, parameters: dict[str, Any], objective_l: float,
            feasibility: float) -> None:
        if not math.isfinite(float(objective_l)) or not math.isfinite(float(feasibility)):
            return
        if any(k not in parameters for k in self.keys):
            raise ValueError("observation missing surrogate parameter")
        self._rows.append((dict(parameters), float(objective_l), float(feasibility)))

    @property
    def observations(self) -> int:
        return len(self._rows)

    def predict(self, parameters: dict[str, Any], k: int = 5) -> SurrogatePrediction:
        if k < 1:
            raise ValueError("k must be positive")
        distances = sorted((self._distance(parameters, row[0]), row[1], row[2])
                           for row in self._rows)
        finite = [x for x in distances if math.isfinite(x[0])]
        if not finite:
            return SurrogatePrediction(float("nan"), float("inf"), float("inf"), 0)
        chosen = finite[:k]
        weights = [1.0 / max(d, 1e-6) for d, _, _ in chosen]
        norm = sum(weights)
        lhat = sum(w * l for w, (_, l, _) in zip(weights, chosen)) / norm
        fhat = sum(w * f for w, (_, _, f) in zip(weights, chosen)) / norm
        spread = math.sqrt(sum(w * (l - lhat) ** 2 for w, (_, l, _) in zip(weights, chosen)) / norm)
        # Distance is an epistemic uncertainty term; no extrapolation is silently
        # treated as a confident prediction.
        uncertainty = spread + min(chosen[-1][0], 10.0) * 0.1
        return SurrogatePrediction(lhat, fhat, uncertainty, len(chosen))

    def acquisition(self, parameters: dict[str, Any], baseline_l: float,
                    min_gain: float = 0.01) -> tuple[bool, SurrogatePrediction]:
        pred = self.predict(parameters)
        if pred.neighbors == 0:
            return True, pred
        # Explore uncertain regions, or exploit only when predicted gain is material.
        return (pred.uncertainty >= 0.05 or
                pred.objective_l - baseline_l >= min_gain), pred


@dataclass(frozen=True)
class FiniteDifferencePolicy:
    """Cost guard for late-stage gradients after a basin has passed promotion."""
    max_coordinates: int = 12
    minimum_projected_l: float = 13.2
    minimum_honest_score: float = 0.66

    def eligible(self, record: BasinRecord) -> bool:
        return (not record.errors and record.status == "evaluated" and
                (record.objective_l >= self.minimum_projected_l or
                 record.honest_score >= self.minimum_honest_score))

    def coordinates(self, total_coordinates: int) -> int:
        if total_coordinates < 1:
            raise ValueError("total_coordinates must be positive")
        return min(total_coordinates, self.max_coordinates)
