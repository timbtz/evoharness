"""Compact, feedback-first evidence for structural-discovery rounds.

The former campaign prepended the 26k-character plan and then truncated the
prompt at 12k characters.  Physics outcomes therefore never reached the
writer.  This module keeps the measured evidence small, structured, and first.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any, Iterable


EVIDENCE_LIMIT = 12_000


def failure_class(row: dict[str, Any]) -> str:
    if row.get("status") == "evaluated":
        metrics = row.get("metrics") or row.get("vlf_metrics") or {}
        feasibility = float(metrics.get("feasibility", math.inf))
        if feasibility <= .01:
            return "feasible"
        if feasibility <= .1:
            return "repairable"
        active = str(metrics.get("active_violation") or "unknown")
        return f"catastrophic:{active}"
    error = str(row.get("error") or " ".join(row.get("errors") or [])).lower()
    if ("geometry" in error or "self-intersection" in error or "aspect" in error or
            "minimum of b is at the boundary" in error or
            "spectrally condensed" in error):
        return "geometry-rejected"
    if "not converged" in error or "fsqr=" in error:
        return "non-converged"
    if "timeout" in error or "timed out" in error:
        return "timeout"
    if row.get("status") == "rejected":
        return "proposal-rejected"
    return "evaluation-failed"


def _round_line(row: dict[str, Any]) -> str:
    operator = row.get("operator") or {}
    family = operator.get("family", "unknown")
    version = operator.get("version", "unknown")
    params = json.dumps(operator.get("parameters") or {}, sort_keys=True,
                        separators=(",", ":"))
    klass = failure_class(row)
    if row.get("status") != "evaluated":
        detail = str(row.get("error") or " ".join(row.get("errors") or []))[:180]
        return f"- r{row.get('round', '?')} {family}/{version} {params} -> {klass}: {detail}"
    metrics = row.get("metrics") or row.get("vlf_metrics") or {}
    return (f"- r{row.get('round', '?')} {family}/{version} {params} -> {klass}; "
            f"L={metrics.get('objective_L')} feasibility={metrics.get('feasibility')} "
            f"honest={metrics.get('honest_score')} active={metrics.get('active_violation')}")


def family_statistics(rounds: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rounds:
        family = (row.get("operator") or {}).get("family")
        if family and family != "none":
            grouped[str(family)].append(row)
    out: dict[str, dict[str, Any]] = {}
    for family, rows in sorted(grouped.items()):
        evaluated = [r for r in rows if r.get("status") == "evaluated"]
        metrics = [(r.get("metrics") or r.get("vlf_metrics") or {}) for r in evaluated]
        out[family] = {
            "attempts": len(rows),
            "evaluated": len(evaluated),
            "failure_classes": dict(sorted(Counter(failure_class(r) for r in rows).items())),
            "best_L": max((float(m["objective_L"]) for m in metrics
                           if m.get("objective_L") is not None), default=None),
            "best_feasibility": min((float(m["feasibility"]) for m in metrics
                                     if m.get("feasibility") is not None), default=None),
        }
    return out


def feedback_evidence(rounds: list[dict[str, Any]], *, latest: int = 24,
                      extra: str = "") -> str:
    """Return bounded evidence with measurements before any background prose."""
    stats = family_statistics(rounds)
    sections = [
        "## Host-measured independent-arm evidence (highest priority)",
        "Every row below is a fresh bank-disabled physics result. Do not repeat a failed "
        "parameter region. If a family repeatedly fails for the same reason, revise its "
        "physical mechanism rather than making a cosmetic parameter change.",
        json.dumps(stats, sort_keys=True, separators=(",", ":")),
        "## Most recent measured rounds",
        *[_round_line(r) for r in rounds[-latest:]],
    ]
    if not rounds:
        sections.append("No independent physics rounds have completed yet; maximize structural breadth.")
    sections.extend([
        "## Fixed decision contract",
        "Target official score 0.70. Construction operators need a plausible >=5% L mechanism. "
        "A candidate is promotable if feasibility <=0.1; L>=12.5 is usable only when its VLF "
        "screen resolution matches official spectral resolution. Gradients, public "
        "boundaries, incumbent fallback, and inherited coefficients are forbidden. VLF is a "
        "screen; LF and official verification remain mandatory.",
    ])
    if extra:
        sections.extend(["## Additional scoped evidence", extra])
    return "\n".join(sections)[:EVIDENCE_LIMIT]


def evidence_window(evidence: str, limit: int = EVIDENCE_LIMIT) -> str:
    """Fail closed: callers must put measurements first; preserve that prefix."""
    return evidence[:limit]


def choose_family(rounds: list[dict[str, Any]], families: Iterable[str],
                  retire_after: int = 6) -> str:
    """Breadth first, then favor the best non-retired measured family.

    A family is scoped-retired after enough evaluated catastrophic samples with
    no repairable point. At least one family always remains selectable.
    """
    ordered = sorted(set(families))
    if not ordered:
        raise ValueError("at least one family is required")
    stats = family_statistics(rounds)
    attempts = {f: int(stats.get(f, {}).get("attempts", 0)) for f in ordered}
    evaluated = {f: int(stats.get(f, {}).get("evaluated", 0)) for f in ordered}
    minimum = min(evaluated.values())
    under_sampled = [f for f in ordered if evaluated[f] == minimum]
    if minimum < 2:
        return min(under_sampled, key=lambda f: (attempts[f], f))
    live = []
    for family in ordered:
        row = stats.get(family, {})
        failures = row.get("failure_classes", {})
        useful = failures.get("feasible", 0) + failures.get("repairable", 0)
        if evaluated[family] < retire_after or useful:
            live.append(family)
    live = live or ordered
    return min(live, key=lambda f: (
        evaluated[f], attempts[f],
        float(stats.get(f, {}).get("best_feasibility") or math.inf),
        f,
    ))
