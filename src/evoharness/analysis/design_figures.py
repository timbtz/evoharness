#!/usr/bin/env python3
"""Render the research-design and evaluation-evidence figures.

The figures are intentionally derived from committed configuration and evidence
tables.  They describe experiment structure and observed outcomes; they do not
turn descriptive comparisons into causal claims.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from evoharness.engine.config import SWITCHES
from evoharness.paths import REPO_ROOT


ROOT = REPO_ROOT
OUT = ROOT / "reports" / "figures"
DATA = ROOT / "reports" / "data"

INK = "#15243b"
MUTED = "#5c6b7f"
FAINT = "#d9e1eb"
PAPER = "#ffffff"
BLUE = "#3164c6"
BLUE_LIGHT = "#edf3ff"
TEAL = "#087e81"
TEAL_LIGHT = "#eaf7f5"
PLUM = "#7952a5"
PLUM_LIGHT = "#f5effa"
AMBER = "#af6516"
AMBER_LIGHT = "#fff5e6"
RED = "#b34a4a"


def _setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": FAINT,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "path",
        "svg.hashsalt": "evoharness",
        "pdf.fonttype": 3,
    })


def _load() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = json.loads((ROOT / "configs" / "stellar_p2.json").read_text())
    stellar = json.loads((DATA / "stellar-evidence.json").read_text())
    benchmark = json.loads((DATA / "benchmark-evidence.json").read_text())
    return config, stellar, benchmark


def _save(fig: plt.Figure, stem: str) -> None:
    creator = "EvoHarness / src/evoharness/analysis/design_figures.py"
    for suffix in ("svg", "pdf", "png"):
        kwargs: dict[str, Any] = {
            "dpi": 180,
            "bbox_inches": "tight",
            "facecolor": PAPER,
        }
        if suffix == "svg":
            kwargs["metadata"] = {"Creator": creator, "Date": None}
        elif suffix == "pdf":
            kwargs["metadata"] = {
                "Creator": creator, "CreationDate": None, "ModDate": None,
            }
        fig.savefig(OUT / f"{stem}.{suffix}", **kwargs)
    svg = OUT / f"{stem}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def _card(ax, xy: tuple[float, float], width: float, height: float,
          *, edge: str, face: str = PAPER, radius: float = 0.08,
          linewidth: float = 1.2, linestyle: str = "solid", zorder: int = 1):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=linewidth,
        linestyle=linestyle, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _pill(ax, x: float, y: float, text: str, *, selected: bool = False,
          color: str = BLUE, width: float | None = None, fontsize: float = 8.8,
          face: str | None = None) -> float:
    width = width if width is not None else max(0.68, 0.087 * len(text) + 0.27)
    fill = color if selected else (face or PAPER)
    edge = color if selected else FAINT
    _card(ax, (x, y), width, 0.34, edge=edge, face=fill, radius=0.14,
          linewidth=1.15 if selected else 0.9, zorder=3)
    ax.text(x + width / 2, y + 0.17, text, ha="center", va="center",
            fontsize=fontsize, color=PAPER if selected else INK,
            fontweight="bold" if selected else "normal", zorder=4)
    return width


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], *,
           color: str = MUTED, dashed: bool = False, width: float = 1.25,
           connection: str = "arc3") -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11, linewidth=width,
        color=color, linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=connection, shrinkA=0, shrinkB=0, zorder=2,
    ))


def _axis_label(value: str) -> str:
    return value.replace("_", " ")


def optimizer_axes(config: dict[str, Any], stellar: dict[str, Any]) -> None:
    """Five configurable switches, orthogonal controls, and two workflow lanes."""
    fig, ax = plt.subplots(figsize=(16.2, 10.2))
    ax.set(xlim=(0, 16.2), ylim=(0, 10.2))
    ax.axis("off")

    ax.text(0.18, 9.79, "The experiment has more than one control plane",
            fontsize=22, fontweight="bold", va="top")
    ax.text(0.18, 9.34,
            "Program-search choices, execution physics, and research orchestration are varied or fixed independently.",
            fontsize=11.3, color=MUTED, va="top")

    # Configurable five-axis surface.
    _card(ax, (0.18, 5.30), 9.45, 3.63, edge=BLUE, face="#fbfcff", radius=0.10)
    ax.text(0.43, 8.66, "A  Configurable optimizer design", fontsize=14.5,
            fontweight="bold", color=BLUE, va="top")
    ax.text(0.43, 8.32, "Implemented choices in engine/config.py", fontsize=9.4,
            color=MUTED, va="top")
    _pill(ax, 7.76, 8.46, "stellar_p2.json", selected=True, color=BLUE,
          width=1.58, fontsize=7.6)

    descriptions = {
        "feedback": "signal returned after evaluation",
        "gate": "candidate admission rule",
        "search": "proposal population / schedule",
        "knowledge": "curated or web context",
        "roles": "model-role allocation",
    }
    y_positions = [7.78, 7.22, 6.66, 6.10, 5.54]
    for (axis_name, options), y in zip(SWITCHES.items(), y_positions):
        ax.text(0.48, y + 0.17, axis_name.upper(), fontsize=8.7,
                fontweight="bold", color=INK, va="center")
        ax.text(1.76, y + 0.17, descriptions[axis_name], fontsize=8.1,
                color=MUTED, va="center")
        ax.plot([4.18, 9.30], [y + 0.17, y + 0.17], color=FAINT, linewidth=1,
                zorder=1)
        cursor = 4.27
        selected = config["switches"][axis_name]
        for option in options:
            label = _axis_label(option)
            width = max(0.91, 0.083 * len(label) + 0.31)
            _pill(ax, cursor, y, label, selected=option == selected,
                  color=BLUE, width=width, fontsize=8.0)
            cursor += width + 0.13
    ax.text(8.88, 8.26, "memory OFF", fontsize=7.8, color=BLUE,
            fontweight="bold", ha="center", va="center")

    # Orthogonal controls.
    _card(ax, (9.88, 5.30), 6.13, 3.63, edge=TEAL, face="#fbfefd", radius=0.10)
    ax.text(10.13, 8.66, "B  Orthogonal controls", fontsize=14.5,
            fontweight="bold", color=TEAL, va="top")
    ax.text(10.13, 8.32, "Implemented outside the five switches", fontsize=9.4,
            color=MUTED, va="top")
    controls = [
        ("EXECUTION", "task_default", "sandbox policy + resource envelope"),
        ("EVALUATOR", "multi-fidelity physics", "VLF / LF / official promotion gates"),
        ("PROVENANCE", "origin recorded", "seed lineage + artifact fingerprints"),
        ("OBJECTIVE", "quality", "score semantics remain task-specific"),
    ]
    for idx, (label, value, note) in enumerate(controls):
        y = 7.72 - idx * 0.55
        ax.text(10.15, y, label, fontsize=8.2, fontweight="bold", color=TEAL,
                va="center")
        _pill(ax, 11.34, y - 0.17, value, color=TEAL, width=1.70,
              fontsize=7.8, face=TEAL_LIGHT)
        ax.text(13.23, y, note, fontsize=7.75, color=MUTED, va="center")
    ax.text(10.15, 5.48,
            "Physics diagnostics—mirror, QI, margins, gradients—are evaluator\ninstruments, separate from the configurable feedback strategy.",
            fontsize=7.8, color=INK, va="bottom", linespacing=1.25)

    # Divider and workflow headings.
    ax.plot([0.18, 16.01], [4.99, 4.99], color=FAINT, linewidth=1)
    ax.text(0.18, 4.80, "TWO STELLARATOR WORKFLOWS", fontsize=9.0,
            fontweight="bold", color=MUTED, va="top")
    ax.text(3.52, 4.80, "Separate entry points; neither is implied by the five-axis baseline",
            fontsize=9.0, color=MUTED, va="top")

    # Latest persistent repair lane.
    _card(ax, (0.18, 2.60), 15.83, 1.84, edge=TEAL, face=TEAL_LIGHT, radius=0.09)
    ax.text(0.44, 4.18, "C  Latest persisted repair workflow", fontsize=13.2,
            fontweight="bold", color=TEAL, va="top")
    repair_nodes = [
        (0.46, 3.02, 2.33, "SCIENTIFIC SEEDS", "imported / self-generated\nprovenance tagged"),
        (3.16, 3.02, 2.08, "PERSISTENT PROPOSER", "GLM-5.2\nmeasured memory"),
        (5.61, 3.02, 2.25, "EXECUTABLE REPAIR", "candidate audit\nseed repair code"),
        (8.23, 3.02, 2.40, "VLF PHYSICS", "public bank disabled\nfeasibility first"),
        (11.00, 3.02, 2.13, "ACCEPT / PERSIST", "lower worst violation\ndurable state"),
    ]
    for x, y, width, title, body in repair_nodes:
        _card(ax, (x, y), width, 0.74, edge=TEAL, face=PAPER, radius=0.06,
              linewidth=1.0, zorder=3)
        ax.text(x + 0.13, y + 0.56, title, fontsize=7.7, color=TEAL,
                fontweight="bold", va="center", zorder=4)
        ax.text(x + 0.13, y + 0.30, body, fontsize=7.6, color=INK,
                va="center", linespacing=1.25, zorder=4)
    for left, right in zip(repair_nodes[:-1], repair_nodes[1:]):
        _arrow(ax, (left[0] + left[2] + 0.03, 3.39), (right[0] - 0.04, 3.39),
               color=TEAL)
    latest = stellar["latest_persisted_state"]["repair_loop"]
    ax.text(13.43, 3.58, "PERSISTED OUTCOME", fontsize=7.8, color=AMBER,
            fontweight="bold", va="top")
    ax.text(13.43, 3.33, f"mirror binds  {latest['best_feasibility']:.5f}",
            fontsize=9.4, color=INK, fontweight="bold", va="top")
    ax.text(13.43, 3.04, "VLF · infeasible\nnot officially verified", fontsize=7.9,
            color=MUTED, va="top", linespacing=1.3)
    _arrow(ax, (11.05, 2.97), (10.55, 2.77), color=TEAL, dashed=True,
           connection="arc3,rad=-0.35")
    ax.text(8.77, 2.75, "next measured proposal", fontsize=7.4, color=TEAL,
            va="top")

    # Independent research DAG.
    _card(ax, (0.18, 0.55), 15.83, 1.62, edge=PLUM, face=PLUM_LIGHT,
          radius=0.09, linestyle=(0, (5, 3)))
    ax.text(0.44, 1.98, "D  Independent research DAG", fontsize=13.2,
            fontweight="bold", color=PLUM, va="top")
    dag_nodes = [
        (0.46, 0.82, 2.14, "COORDINATOR", "portfolio + dependencies"),
        (3.12, 1.21, 2.15, "EXPLORER / ANALYST", "mechanisms + evidence"),
        (3.12, 0.68, 2.15, "EXPERIMENTALIST", "isolated experiment"),
        (6.08, 0.94, 2.20, "ARTIFACT CONTRACT", "files + hashes + claims"),
        (9.09, 0.94, 2.17, "REFEREE / REPLICATOR", "check or reproduce"),
        (12.08, 0.94, 2.37, "ADJUDICATION", "synthesis → next round"),
    ]
    for x, y, width, title, body in dag_nodes:
        _card(ax, (x, y), width, 0.55, edge=PLUM, face=PAPER, radius=0.055,
              linewidth=1.0, zorder=3)
        ax.text(x + 0.11, y + 0.36, title, fontsize=7.5, color=PLUM,
                fontweight="bold", va="center", zorder=4)
        ax.text(x + 0.11, y + 0.15, body, fontsize=7.45, color=INK,
                va="center", zorder=4)
    _arrow(ax, (2.64, 1.10), (3.05, 1.48), color=PLUM)
    _arrow(ax, (2.64, 1.10), (3.05, 0.96), color=PLUM)
    _arrow(ax, (5.31, 1.48), (6.03, 1.25), color=PLUM)
    _arrow(ax, (5.31, 0.96), (6.03, 1.19), color=PLUM)
    _arrow(ax, (8.32, 1.21), (9.04, 1.21), color=PLUM)
    _arrow(ax, (11.30, 1.21), (12.03, 1.21), color=PLUM)
    _arrow(ax, (14.49, 1.20), (15.43, 1.64), color=PLUM, dashed=True,
           connection="arc3,rad=-0.26")
    ax.text(14.44, 1.70, "replan", fontsize=7.5, color=PLUM)

    # Legend/footer.
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
               markeredgecolor=BLUE, markersize=8, label="selected in stellar_p2.json"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PAPER,
               markeredgecolor=MUTED, markersize=8, label="implemented alternative"),
        Line2D([0], [0], color=PLUM, linestyle=(0, (4, 3)), linewidth=1.4,
               label="separate orchestration / iteration"),
    ]
    ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.18, 0.005),
              ncol=3, frameon=False, fontsize=8.2, handlelength=2.1,
              columnspacing=2.0)
    ax.text(16.0, 0.07,
            "Architecture view · states come from committed config, code contracts, and stellar-evidence.json",
            ha="right", fontsize=7.7, color=MUTED)
    _save(fig, "optimizer-axes")


def _study(benchmark: dict[str, Any], study_id: str) -> dict[str, Any]:
    return next(row for row in benchmark["studies"] if row["id"] == study_id)


def evaluation_trajectory(stellar: dict[str, Any], benchmark: dict[str, Any]) -> None:
    """Observed progression and supported negative evidence across experiment regimes."""
    milestones = stellar["plot_series"]["database_repair_acceptance_milestones"]
    points = milestones["points"]
    rounds = np.array([row["round"] for row in points], dtype=float)
    violation = np.array([row["worst_violation"] for row in points], dtype=float)
    unaided = stellar["plot_series"]["unaided_construction_terminal_worst_violation"]
    fidelity = stellar["plot_series"]["transform_fidelity_reversal"]
    census = next(row for row in stellar["run_families"]
                  if row["family"] == "independent_structural_census")
    binpack = _study(benchmark, "initial_binpacking_5_axis_factorial")
    cvrp = _study(benchmark, "cvrp_memory_vs_score_only_seed3")
    latest = stellar["latest_persisted_state"]["repair_loop"]

    fig = plt.figure(figsize=(15.8, 9.4), facecolor=PAPER)
    gs = fig.add_gridspec(2, 3, height_ratios=(1.42, 1.0),
                          width_ratios=(1.42, 0.78, 0.92),
                          left=0.06, right=0.97, top=0.86, bottom=0.09,
                          hspace=0.47, wspace=0.34)
    repair_ax = fig.add_subplot(gs[0, :2])
    unaided_ax = fig.add_subplot(gs[0, 2])
    fidelity_ax = fig.add_subplot(gs[1, :2])
    ladder_ax = fig.add_subplot(gs[1, 2])

    fig.text(0.06, 0.955, "Evidence changes when the evaluation regime changes",
             fontsize=22, fontweight="bold", va="top")
    fig.text(0.06, 0.910,
             "Accepted VLF repair milestones show measured progress, while independent construction and fidelity transfer bound the claim.",
             fontsize=11.1, color=MUTED, va="top")

    # A — repair milestone trajectory.
    repair_ax.step(rounds, violation, where="post", color=TEAL, linewidth=2.1,
                   zorder=2)
    repair_ax.scatter(rounds, violation, s=34, facecolor=PAPER, edgecolor=TEAL,
                      linewidth=1.4, zorder=3)
    repair_ax.axhspan(0.007, 0.01, color=TEAL, alpha=0.08, zorder=0)
    repair_ax.axhline(0.01, color=TEAL, linestyle=(0, (5, 3)), linewidth=1.2)
    repair_ax.text(216, 0.0107, "feasibility threshold  0.01", color=TEAL,
                   fontsize=8.5, ha="right", va="bottom")
    repair_ax.set_yscale("log")
    repair_ax.set_xlim(0, 220)
    repair_ax.set_ylim(0.007, 0.72)
    repair_ax.set_xlabel("Repair round")
    repair_ax.set_ylabel("Accepted worst normalized violation  (log scale)")
    repair_ax.set_title("A  Database-repair acceptance frontier · VLF", loc="left",
                        fontsize=13, fontweight="bold", pad=13)
    repair_ax.grid(axis="both", color=FAINT, alpha=0.7, linewidth=0.7)
    repair_ax.set_axisbelow(True)
    repair_ax.annotate(f"round 4\n{violation[0]:.3f}", (rounds[0], violation[0]),
                       xytext=(19, 0.59), textcoords="data", fontsize=8.3,
                       color=MUTED, arrowprops={"arrowstyle": "-", "color": MUTED,
                                                  "linewidth": 0.8})
    repair_ax.annotate(f"round 93\n{violation[-2]:.5f}",
                       (rounds[-2], violation[-2]), xytext=(112, 0.071),
                       textcoords="data", fontsize=8.3, color=INK,
                       arrowprops={"arrowstyle": "-", "color": MUTED,
                                  "linewidth": 0.8})
    repair_ax.annotate(f"round 212\n{violation[-1]:.5f}",
                       (rounds[-1], violation[-1]), xytext=(174, 0.026),
                       textcoords="data", fontsize=8.3, color=INK,
                       arrowprops={"arrowstyle": "-", "color": MUTED,
                                  "linewidth": 0.8})
    repair_ax.text(0.01, -0.27,
                   "Accepted points may switch among imported or self-generated seeds; the line is a campaign frontier, not one boundary lineage.",
                   transform=repair_ax.transAxes, fontsize=8.2, color=MUTED)

    # B — independent terminal outcomes.
    vals = np.array([row["value"] for row in unaided], dtype=float)
    names = ["s42", "s7", "s11", "persistent\nconstructor"]
    ypos = np.arange(len(vals))[::-1]
    unaided_ax.hlines(ypos, 0.01, vals, color=FAINT, linewidth=2.2, zorder=1)
    unaided_ax.scatter(vals, ypos, s=56, color=AMBER, edgecolor=PAPER,
                       linewidth=0.8, zorder=3)
    unaided_ax.axvline(0.01, color=TEAL, linestyle=(0, (5, 3)), linewidth=1.2)
    for y, val in zip(ypos, vals):
        unaided_ax.text(val * 1.06, y, f"{val:.3f}", va="center", fontsize=8.6,
                        color=INK)
    unaided_ax.set_xscale("log")
    unaided_ax.set_xlim(0.008, 0.82)
    unaided_ax.set_ylim(-0.8, 3.8)
    unaided_ax.set_yticks(ypos, names)
    unaided_ax.set_xlabel("Terminal worst violation")
    unaided_ax.set_title("B  Unaided construction stayed outside the gate", loc="left",
                         fontsize=12.2, fontweight="bold", pad=13)
    unaided_ax.grid(axis="x", color=FAINT, alpha=0.7, linewidth=0.7)
    unaided_ax.set_axisbelow(True)
    unaided_ax.text(0.01, -0.27,
                    f"Independent structural census: {census['promotable']} promotable / {census['evaluated']} evaluated ({census['failed']} failed).",
                    transform=unaided_ax.transAxes, fontsize=8.2, color=MUTED)

    # C — coarse-to-official fidelity reversal.
    cases = [row["case"].replace("band_seed_", "seed ") for row in fidelity]
    x = np.arange(len(cases))
    width = 0.27
    vlf = np.array([row["vlf_delta"] for row in fidelity])
    official = np.array([row["official_delta"] for row in fidelity])
    fidelity_ax.bar(x - width / 2, vlf, width, color=TEAL, label="VLF honest Δ")
    fidelity_ax.bar(x + width / 2, official, width, color=AMBER,
                    label="official Δ")
    fidelity_ax.axhline(0, color=INK, linewidth=0.9)
    fidelity_ax.set_xticks(x, cases)
    fidelity_ax.set_ylabel("Score change from source boundary")
    fidelity_ax.set_title("C  Two high-order transforms gained at VLF and reversed officially",
                          loc="left", fontsize=12.5, fontweight="bold", pad=13)
    fidelity_ax.grid(axis="y", color=FAINT, alpha=0.7, linewidth=0.7)
    fidelity_ax.set_axisbelow(True)
    fidelity_ax.legend(frameon=False, loc="upper right", fontsize=8.5, ncol=2)
    fidelity_ax.set_ylim(-0.0036, 0.0164)
    for xi, vv, oo in zip(x, vlf, official):
        fidelity_ax.text(xi - width / 2, vv + 0.00045, f"{vv:+.4f}",
                         ha="center", va="bottom", fontsize=8.2, color=TEAL)
        fidelity_ax.text(xi + width / 2, oo - 0.00038, f"{oo:+.4f}",
                         ha="center", va="top", fontsize=8.2, color=AMBER)
    fidelity_ax.text(0.01, -0.20,
                     "Within-case fidelity audit. It supports a resolution artifact for these transforms; it does not estimate a general treatment effect.",
                     transform=fidelity_ax.transAxes, fontsize=8.2, color=MUTED)

    # D — compact evidence ladder, using exact committed evidence grades.
    ladder_ax.set(xlim=(0, 1), ylim=(0, 1))
    ladder_ax.axis("off")
    ladder_ax.set_title("D  What each study can support", loc="left",
                        fontsize=12.2, fontweight="bold", pad=13)
    ladder = [
        (0.78, BLUE, "FIVE-AXIS SCREEN", "32 configs × 3 seeds", "descriptive main effects"),
        (0.52, PLUM, "CVRP FEEDBACK PAIR", "gap 0.748 → 0.581", "paired · one seed · unequal calls"),
        (0.26, TEAL, "PERSISTENT REPAIR", f"{latest['accepted']} accepted / {latest['evaluated']} evaluated", "VLF frontier · mixed seeds"),
    ]
    for y, color, title, metric, grade in ladder:
        _card(ladder_ax, (0.03, y - 0.095), 0.94, 0.19, edge=color,
              face={BLUE: BLUE_LIGHT, PLUM: PLUM_LIGHT, TEAL: TEAL_LIGHT}[color],
              radius=0.035, linewidth=1.0)
        ladder_ax.text(0.07, y + 0.044, title, fontsize=7.7, color=color,
                       fontweight="bold", va="center")
        ladder_ax.text(0.07, y - 0.002, metric, fontsize=9.2, color=INK,
                       fontweight="bold", va="center")
        ladder_ax.text(0.07, y - 0.052, grade, fontsize=7.5, color=MUTED,
                       va="center")
    _arrow(ladder_ax, (0.50, 0.67), (0.50, 0.62), color=MUTED, width=0.8)
    _arrow(ladder_ax, (0.50, 0.41), (0.50, 0.36), color=MUTED, width=0.8)
    ladder_ax.text(0.03, 0.045,
                   f"Current conclusion: {latest['best_feasibility']:.5f} worst violation at VLF; no official feasible boundary.",
                   fontsize=8.1, color=AMBER, fontweight="bold", va="bottom",
                   wrap=True)

    fig.text(0.06, 0.008,
             "Sources: reports/data/benchmark-evidence.json and stellar-evidence.json · values are observed endpoints or accepted milestones; uncertainty labels are design-based.",
             fontsize=8.0, color=MUTED)
    _save(fig, "evaluation-trajectory")


def reward_feedback() -> None:
    """Exact Stellar P2 score, shaping, and feedback-payload contract."""
    fig, ax = plt.subplots(figsize=(16.0, 7.1))
    ax.set(xlim=(0, 16), ylim=(0, 7.1))
    ax.axis("off")

    ax.text(0.18, 6.78, "From physics measurements to selection and feedback",
            fontsize=21.5, fontweight="bold", va="top")
    ax.text(0.18, 6.36,
            "A fresh evaluator re-scores the returned boundary; the official benchmark score stays separate from search-only shaping and diagnostics.",
            fontsize=11.0, color=MUTED, va="top")

    # A — exact piecewise score contract.
    _card(ax, (0.18, 1.39), 5.14, 4.48, edge=BLUE, face="#fbfcff", radius=0.09)
    ax.text(0.43, 5.58, "A  Clean-room score contract", fontsize=14.0,
            color=BLUE, fontweight="bold", va="top")
    ax.text(0.43, 5.24, "Candidate code is absent from verification", fontsize=8.6,
            color=MUTED, va="top")

    _card(ax, (0.47, 4.48), 1.43, 0.48, edge=BLUE, face=PAPER,
          radius=0.06, linewidth=1.0)
    ax.text(1.185, 4.72, "returned boundary", ha="center", va="center",
            fontsize=7.8, fontweight="bold")
    _arrow(ax, (1.92, 4.72), (2.28, 4.72), color=BLUE)
    _card(ax, (2.31, 4.38), 2.62, 0.68, edge=TEAL, face=TEAL_LIGHT,
          radius=0.06, linewidth=1.0)
    ax.text(3.62, 4.80, "physics metrics → feasibility  f", ha="center",
            va="center", fontsize=8.5, color=TEAL, fontweight="bold")
    ax.text(3.62, 4.57, "feasible iff  f ≤ 0.01", ha="center", va="center",
            fontsize=8.3, color=INK)

    ax.text(1.76, 4.04, "OFFICIAL P2", fontsize=8.1, color=BLUE,
            fontweight="bold", ha="center")
    ax.text(4.14, 4.04, "TRAIN / VAL / PUBLIC BASE", fontsize=8.1, color=TEAL,
            fontweight="bold", ha="center")
    ax.text(0.47, 3.63, "f ≤ .01", fontsize=7.8, color=TEAL,
            fontweight="bold", va="center")
    ax.text(0.47, 2.78, "f > .01", fontsize=7.8, color=AMBER,
            fontweight="bold", va="center")

    _card(ax, (1.10, 3.27), 1.52, 0.70, edge=BLUE, face=BLUE_LIGHT,
          radius=0.05, linewidth=1.0)
    ax.text(1.86, 3.72, "benchmark score", fontsize=7.4, color=BLUE,
            fontweight="bold", ha="center")
    ax.text(1.86, 3.45, r"$S_{official}=L/20$", fontsize=10.4,
            color=INK, ha="center")
    _card(ax, (3.13, 3.27), 1.80, 0.70, edge=TEAL, face=TEAL_LIGHT,
          radius=0.05, linewidth=1.0)
    ax.text(4.03, 3.72, "base shaped score", fontsize=7.4, color=TEAL,
            fontweight="bold", ha="center")
    ax.text(4.03, 3.45, r"$S_{base}=L/20$", fontsize=11.0,
            color=INK, ha="center")

    _card(ax, (1.10, 2.42), 1.52, 0.70, edge=AMBER, face=AMBER_LIGHT,
          radius=0.05, linewidth=1.0)
    ax.text(1.86, 2.87, "benchmark score", fontsize=7.4, color=AMBER,
            fontweight="bold", ha="center")
    ax.text(1.86, 2.60, r"$S_{official}=0$", fontsize=10.7,
            color=INK, ha="center")
    _card(ax, (3.13, 2.42), 1.80, 0.70, edge=AMBER, face=AMBER_LIGHT,
          radius=0.05, linewidth=1.0)
    ax.text(4.03, 2.87, "continuous fallback", fontsize=7.4, color=AMBER,
            fontweight="bold", ha="center")
    ax.text(4.03, 2.60, r"$S_{base}=-f$", fontsize=11.0,
            color=INK, ha="center")

    ax.text(0.47, 1.92, "Fidelity", fontsize=7.7, color=MUTED,
            fontweight="bold", va="center")
    _pill(ax, 1.11, 1.75, "benchmark  official", color=BLUE, width=1.50,
          fontsize=7.4, face=BLUE_LIGHT)
    _pill(ax, 2.73, 1.75, "val  low", color=TEAL, width=0.92,
          fontsize=7.4, face=TEAL_LIGHT)
    _pill(ax, 3.77, 1.75, "train/public  VLF", color=TEAL, width=1.34,
          fontsize=7.4, face=TEAL_LIGHT)

    # B — exact optional shaping with defaults.
    _card(ax, (5.57, 1.39), 5.24, 4.48, edge=TEAL, face="#fbfefd", radius=0.09)
    ax.text(5.82, 5.58, "B  Selection pressure", fontsize=14.0,
            color=TEAL, fontweight="bold", va="top")
    ax.text(5.82, 5.24, "Applied to non-private splits after fresh evaluation", fontsize=8.6,
            color=MUTED, va="top")

    _card(ax, (5.86, 4.42), 4.66, 0.58, edge=TEAL, face=TEAL_LIGHT,
          radius=0.06, linewidth=1.0)
    ax.text(8.19, 4.71,
            r"$S_{select}=S_{base}-P_{margin}-P_{novelty}$",
            fontsize=12.4, color=INK, ha="center", va="center")

    ax.text(5.87, 4.08, "FEASIBILITY-MARGIN PENALTY", fontsize=7.7,
            color=TEAL, fontweight="bold", va="center")
    ax.text(5.87, 3.72,
            r"$P_{margin}=0.92\,\max(0, f-0.002)$",
            fontsize=11.0, color=INK, va="center")
    ax.text(5.87, 3.40, "Only when P2 > 0; reported as margin_penalty + honest_score.",
            fontsize=7.9, color=MUTED, va="center")

    ax.plot([5.87, 10.51], [3.13, 3.13], color=FAINT, linewidth=0.9)
    ax.text(5.87, 2.89, "PUBLIC-BANK NOVELTY PENALTY", fontsize=7.7,
            color=PLUM, fontweight="bold", va="center")
    ax.text(5.87, 2.53,
            r"$P_{novelty}=0.05\,(1-d/0.001)$  if  $d<0.001$; otherwise 0",
            fontsize=10.1, color=INK, va="center")
    ax.text(5.87, 2.20,
            "Applied only with a positive score and available same-NFP distance d;\npenalty is rounded to 4 decimals.",
            fontsize=7.55, color=MUTED, va="center", linespacing=1.22)

    ax.text(5.87, 1.72,
            "Infeasible: penalties inactive  ⇒  selection = −f.",
            fontsize=8.3, color=AMBER, fontweight="bold", va="center")

    # C — payload fields and configurable feedback consumers.
    _card(ax, (11.06, 1.39), 4.76, 4.48, edge=PLUM, face="#fdfbff", radius=0.09)
    ax.text(11.31, 5.58, "C  Returned feedback", fontsize=14.0,
            color=PLUM, fontweight="bold", va="top")
    ax.text(11.31, 5.24, "EvalResult → ledger → selected feedback strategy", fontsize=8.6,
            color=MUTED, va="top")

    payloads = [
        (4.53, "HARNESS VALUE", "score · error · seconds", BLUE, BLUE_LIGHT),
        (3.78, "CORE PHYSICS", "p2_score · feasibility · objective_L", TEAL, TEAL_LIGHT),
        (3.03, "NON-PRIVATE DIAGNOSTICS",
         "aspect_ratio · max_elongation · edge_iota_per_nfp\nedge_mirror_ratio · log10_qi · shaped_score",
         TEAL, PAPER),
        (2.30, "CONDITIONAL / ACCOUNTING",
         "bank_dist · bank_cos · penalty fields\nevals_used · seconds_used · submittable",
         PLUM, PLUM_LIGHT),
    ]
    for y, title, body, color, face in payloads:
        height = 0.66 if "\n" in body else 0.57
        _card(ax, (11.33, y - height / 2), 4.22, height, edge=color,
              face=face, radius=0.045, linewidth=0.9)
        ax.text(11.48, y + (0.14 if "\n" in body else 0.11), title,
                fontsize=7.2, color=color, fontweight="bold", va="center")
        ax.text(11.48, y - (0.14 if "\n" in body else 0.12), body,
                fontsize=7.45, color=INK, va="center", linespacing=1.22)

    ax.text(11.34, 1.67, "MODE", fontsize=7.3, color=MUTED,
            fontweight="bold", va="center")
    _pill(ax, 12.35, 1.50, "score only", selected=True, color=PLUM,
          width=0.97, fontsize=7.4)
    _pill(ax, 13.43, 1.50, "reflections", color=PLUM, width=1.02,
          fontsize=7.4, face=PLUM_LIGHT)
    _pill(ax, 14.56, 1.50, "memory", color=PLUM, width=0.81,
          fontsize=7.4, face=PLUM_LIGHT)

    # Bottom contract notes.
    ax.text(0.18, 0.93, "Official P2", fontsize=8.0, color=BLUE,
            fontweight="bold", va="center")
    ax.text(1.08, 0.93,
            "the fresh pinned evaluator's benchmark score is returned unchanged; honest_score and tolerance_used are diagnostics only.",
            fontsize=8.2, color=INK, va="center")
    ax.text(0.18, 0.56, "Scope", fontsize=8.0, color=AMBER,
            fontweight="bold", va="center")
    ax.text(0.78, 0.56,
            "Penalty constants are environment-overridable and fingerprinted in experiment_spec; the diagram shows code defaults, not measured effects.",
            fontsize=8.2, color=INK, va="center")
    ax.text(15.82, 0.18, "Definitions: src/evoharness/tasks/stellar_p2/task.py",
            fontsize=7.6, color=MUTED, ha="right")
    _save(fig, "reward-feedback")


def _write_design_view(config: dict[str, Any], stellar: dict[str, Any],
                       benchmark: dict[str, Any]) -> None:
    """Persist the compact, explicit data view used by these two figures."""
    latest = stellar["latest_persisted_state"]["repair_loop"]
    payload = {
        "schema_version": 1,
        "sources": [
            "configs/stellar_p2.json",
            "src/evoharness/engine/config.py",
            "src/evoharness/tasks/stellar_p2/workflows/repair.py",
            "src/evoharness/tasks/stellar_p2/workflows/research.py",
            "reports/data/benchmark-evidence.json",
            "reports/data/stellar-evidence.json",
        ],
        "configurable_axes": [
            {"axis": name, "implemented": list(options),
             "stellar_p2_baseline": config["switches"][name]}
            for name, options in SWITCHES.items()
        ],
        "baseline_context": {
            "task": config["task"], "execution": config["execution"],
            "model": config["models"]["strong"], "objective": config["objective"],
            "memory_enabled": config["switches"]["feedback"] == "memory",
        },
        "latest_persisted_repair": {
            "run": latest["run"], "rounds": latest["rounds"],
            "evaluated": latest["evaluated"], "accepted": latest["accepted"],
            "fidelity": latest["best_fidelity"],
            "worst_violation": latest["best_feasibility"],
            "active_violation": latest["active_violation"],
            "officially_verified": latest["officially_verified"],
        },
        "plot_series": stellar["plot_series"],
        "benchmark_studies": {
            key: _study(benchmark, key)
            for key in ("initial_binpacking_5_axis_factorial",
                        "cvrp_memory_vs_score_only_seed3")
        },
    }
    (DATA / "design-view.json").write_text(json.dumps(payload, indent=2) + "\n")


def generate() -> None:
    """Generate both design figures and their explicit compact data view."""
    _setup()
    config, stellar, benchmark = _load()
    optimizer_axes(config, stellar)
    evaluation_trajectory(stellar, benchmark)
    reward_feedback()
    _write_design_view(config, stellar, benchmark)


if __name__ == "__main__":
    generate()
    print(f"Design figures written to {OUT.relative_to(ROOT)}")
