#!/usr/bin/env python3
"""Rebuild reproducible charts from committed evidence tables.

AI-generated system illustrations are preserved separately with their prompts.

Run with `uv run --group analysis evoharness figures`.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evoharness.paths import REPO_ROOT

ROOT = REPO_ROOT
OUT = ROOT / "reports/figures"
INK, MUTED = "#15243b", "#53647c"
BLUE, TEAL, PLUM, AMBER = "#3164c6", "#087e81", "#7952a5", "#af6516"


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": INK, "axes.labelcolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "path", "svg.hashsalt": "evoharness",
                         "pdf.fonttype": 42})


def local_fonts():
    """Package redistributable fonts so the offline explorer is self-contained."""
    from fontTools.ttLib import TTFont
    source = Path(matplotlib.get_data_path()) / "fonts/ttf"
    target = OUT / "fonts"
    target.mkdir(exist_ok=True)
    for name in ("DejaVuSans", "DejaVuSans-Bold"):
        font = TTFont(source / f"{name}.ttf", recalcTimestamp=False)
        font.flavor = "woff"
        font.save(target / f"{name}.woff")
    license_text = (source / "LICENSE_DEJAVU").read_text()
    (target / "LICENSE_DEJAVU.txt").write_text(
        "\n".join(line.rstrip() for line in license_text.splitlines()).rstrip() + "\n"
    )


def save(fig, name):
    for suffix in ("svg", "pdf", "png"):
        metadata = {"Creator": "EvoHarness / src/evoharness/analysis/render_figures.py"}
        if suffix == "svg":
            metadata["Date"] = None
        elif suffix == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        fig.savefig(OUT / f"{name}.{suffix}", dpi=150, bbox_inches="tight", facecolor="white",
                    metadata=metadata if suffix != "png" else None)
    svg = OUT / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def screening():
    with (ROOT / "reports/data/screening-binpacking.csv").open() as f:
        rows=list(csv.DictReader(f))
    axes=[("feedback","score_only","reflections"),("gate","public_only","holdout"),
          ("search","greedy","islands"),("knowledge","off","wiki_fs"),
          ("roles","single_strong","split_roles")]
    values=[]
    labels=[]
    for key,a,b in axes:
        av=np.array([float(r["private"]) for r in rows if r[key]==a])
        bv=np.array([float(r["private"]) for r in rows if r[key]==b])
        values.append(bv.mean()-av.mean()); labels.append(f"{b} − {a}")
    fig,ax=plt.subplots(figsize=(10,4.5))
    colors=[TEAL if v>=0 else AMBER for v in values]
    ax.barh(range(len(values)),values,color=colors,height=.58)
    ax.axvline(0,color=MUTED,lw=1)
    ax.set_yticks(range(len(values)),labels);ax.invert_yaxis()
    ax.set_xlabel("Difference in mean private score (higher is better)")
    ax.set_title("Initial bin-packing screen  /  five design choices",loc="left",weight="bold",pad=22)
    for i,v in enumerate(values):
        ax.text(v+(0.0003 if v>=0 else -0.0003),i,f"{v:+.4f}",va="center",ha="left" if v>=0 else "right",fontsize=10)
    span=max(abs(v) for v in values); ax.set_xlim(-span*1.7,span*1.7)
    ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
    fig.text(.01,.015,f"{len(rows)} runs · original 32 configurations × 3 seeds · descriptive marginal means; interactions and uncertainty are not resolved",fontsize=8.5,color=MUTED)
    fig.tight_layout(rect=(0,.065,1,1));save(fig,"screening-effects")


def trajectory():
    phases=[("01 / JUL 18","Five-axis workbench","Small tasks\nMeasured splits",BLUE),
            ("02 / JUL 22–27","Memory + campaigns","Predictions and analyst\nBranch / merge",BLUE),
            ("03 / JUL 27–AUG 06","Audit the objective","Seed provenance\nShaping and gradients",TEAL),
            ("04 / AUG 13","Audit the evaluator","Reject fidelity artifacts\nWiden the search",AMBER),
            ("05 / AUG 15+","Research + repair","Constructor families\nImported seeds\nEvidence review",PLUM)]
    fig,ax=plt.subplots(figsize=(15,3.7));ax.set(xlim=(0,15),ylim=(0,3.7));ax.axis("off")
    ax.text(.1,3.32,"A research trajectory, not a monotonic score curve",fontsize=20,weight="bold")
    for i,(date,title,body,color) in enumerate(phases):
        x=.1+i*3
        ax.plot([x+.15,x+2.85],[2.65,2.65],color=color,lw=3)
        ax.text(x+.03,2.3,date,color=color,fontsize=10,weight="bold")
        ax.text(x+.03,1.96,title,fontsize=11,weight="bold",va="top")
        ax.text(x+.03,1.53,body,fontsize=9.5,color=MUTED,linespacing=1.6,va="top")
    ax.text(.12,.3,"Commit history supports stages 1–4. Stage 5 includes later working-tree code and persisted research runs; see the development history for evidence.",fontsize=9,color=MUTED)
    save(fig,"research-trajectory")


def hero():
    (OUT / "hero.svg").write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="360" viewBox="0 0 1440 360" role="img" aria-labelledby="title desc">
<title id="title">EvoHarness — agentic optimization, examined</title><desc id="desc">Research workbench for executable discovery, centered on stellarator design. Decorative orbital lines represent iterative search.</desc>
<defs><linearGradient id="bg" x2="1" y2="1"><stop stop-color="#10213c"/><stop offset="1" stop-color="#103d47"/></linearGradient><linearGradient id="line"><stop stop-color="#54ddca"/><stop offset="1" stop-color="#9ba7ff"/></linearGradient></defs>
<rect width="1440" height="360" rx="20" fill="url(#bg)"/>
<g fill="none" stroke="url(#line)" stroke-width="1.6" opacity=".42" transform="translate(1170 180)">
<ellipse rx="170" ry="67" transform="rotate(0)"/><ellipse rx="170" ry="67" transform="rotate(30)"/><ellipse rx="170" ry="67" transform="rotate(60)"/><ellipse rx="170" ry="67" transform="rotate(90)"/><ellipse rx="170" ry="67" transform="rotate(120)"/><ellipse rx="170" ry="67" transform="rotate(150)"/>
<circle r="112" stroke-dasharray="3 9"/><circle r="13" fill="#65e2d0" stroke="none"/></g>
<g font-family="DejaVu Sans,Arial,sans-serif"><text x="64" y="68" fill="#77dfcf" font-size="15" letter-spacing="4">EXECUTABLE DISCOVERY / OPEN RESEARCH</text><text x="60" y="158" fill="#fff" font-size="76" font-weight="700" letter-spacing="-3">EvoHarness</text><text x="64" y="213" fill="#e5edf6" font-size="28">Agentic optimization, examined.</text><text x="64" y="258" fill="#b6cbd7" font-size="19">Search architectures · evaluator feedback · scientific novelty</text>
<rect x="64" y="290" width="225" height="32" rx="16" fill="#255b62"/><text x="80" y="312" fill="#a1f1df" font-size="13">◉ STELLARATOR CASE STUDY</text><text x="313" y="312" fill="#b6cbd7" font-size="13">6 TASKS / EVIDENCE FIRST / ACTIVELY EVOLVING</text></g></svg>''')


if __name__ == "__main__":
    import argparse
    from evoharness.analysis.design_figures import generate as design_figures

    argparse.ArgumentParser(description=__doc__).parse_args()
    setup(); local_fonts(); hero(); screening(); trajectory()
    design_figures()
    print(f"Figures written to {OUT.relative_to(ROOT)}")
