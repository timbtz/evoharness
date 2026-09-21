"""Generate provenance-tracked Stellar P2 repair seeds from bounded recipes.

The model returns structured coefficient edits; the host applies and evaluates
them with the pinned physics image. Model text is never executed. The current
platform and generated seeds live in ``.local/runs/repair-seeds``.

Run with ``python -m evoharness.tasks.stellar_p2.workflows.factory``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from evoharness.paths import REPO_ROOT, RUNS_DIR

from evoharness.engine.ledger import BudgetGuard, Ledger  # noqa: E402
from evoharness.engine.llm import LLM  # noqa: E402

MODEL = "glm-5.2"
SCHEMA_VERSION = 1
MAX_DELTA = 2e-3
MAX_ENTRIES = 12
SEED_DIR = RUNS_DIR / "repair-seeds"
PLATFORM_PREFERRED = "qidb-nfp6-PLATFORM.json"
PLATFORM_FALLBACK = "repair2-best-0.1464.json"

FACTS = """Measured campaign facts (treat as ground truth):
- The platform's qi violation is FIRST-ORDER TRUNCATION (qi scales ~ r^2.7);
  incoherent coefficient noise has been gate-flat in every basis.
- IMPORTANT: an FD gradient-descent lane is ALREADY running on this exact
  platform and it STALLS around worst=0.53 — plain small deltas on the
  dominant modes are covered and exhausted. Your value is STRUCTURALLY
  DIFFERENT moves: phase-coherent multi-mode patterns (fixed helicity
  relations across m rows), mirror-targeting m=0 / n>=1 |B|-well content,
  and shaping aimed at the concave bean inboard section (the universal worst
  L_gradB spot). Do not propose what a gradient step would find.
- Second violation is MIRROR (0.379) — recipes that trade a little aspect
  headroom (~0.01 available) for mirror relief are welcome.
- mpol/ntor > 3 content transfers badly through VMEC here (platform is 4x7 =
  m 0..3, n -3..3); stay on existing low-order modes.
- X20-type m=2 content helped convergence and edge iota on sibling seeds.
- L >= 12 must be preserved (the mission's capital; platform L 13.67 already
  clears the 0.70 bar if worst violation reaches < 0.01)."""

PROMPT_CONTRACT = """First write a SHORT reasoning paragraph (3-6 sentences,
mechanism only). Then output ONE fenced code block, exactly:
```json
{"recipes": [
  {"idea": "<one-sentence mechanism>",
   "prediction": "<falsifiable: which violation moves, by how much>",
   "entries": [{"target": "r_cos", "m": 2, "n": 1, "delta": 0.0005}]}
]}
```
with exactly %d recipes, each <= 12 entries, every |delta| <= 2e-3, target
"r_cos" or "z_sin". Rules: coherent multi-entry moves beat single-coefficient
noise; never target m=0 with n<0 (stellarator symmetry); n runs from -%d to
%d, m from 0 to %d; do not repeat a recipe family the outcome table already
shows failing. End your reply immediately after the closing fence."""


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def load_platform(explicit: str | None):
    path = (Path(explicit) if explicit else
            SEED_DIR / PLATFORM_PREFERRED if (SEED_DIR / PLATFORM_PREFERRED).exists()
            else SEED_DIR / PLATFORM_FALLBACK)
    payload = json.loads(path.read_text())
    return path, payload.get("boundary", payload)


def evaluate(boundary: dict) -> dict:
    """Full metrics via the in-image oracle (verify_boundary's clean-room
    metrics are sparse for infeasible boundaries — L comes back None)."""
    import os
    import subprocess
    import tempfile
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=RUNS_DIR,
                                     delete=False) as f:
        json.dump({"boundary": boundary}, f)
        tmp = f.name
    try:
        rel = os.path.relpath(tmp, REPO_ROOT)
        out = subprocess.run(
            ["docker", "run", "--rm", "--cpus", "1", "--memory", "3g",
             "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
             "-v", f"{REPO_ROOT}:/work", "-w", "/work", "-e", "PYTHONPATH=/work/src",
             "evoharness-stellar-eval", "python", "-u", "-m",
             "evoharness.tasks.stellar_p2.physics.oracle", rel],
            capture_output=True, text=True, timeout=600)
        last = [l for l in out.stdout.splitlines() if l.strip().startswith("{")]
        if not last:
            return {"worst": None, "error": (out.stderr or out.stdout)[-200:]}
        return json.loads(last[-1])
    finally:
        Path(tmp).unlink(missing_ok=True)


def apply_recipe(boundary: dict, entries: list[dict]) -> dict | None:
    rc = np.array(boundary["r_cos"], float)
    zs = np.array(boundary["z_sin"], float)
    ntor = (rc.shape[1] - 1) // 2
    for e in entries:
        m, n, delta = int(e["m"]), int(e["n"]), float(e["delta"])
        if not (0 <= m < rc.shape[0]) or not (-ntor <= n <= ntor):
            return None
        if m == 0 and n < 0:
            return None
        arr = rc if e["target"] == "r_cos" else zs
        arr[m, n + ntor] += delta
    out = dict(boundary)
    out["r_cos"], out["z_sin"] = rc.tolist(), zs.tolist()
    return out


def valid_recipes(reply: str, k: int) -> list[dict]:
    try:
        if "```" in reply:  # prefer the fenced block (contract format)
            import re
            m = re.search(r"```(?:json)?\s*\n(.*?)```", reply, re.S)
            blob = m.group(1) if m else reply
        else:
            blob = reply
        blob = blob[blob.index("{"): blob.rindex("}") + 1]
        data = json.loads(blob)
    except Exception:
        return []
    out = []
    for r in (data.get("recipes") or [])[:k]:
        entries = r.get("entries") or []
        if not entries or len(entries) > MAX_ENTRIES:
            continue
        ok = True
        for e in entries:
            try:
                if e["target"] not in ("r_cos", "z_sin"):
                    ok = False
                int(e["m"]); int(e["n"])
                if abs(float(e["delta"])) > MAX_DELTA:
                    ok = False
            except Exception:
                ok = False
        if ok:
            out.append({"idea": str(r.get("idea", ""))[:400],
                        "prediction": str(r.get("prediction", ""))[:400],
                        "entries": entries})
    return out


def outcome_table(attempts: list[dict], limit: int = 12) -> str:
    rows = []
    for a in attempts[-limit:]:
        if a.get("worst") is None:
            rows.append(f"r{a['round']}.{a['slot']} DEAD({a.get('error','?')[:40]}) :: {a['idea'][:80]}")
        else:
            rows.append(f"r{a['round']}.{a['slot']} dWorst={a['d_worst']:+.5f} "
                        f"dL={a['d_L']:+.4f} worst={a['worst']:.5f} :: {a['idea'][:80]}")
    return "\n".join(rows) or "(no attempts yet)"


MOCK_REPLY = json.dumps({"recipes": [
    {"idea": "dry-run probe: tiny m=2 z_sin bump", "prediction": "worst moves < 0.01",
     "entries": [{"target": "z_sin", "m": 2, "n": 1, "delta": 5e-4}]},
    {"idea": "dry-run probe: tiny m=3 r_cos dent", "prediction": "worst moves < 0.01",
     "entries": [{"target": "r_cos", "m": 3, "n": 2, "delta": -5e-4}]},
]})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=RUNS_DIR / "zai-seed-factory-1")
    ap.add_argument("--platform", default=None)
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-usd", type=float, default=10000.0)
    ap.add_argument("--max-calls", type=int, default=3000)
    ap.add_argument("--walltime-hours", type=float, default=24.0)
    ap.add_argument("--evals-per-hour", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    a.run_dir.mkdir(parents=True, exist_ok=True)
    state_p = a.run_dir / "state.json"
    state = (json.loads(state_p.read_text()) if state_p.exists() else
             {"schema_version": SCHEMA_VERSION, "arm": "seed-factory",
              "round": 0, "attempts": [], "saved": [],
              "platform_sha": None, "platform_metrics": None})
    if state.get("schema_version") != SCHEMA_VERSION or state.get("arm") != "seed-factory":
        raise RuntimeError("incompatible state; use a new run dir")

    ledger = Ledger(a.run_dir)
    guard = BudgetGuard(a.max_usd, a.max_calls, a.walltime_hours * 3600)
    llm = LLM(ledger, guard)
    eval_times: list[float] = []

    def metered_eval(boundary):
        now = time.time()
        while len([t for t in eval_times if now - t < 3600]) >= a.evals_per_hour:
            time.sleep(60); now = time.time()
        eval_times.append(now)
        return evaluate(boundary)

    plat_path, platform = load_platform(a.platform)
    if state["platform_sha"] != _sha(platform):
        print(f"[factory] baselining platform {plat_path.name}", flush=True)
        state["platform_sha"] = _sha(platform)
        state["platform_metrics"] = metered_eval(platform)
        ledger.append({"type": "platform_baseline", "path": str(plat_path),
                       **state["platform_metrics"]})
    base = state["platform_metrics"]
    print(f"[factory] platform worst={base['worst']:.5f} L={base['L']:.4f} "
          f"aspect={base['aspect']:.4f}", flush=True)

    for _ in range(a.rounds):
        # hot-swap to an improved platform if the host dropped one
        new_path, new_platform = load_platform(a.platform)
        if _sha(new_platform) != state["platform_sha"]:
            print(f"[factory] NEW PLATFORM {new_path.name} — re-baselining", flush=True)
            platform, plat_path = new_platform, new_path
            state["platform_sha"] = _sha(platform)
            state["platform_metrics"] = metered_eval(platform)
            base = state["platform_metrics"]
            ledger.append({"type": "platform_baseline", "path": str(new_path), **base})

        state["round"] += 1
        rnd = state["round"]
        prompt = (
            f"# Seed-factory round {rnd}\n\n"
            f"Platform boundary (nfp=6, second-order QI transfer; mpol 3, ntor 3):\n"
            f"```json\n{json.dumps({'r_cos': platform['r_cos'], 'z_sin': platform['z_sin']})}\n```\n"
            f"Pinned-evaluator baseline: worst violation {base['worst']:.5f} "
            f"(violations vector {base['violations']}), L_gradB objective {base['L']:.4f}, "
            f"aspect {base['aspect']:.4f}, qi {base['qi']:.3e}.\n\n{FACTS}\n\n"
            f"## Outcome table (your recent attempts, normalized vs platform)\n"
            f"{outcome_table(state['attempts'])}\n\n"
            + PROMPT_CONTRACT % (a.k, 3, 3, 3))
        try:
            reply = MOCK_REPLY if a.dry_run else llm.chat(
                MODEL, [{"role": "user", "content": prompt}],
                temperature=0.85, role="factory", max_tokens=8192)
        except Exception as e:
            if type(e).__name__ == "BudgetExceeded":
                print("[factory] budget/walltime reached — clean exit", flush=True)
                break
            print(f"[factory] llm error: {e}", flush=True)
            time.sleep(30)
            continue

        recipes = valid_recipes(reply, a.k)
        if not recipes:
            print(f"[factory] r{rnd}: no valid recipes", flush=True)
            state["attempts"].append({"round": rnd, "slot": 0, "idea": "(parse failure)",
                                      "worst": None, "error": "no valid recipes"})
        for slot, rec in enumerate(recipes):
            cand = apply_recipe(platform, rec["entries"])
            att = {"round": rnd, "slot": slot, "idea": rec["idea"],
                   "prediction": rec["prediction"], "entries": rec["entries"]}
            if cand is None:
                att.update({"worst": None, "error": "illegal entry"})
            else:
                try:
                    met = metered_eval(cand)
                except Exception as e:
                    met = {"worst": None}
                    att["error"] = f"{type(e).__name__}: {e}"[:120]
                att.update(met)
                if met.get("worst") is not None:
                    att["d_worst"] = met["worst"] - base["worst"]
                    att["d_L"] = (met["L"] or 0) - base["L"]
                    if met["worst"] < base["worst"] - 1e-6 and (met["L"] or 0) >= 12.0:
                        h = _sha(cand)[:12]
                        out = SEED_DIR / f"factory-{h}.json"
                        out.write_text(json.dumps(
                            {"boundary": cand, "provenance": {
                                "source": "zai_seed_factory", "round": rnd,
                                "platform": plat_path.name, "idea": rec["idea"],
                                "measured": {k: met[k] for k in ("worst", "L", "qi", "aspect")},
                            }}, indent=1))
                        state["saved"].append(str(out.name))
                        print(f"[factory] *** SEED SAVED {out.name} worst={met['worst']:.5f} "
                              f"(platform {base['worst']:.5f}) L={met['L']:.4f} ***", flush=True)
            state["attempts"].append(att)
            ledger.append({"type": "factory_attempt", **{k: att.get(k) for k in
                           ("round", "slot", "idea", "worst", "L", "qi", "aspect", "error")}})
        state_p.write_text(json.dumps(state, indent=1, default=str))
        good = [x for x in state["attempts"][-a.k:] if x.get("worst") is not None]
        best = min((x["worst"] for x in good), default=None)
        print(f"[factory] r{rnd}: {len(recipes)} recipes, best worst this round "
              f"{best if best is None else round(best, 5)} (platform {base['worst']:.5f})",
              flush=True)
        if a.dry_run:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
