import json

import pytest

import evoharness.tasks.stellar_p2.workflows.repair as R
from evoharness.tasks.stellar_p2.workflows.repair import (ARM, MODEL, _load_state,
                                                  audit_repair_candidate,
                                                  build_prompt, load_seed_dir,
                                                  seed_manifest, seed_preamble)


def _write_seed(d, name="seedA", nfp=2):
    (d / f"{name}.json").write_text(json.dumps({
        "boundary": {"r_cos": [[0.0, 1.0, 0.1]], "z_sin": [[0.0, 0.0, 0.1]],
                     "n_field_periods": nfp, "is_stellarator_symmetric": True},
        "meta": {"variant": "r2c0", "nfp": nfp, "provenance": "nae2 batch2"}}))


def test_arm_is_database_repair_on_one_model_with_named_seeds(tmp_path):
    _write_seed(tmp_path)
    seeds = load_seed_dir(tmp_path)
    prompt = build_prompt("task contract", {"attempts": []}, seeds, None, 1)
    assert MODEL == "glm-5.2" and ARM == "database-repair"
    assert "feasibility-first" in prompt
    assert "`seedA`" in prompt and "nae2 batch2" in prompt
    assert "load_seed" in prompt
    assert "public seed bank is technically disabled" in prompt
    assert "web_search at least once" in prompt


def test_audit_allows_seed_references_but_keeps_all_prohibitions():
    ok = ("def solve(fm, rng):\n"
          "    b = load_seed('seedA')\n"
          "    names = seed_names()\n"
          "    return b\n")
    assert audit_repair_candidate(ok) == []
    dump = ("def solve(fm, rng):\n"
            "    b = {'r_cos': [" + ",".join(["0.01"] * 30) + "]}\n"
            "    return b\n")
    assert any("boundary coefficients" in e or "numeric blocks" in e
               for e in audit_repair_candidate(dump))
    bank = "def solve(fm, rng):\n    return fm.seed_bank(0)\n"
    assert any("public-bank" in e for e in audit_repair_candidate(bank))


def test_preamble_provides_deepcopy_seed_access(tmp_path):
    _write_seed(tmp_path)
    seeds = load_seed_dir(tmp_path)
    ns: dict = {}
    exec(seed_preamble(seeds), ns)
    b1 = ns["load_seed"]("seedA")
    b1["r_cos"][0][0] = 99.0
    assert ns["load_seed"]("seedA")["r_cos"][0][0] == 0.0
    assert ns["seed_names"]() == ["seedA"]


def test_state_rejects_arm_mismatch_and_seed_mutation(tmp_path):
    _write_seed(tmp_path)
    seeds = load_seed_dir(tmp_path)
    manifest = seed_manifest(seeds)
    settings = {"max_evals": 16}
    run = tmp_path / "run"
    run.mkdir()
    state = _load_state(run, settings, "fp", manifest)
    assert state["arm"] == ARM and state["seed_manifest"] == manifest

    (run / "state.json").write_text(json.dumps({**state, "arm": "independent"}))
    with pytest.raises(RuntimeError, match="incompatible"):
        _load_state(run, settings, "fp", manifest)

    (run / "state.json").write_text(json.dumps(state))
    mutated = [dict(manifest[0], sha256="0" * 64)]
    with pytest.raises(RuntimeError, match="append-only"):
        _load_state(run, settings, "fp", mutated)
    # additive seeds are fine
    grown = manifest + [{"name": "seedB", "sha256": "1" * 64, "provenance": "{}"}]
    assert _load_state(run, settings, "fp", grown)["seed_manifest"] == grown


def test_steering_hook_reaches_the_prompt(tmp_path, monkeypatch):
    _write_seed(tmp_path)
    seeds = load_seed_dir(tmp_path)
    steer = tmp_path / "STEERING.md"
    steer.write_text("Attack edge mirror via d(phi) content first.")
    monkeypatch.setattr(R, "STEERING_PATH", steer)
    prompt = build_prompt("task", {"attempts": []}, seeds, None, 2)
    assert "Directive from the host operator" in prompt
    assert "Attack edge mirror" in prompt
    monkeypatch.setattr(R, "STEERING_PATH", tmp_path / "missing.md")
    assert "Directive from the host operator" not in build_prompt(
        "task", {"attempts": []}, seeds, None, 3)
