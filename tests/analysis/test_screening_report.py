from __future__ import annotations

import pytest

from evoharness.engine.config import Config
from evoharness.analysis import report, screening


def _row(**switches):
    return {"task": "binpacking", "private": 0.0, **switches}


def test_original_screening_space_preserves_32_binary_combo_ids():
    configs = list(screening.combos())
    assert len(configs) == 32
    assert {screening.combo_id(config) for config in configs} == {
        f"{n:05b}" for n in range(32)
    }
    assert all(Config(switches=dict(config)) for config in configs)


def test_current_screening_space_has_only_valid_configs_and_168_cells():
    configs = list(screening.combos("current"))
    assert len(configs) == 168
    assert all(Config(switches=dict(config)) for config in configs)
    assert not any(config["knowledge"] == "web" and config["feedback"] != "memory"
                   for config in configs)


def test_main_effects_reports_multiple_options_and_finite_denominators():
    rows = [
        _row(feedback="score_only", gate="public_only", search="greedy",
             knowledge="off", roles="single_strong", private=1.0),
        _row(feedback="reflections", gate="public_only", search="greedy",
             knowledge="off", roles="single_strong", private=3.0),
        _row(feedback="memory", gate="public_only", search="greedy",
             knowledge="off", roles="single_strong", private=5.0),
        _row(feedback="memory", gate="public_only", search="greedy",
             knowledge="off", roles="single_strong", private=float("nan")),
        _row(feedback="memory", gate="public_only", search="greedy",
             knowledge="off", roles="single_strong", private=None),
    ]
    effect = next(item for item in report.main_effects(rows) if item["axis"] == "feedback")
    assert effect["baseline"]["option"] == "score_only"
    assert effect["baseline"]["mean"] == 1.0
    reflection, memory = effect["comparisons"]
    assert (reflection["option"], reflection["mean"], reflection["delta"]) == (
        "reflections", 3.0, 2.0)
    assert memory["option"] == "memory"
    assert memory["mean"] == 5.0
    assert memory["delta"] == 4.0
    assert (memory["observed"], memory["finite"], memory["missing"], memory["nonfinite"]) == (3, 1, 1, 1)
    text = report.markdown(rows)
    assert "n=1/3 finite; missing=1; nonfinite=1" in text
    assert "memory 5.0000 → delta +4.0000" in text


def test_main_effects_rejects_mixed_task_pooling():
    base = {axis: options[0] for axis, options in report.SWITCHES.items()}
    other = {**base, "feedback": "reflections"}
    rows = [
        {"task": "binpacking", "private": 1.0, **base},
        {"task": "cvrp", "private": 2.0, **other},
    ]
    with pytest.raises(ValueError, match="cannot pool main effects across tasks"):
        report.main_effects(rows)


def test_main_effects_excludes_finite_failure_sentinel():
    base = {axis: options[0] for axis, options in report.SWITCHES.items()}
    bad = {**base, "feedback": "reflections"}
    rows = [
        {"task": "tsp", "private": 2.0, **base},
        {"task": "tsp", "private": -1.7976931348623157e308, **bad},
    ]
    effect = next(item for item in report.main_effects(rows) if item["axis"] == "feedback")
    failed = effect["comparisons"][0]
    assert failed["mean"] is None
    assert (failed["finite"], failed["nonfinite"], failed["sentinel"]) == (0, 0, 1)
    assert "failure_sentinel=1" in report.markdown(rows)
