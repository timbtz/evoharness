from __future__ import annotations

import pytest

from core.structural_discovery import (BasinRecord, BehaviorArchive, OperatorSpec,
    Provenance, census_promotable, constraint_violations, describe, enrich_metrics,
    material_gain, promotion_decision, stable_hash,
    successive_halving)
from experiments.structural_census import (candidate_code, design, design_axis_radius, design_axis_repair,
                                           load_operator_specs, retryable)


def boundary(nfp=2, tweak=0.0):
    return {"n_field_periods": nfp,
            "r_cos": [[0, 1 + tweak, 0], [0.1, 0, 0.1]],
            "z_sin": [[0, 0, 0], [-0.1, 0, 0.1]]}


def record(name, score, feasibility, nfp=2):
    p = Provenance(name, 1)
    r = BasinRecord(name, boundary(nfp), {"honest_score": score,
        "objective_L": score * 20, "feasibility": feasibility}, p, "fp", "vlf")
    describe(r)
    return r


def test_operator_contract_and_deterministic_design():
    assert [s.id for s in design(7, 3)] == [s.id for s in design(7, 3)]
    assert candidate_code(design(7, 1)[0]).count("fm.eval(") == 1
    with pytest.raises(ValueError, match="5.0%"):
        OperatorSpec("tiny", "1", {}, "polish", .01, "stop")
    # A transform acts on a converged basin, where the measured frontier puts
    # the winning move near 2% L: the construction floor would forbid it.
    OperatorSpec("mode_continuation", "1", {}, "band", .004, "stop")
    with pytest.raises(ValueError, match="0.2%"):
        OperatorSpec("mode_continuation", "1", {}, "band", .001, "stop")
    # Transform families are executable now, but only against a source basin;
    # families with no executor at all still refuse.
    inherited = OperatorSpec("mode_continuation", "1", {}, "new support band", .1, "stop")
    with pytest.raises(ValueError, match="requires a source boundary"):
        candidate_code(inherited)
    unsupported = OperatorSpec("nfp_pivot", "1", {}, "topology pivot", .1, "stop")
    with pytest.raises(ValueError, match="does not support"):
        candidate_code(unsupported)


def test_accepted_operator_file_is_an_executable_queue(tmp_path):
    spec = design(7, 1)[0]
    path = tmp_path / "accepted.json"
    from dataclasses import asdict
    path.write_text(__import__("json").dumps({"accepted": [{"operator": asdict(spec)}]}))
    loaded, rejected = load_operator_specs(path)
    assert loaded == [spec] and rejected == []


def test_axis_codesign_is_deterministic_and_executable():
    specs = [s for s in design(9, 2, 3) if s.family == "nae_axis"]
    assert specs == [s for s in design(9, 2, 3) if s.family == "nae_axis"]
    assert all("seed_nae_axis" in candidate_code(s) for s in specs)
    repair = design_axis_repair(10, 5)
    assert len(repair) == 5 and all(s.parameters["n_field_periods"] == 3 for s in repair)
    assert all(0 <= s.parameters["mirror_ratio"] <= .08 for s in repair)
    radius = design_axis_radius(11, 5)
    assert min(s.parameters["aspect_ratio"] for s in radius) >= 9.5


def test_independent_provenance_closure():
    with pytest.raises(ValueError, match="independent"):
        Provenance("op", 1, ("public-hash",), True, False)
    with pytest.raises(ValueError, match="independent"):
        Provenance("op", 1, (), True, True)


def test_behavior_archive_replaces_only_cell_champion_and_keeps_history():
    archive = BehaviorArchive()
    weak, strong = record("a", .50, .005), record("b", .51, .005)
    assert archive.add(weak)
    assert archive.add(strong)
    assert list(archive.champions.values()) == [strong]
    assert archive.history == [weak, strong]


def test_successive_halving_preserves_diverse_cells_before_ranking():
    records = [record("weak2", .50, .005, 2), record("strong2", .60, .005, 2),
               record("nfp3", .55, .005, 3)]
    kept = successive_halving(records, 2)
    assert {r.job_id for r in kept} == {"strong2", "nfp3"}


def test_materiality_and_stable_hash():
    assert material_gain(.66, .64)
    assert not material_gain(.649, .64)
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})
    assert census_promotable(record("repair", .5, .05))
    assert not census_promotable(record("bad", .1, 1.5))
    assert retryable("timeout after 75s")
    assert not retryable("Minimum of B is at the boundary")


def test_constraint_decomposition_and_selection_score():
    metrics = enrich_metrics({"aspect_ratio": 10.1, "edge_iota_per_nfp": .25,
        "log10_qi": -4, "edge_mirror_ratio": .2, "max_elongation": 5,
        "feasibility": .01, "p2_score": .65, "shaped_score": .65})
    assert constraint_violations(metrics)["aspect"] == pytest.approx(.01)
    assert metrics["active_violation"] == "aspect"
    assert metrics["selection_score"] == pytest.approx(.65 - .92 * .008)


def test_fidelity_promotion_cannot_bypass_lf():
    candidate = record("promising", .67, .05)
    assert promotion_decision(candidate, "vlf").next_fidelity == "low_fidelity"
    assert not promotion_decision(candidate, "low_fidelity", lf_rank_reversed=True).promote
    assert promotion_decision(candidate, "low_fidelity").next_fidelity == "official"
    assert promotion_decision(candidate, "official").next_fidelity is None


def test_construction_surrogate_is_uncertain_and_local():
    from core.structural_discovery import ConstructionSurrogate
    s = ConstructionSurrogate(("aspect_ratio", "n_field_periods"))
    ok, prior = s.acquisition({"aspect_ratio": 8.0, "n_field_periods": 2}, 10.0)
    assert ok and prior.neighbors == 0
    s.add({"aspect_ratio": 8.0, "n_field_periods": 2}, 12.0, .1)
    p = s.predict({"aspect_ratio": 8.0, "n_field_periods": 2})
    assert p.objective_l == 12.0 and p.feasibility == .1
    assert s.predict({"aspect_ratio": 8.0, "n_field_periods": 3}).uncertainty > 0.0


def test_finite_difference_policy_requires_promoted_basin():
    from core.structural_discovery import FiniteDifferencePolicy, BasinRecord, Provenance
    policy = FiniteDifferencePolicy()
    rec = BasinRecord("x", {}, {"objective_L": 13.3, "honest_score": .4, "feasibility": .1},
                      Provenance("x", 1, True, False), "fp", "vlf")
    assert policy.eligible(rec)
    assert policy.coordinates(100) == 12
