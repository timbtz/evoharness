from __future__ import annotations

import pytest

from evoharness.tasks.stellar_p2.discovery.boundaries import (BasinRecord, BehaviorArchive, OperatorSpec,
    ParetoArchive, Provenance, boundary_bank_kin, boundary_distance, census_promotable, constraint_violations, describe, enrich_metrics,
    material_gain, promotion_decision, stable_hash,
    screen_resolution, successive_halving)
from evoharness.tasks.stellar_p2.discovery.census import (candidate_code, design, design_axis_radius, design_axis_repair,
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
    # Independent descendants may have independent parents; ancestry, not the
    # mere existence of a parent hash, is what determines clean-room status.
    Provenance("op", 1, ("independent-parent",), True, False, False)
    with pytest.raises(ValueError, match="independent"):
        Provenance("op", 1, ("public-hash",), True, False, True)
    with pytest.raises(ValueError, match="independent"):
        Provenance("op", 1, (), True, True)


def test_behavior_archive_replaces_only_cell_champion_and_keeps_history():
    archive = BehaviorArchive()
    weak, strong = record("a", .50, .005), record("b", .51, .005)
    assert archive.add(weak)
    assert archive.add(strong)
    assert list(archive.champions.values()) == [strong]
    assert archive.history == [weak, strong]


def test_boundary_distance_and_pareto_archive_preserve_stepping_stones():
    def physical(name, objective, qi):
        metrics = {"objective_L": objective, "honest_score": 0.0,
            "feasibility": qi, "aspect_ratio": 8.0, "edge_iota_per_nfp": .3,
            "log10_qi": -4 + qi * 4, "edge_mirror_ratio": .1,
            "max_elongation": 3.0}
        result = BasinRecord(name, boundary(2, .1 if name == "high" else 0),
                             enrich_metrics(metrics), Provenance(name, 1), "fp", "vlf")
        describe(result); return result
    high_l, good_qi = physical("high", 13.0, .2), physical("repair", 10.0, .02)
    archive = ParetoArchive(minimum_distance=.003)
    assert archive.add(high_l) and archive.add(good_qi)
    assert {r.job_id for r in archive.records} == {"high", "repair"}
    assert boundary_distance(boundary(2), boundary(3)) == float("inf")
    assert high_l.descriptor.constraint_bins and high_l.descriptor.shape_bin


def test_boundary_distance_ignores_scale_and_discrete_parameter_gauges():
    base = boundary(2)
    scaled = {**base, "r_cos": [[2 * x for x in row] for row in base["r_cos"]],
              "z_sin": [[2 * x for x in row] for row in base["z_sin"]]}
    # theta -> theta+pi flips odd-m rows but describes the identical surface.
    relabelled = {**base,
        "r_cos": [[((-1) ** m) * x for x in row]
                  for m, row in enumerate(base["r_cos"])],
        "z_sin": [[((-1) ** m) * x for x in row]
                  for m, row in enumerate(base["z_sin"])]}
    assert boundary_distance(base, scaled) == pytest.approx(0)
    assert boundary_distance(base, relabelled) == pytest.approx(0)
    distance, cosine = boundary_bank_kin(relabelled, [{"boundary": base}])
    assert distance == pytest.approx(0) and cosine == pytest.approx(1)


def test_screen_resolution_reproduces_pinned_vlf_and_official_presets():
    low = screen_resolution(boundary(2))
    assert (low["vlf_mpol"], low["official_mpol"], low["delta_m"]) == (3, 10, -7)
    high = boundary(2)
    high["r_cos"] = [[0.0] * 17 for _ in range(9)]
    high["z_sin"] = [[0.0] * 17 for _ in range(9)]
    high["r_cos"][0][8] = 1.0
    high["r_cos"][8][16] = 1e-14
    resolution = screen_resolution(high)
    assert resolution["official_comparable"]
    assert (resolution["vlf_mpol"], resolution["vlf_ntor"]) == (10, 10)


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
    # Its support-dependent LF grid is not comparable to official, so a merely
    # repairable result cannot use L/honest thresholds to buy an official solve.
    assert not promotion_decision(candidate, "low_fidelity").promote
    candidate.metrics["feasibility"] = .005
    assert promotion_decision(candidate, "low_fidelity").next_fidelity == "official"
    assert promotion_decision(candidate, "official").next_fidelity is None


def test_construction_surrogate_is_uncertain_and_local():
    from evoharness.tasks.stellar_p2.discovery.boundaries import ConstructionSurrogate
    s = ConstructionSurrogate(("aspect_ratio", "n_field_periods"))
    ok, prior = s.acquisition({"aspect_ratio": 8.0, "n_field_periods": 2}, 10.0)
    assert ok and prior.neighbors == 0
    s.add({"aspect_ratio": 8.0, "n_field_periods": 2}, 12.0, .1)
    p = s.predict({"aspect_ratio": 8.0, "n_field_periods": 2})
    assert p.objective_l == 12.0 and p.feasibility == .1
    assert s.predict({"aspect_ratio": 8.0, "n_field_periods": 3}).uncertainty > 0.0


def test_finite_difference_policy_requires_promoted_basin():
    from evoharness.tasks.stellar_p2.discovery.boundaries import FiniteDifferencePolicy, BasinRecord, Provenance
    policy = FiniteDifferencePolicy()
    rec = BasinRecord("x", {}, {"objective_L": 13.3, "honest_score": .4, "feasibility": .1},
                      Provenance("x", 1, independent=True,
                                 public_bank_enabled=False), "fp", "vlf")
    assert not policy.eligible(rec)  # missing resolution provenance fails closed
    rec.metrics["screen_resolution"] = {"delta_m": 0, "delta_n": 0}
    assert policy.eligible(rec)
    assert policy.coordinates(100) == 12
