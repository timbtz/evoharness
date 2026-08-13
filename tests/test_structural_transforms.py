"""Executable transform families: the inherited arm the schema always named."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.stellar_operators import dilate, geometry_screen, reconstruct
from core.structural_discovery import OperatorSpec
from core.structural_llm import TRANSFORM_FAMILIES, Proposal
from experiments.structural_census import (apply_transform, candidate_code,
                                           near_converged)


def source() -> dict:
    path = ROOT / "runs/composite_polish_campaign/champion/best_boundary.json"
    payload = json.loads(path.read_text())
    return payload.get("boundary", payload)


def spec(family: str, **parameters) -> OperatorSpec:
    return OperatorSpec(family, "test-v1", parameters, "mechanism", .05, "kill")


def test_dilate_pins_the_major_radius_and_scales_shaping():
    base = source()
    out = dilate(base, 1.1, 1.1, 0.)
    ntor = (len(base["r_cos"][0]) - 1) // 2
    assert out["r_cos"][0][ntor] == pytest.approx(base["r_cos"][0][ntor])
    assert out["r_cos"][1][ntor] == pytest.approx(base["r_cos"][1][ntor] * 1.1)
    assert geometry_screen(out)[0]


def test_dilate_decay_leaves_high_order_modes_alone():
    base = source()
    out = dilate(base, 1.5, 1.5, 1.)     # full decay: only order-1 modes move
    assert out["r_cos"][3][0] == pytest.approx(base["r_cos"][3][0])


def test_dilate_rejects_scales_outside_the_documented_range():
    for bad in ((.4, 1., 0.), (1., 2.5, 0.), (1., 1., 1.5)):
        with pytest.raises(ValueError):
            dilate(source(), *bad)


def test_reconstruct_keeps_the_core_and_rebuilds_the_shell():
    base = source()
    out = reconstruct(base, 2, 2, .01, 7)
    ntor = (len(base["r_cos"][0]) - 1) // 2
    assert out["r_cos"][1][ntor] == pytest.approx(base["r_cos"][1][ntor])
    # Outside the core the spectrum is rebuilt, not copied.
    assert out["r_cos"][5][0] != pytest.approx(base["r_cos"][5][0])
    assert geometry_screen(out)[0]


def test_reconstruct_is_deterministic_in_its_phase_seed():
    assert reconstruct(source(), 2, 2, .01, 7) == reconstruct(source(), 2, 2, .01, 7)
    assert reconstruct(source(), 2, 2, .01, 7) != reconstruct(source(), 2, 2, .01, 8)


def test_apply_transform_enforces_the_exact_parameter_schema():
    with pytest.raises(ValueError, match="parameters must be exactly"):
        apply_transform(spec("structural_dilation", radial_scale=1.1), source())


def test_apply_transform_rejects_an_unknown_family():
    with pytest.raises(ValueError, match="does not support"):
        apply_transform(spec("nfp_pivot", n_field_periods=2), source())


def test_candidate_code_requires_a_source_for_transform_families():
    with pytest.raises(ValueError, match="requires a source boundary"):
        candidate_code(spec("structural_dilation", radial_scale=1.1,
                            vertical_scale=1.1, mode_decay=0.))


def test_candidate_code_embeds_the_transformed_boundary():
    code = candidate_code(spec("mode_continuation", max_poloidal_mode=8,
                               max_toroidal_mode=8, band_amplitude=.01,
                               phase_seed=3), source())
    assert "fm.eval(boundary)" in code and "seed_bank" not in code
    assert "n_field_periods" in code


def test_near_converged_separates_a_cut_off_solve_from_a_diverged_one():
    assert near_converged("VMEC not converged (fsqr=2.2e-09 fsqz=4.6e-11 fsql=5.9e-12)")
    assert not near_converged("VMEC not converged (fsqr=2.9e-02 fsqz=2.2e-02 fsql=8.0e-05)")
    assert not near_converged("Minimum of B is at the boundary.")
    assert not near_converged(None)


def transform_proposal(**parameters) -> dict:
    return {"family": "structural_dilation", "version": "v1",
            "parameters": parameters, "mechanism": "coordinated shaping dilation",
            "expected_l_gain_fraction": .07,
            "expected_constraint_effects": {"aspect": "up", "iota": "flat",
                                            "qi": "flat", "mirror": "flat",
                                            "elongation": "up"},
            "novelty_mechanism": "moves the basin off the public spectrum",
            "kill_criterion": "retire if qi degrades",
            "falsifiable_prediction": "L rises by >=5% at fixed qi"}


def test_transform_proposal_is_accepted_and_marked_inherited():
    proposal = Proposal.parse(transform_proposal(
        radial_scale=1.1, vertical_scale=1.05, mode_decay=.2))
    assert proposal.operator.family in TRANSFORM_FAMILIES
    assert proposal.operator.independent is False


def test_transform_proposal_out_of_range_is_rejected_before_physics():
    with pytest.raises(ValueError, match="radial_scale outside"):
        Proposal.parse(transform_proposal(radial_scale=9., vertical_scale=1.,
                                          mode_decay=.2))
