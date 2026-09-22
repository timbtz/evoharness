from __future__ import annotations

import json

from evoharness.tasks.stellar_p2.discovery.boundaries import OperatorSpec, construction_acquisition
from evoharness.tasks.stellar_p2.discovery.evidence import (choose_family, failure_class,
                                      feedback_evidence)
from evoharness.tasks.stellar_p2.discovery.designer import OperatorDesigner, Proposal
from evoharness.tasks.stellar_p2.discovery.basins import collect_candidates
from evoharness.tasks.stellar_p2.discovery.census import candidate_code
from evoharness.tasks.stellar_p2.discovery.novelty import audit_leaders


class CaptureLLM:
    def __init__(self, proposal):
        self.proposal = proposal
        self.prompts = []

    def chat(self, model, messages, temperature, role=None, **kwargs):
        self.prompts.append((role, messages[0]["content"]))
        if role == "structural_critic":
            return json.dumps({"accept": True, "reasons": [], "risk": "medium"})
        return json.dumps(self.proposal)


class Events:
    def append(self, row):
        pass


def nae_proposal():
    return {"family": "nae", "version": "feedback-v1", "parameters": {
        "aspect_ratio": 8.0, "max_elongation": 3.0,
        "rotational_transform": .6, "mirror_ratio": .1,
        "n_field_periods": 2, "max_poloidal_mode": 3,
        "max_toroidal_mode": 3}, "mechanism": "independent topology change",
        "expected_l_gain_fraction": .08,
        "expected_constraint_effects": {k: "measure" for k in
            ("aspect", "iota", "qi", "mirror", "elongation")},
        "novelty_mechanism": "clean-room construction",
        "kill_criterion": "one fresh screen",
        "falsifiable_prediction": "L improves eight percent"}


def test_feedback_is_first_and_reaches_writer_and_critic():
    rounds = [{"round": 1, "status": "evaluated",
        "operator": {"family": "ellipse", "version": "bad", "parameters": {"x": 1}},
        "metrics": {"objective_L": 9.0, "feasibility": .7,
                    "active_violation": "qi"}}]
    evidence = feedback_evidence(rounds, extra="x" * 20_000)
    llm = CaptureLLM(nae_proposal())
    designer = OperatorDesigner(llm, "w", "c", Events())
    assert designer.propose(evidence)[0]
    assert len(evidence) <= 12_000
    assert all("L=9.0" in prompt and "catastrophic:qi" in prompt
               for _, prompt in llm.prompts)


def test_failure_class_and_adaptive_family_breadth():
    assert failure_class({"status": "failed", "error": "VMEC not converged fsqr=1e-3"}) == \
        "non-converged"
    assert failure_class({"status": "failed",
                          "error": "Minimum of B is at the boundary."}) == "geometry-rejected"
    assert failure_class({"status": "failed",
                          "error": "isn't spectrally condensed enough"}) == "geometry-rejected"
    assert choose_family([], {"nae", "ellipse"}) == "ellipse"
    first = [{"status": "failed", "operator": {"family": "ellipse"},
              "error": "timeout"}]
    assert choose_family(first, {"nae", "ellipse"}) == "nae"


def test_new_independent_generator_families_are_typed_and_executable():
    axis = nae_proposal()
    axis.update({"family": "axis_boundary_codesign", "version": "axis-1"})
    axis["parameters"].update({"torsion": .1, "radial_scale": 1.02,
                               "vertical_scale": .98, "mode_decay": .5})
    axis_spec = Proposal.parse(axis).operator
    code = candidate_code(axis_spec)
    assert "seed_nae_axis" in code and "R0 stays pinned" in code

    ellipse = nae_proposal()
    ellipse.update({"family": "ellipse_qi_repair", "version": "repair-1"})
    ellipse["parameters"] = {"aspect_ratio": 8.0, "elongation": 3.0,
        "rotational_transform": .6, "n_field_periods": 2,
        "repair_amplitude": .001, "repair_mode": 1, "phase_seed": 7}
    ellipse_spec = Proposal.parse(ellipse).operator
    assert "one deterministic low-order helical" in candidate_code(ellipse_spec)


def test_surrogate_gate_stays_off_until_enough_matching_physics():
    spec = OperatorSpec("nae", "x", nae_proposal()["parameters"], "mechanism", .08, "kill")
    ok, info = construction_acquisition([], spec)
    assert ok and not info["active"]


def test_novelty_audit_and_tournament_keep_provenance_separate():
    boundary = {"n_field_periods": 2, "r_cos": [[0, 1, 0], [.1, 0, .1]],
                "z_sin": [[0, 0, 0], [-.1, 0, .1]]}
    audit = audit_leaders([{"job_id": "i", "boundary": boundary}],
                          lambda _: (.004, .999))
    assert audit[0]["robust_novelty"]
    missing = audit_leaders([{"job_id": "i", "boundary": boundary}],
                            lambda _: (None, None))
    assert not missing[0]["robust_novelty"]
    assert missing[0]["audit_error"] == "no same-NFP public reference"
    reports = [
        {"arm": "independent", "leaders": [{"job_id": "same", "boundary": boundary,
            "vlf_metrics": {"selection_score": .5}}]},
        {"arm": "inherited", "leaders": [{"job_id": "same", "boundary": boundary,
            "vlf_metrics": {"selection_score": .6}}]},
    ]
    assert {row["arm"] for row in collect_candidates(reports)} == {"independent", "inherited"}
