from __future__ import annotations

import json

import pytest

from core.structural_llm import OperatorDesigner, Proposal, parse_json_object


def valid():
    return {"family": "nae", "version": "1", "parameters": {"n_field_periods": 2},
            "mechanism": "topology changes field-period scaling",
            "expected_l_gain_fraction": .1,
            "expected_constraint_effects": {k: "measure" for k in
                ("aspect", "iota", "qi", "mirror", "elongation")},
            "novelty_mechanism": "independent construction", "kill_criterion": "stop at 8",
            "falsifiable_prediction": "one of eight reaches L above 10"}


class FakeLLM:
    def __init__(self, replies): self.replies = iter(replies)
    def chat(self, *args, **kwargs): return next(self.replies)


class Events:
    def __init__(self): self.rows = []
    def append(self, row): self.rows.append(row)


def test_proposal_schema_and_security():
    good = valid(); good["parameters"] = {"aspect_ratio": 8, "max_elongation": 3,
        "rotational_transform": .6, "mirror_ratio": .1, "n_field_periods": 2,
        "max_poloidal_mode": 3, "max_toroidal_mode": 3}
    assert Proposal.parse(good).operator.family == "nae"
    bad = valid(); bad["parameters"] = {"seed_bank": 0}
    with pytest.raises(ValueError, match="security"):
        Proposal.parse(bad)
    bad = valid(); del bad["expected_constraint_effects"]["qi"]
    with pytest.raises(ValueError, match="five"):
        Proposal.parse(bad)


def test_json_fence_parser():
    assert parse_json_object("```json\n{\"a\": 1}\n```") == {"a": 1}


def test_writer_and_critic_must_both_accept():
    payload = valid(); payload["parameters"] = {"aspect_ratio": 8, "max_elongation": 3,
        "rotational_transform": .6, "mirror_ratio": .1, "n_field_periods": 2,
        "max_poloidal_mode": 3, "max_toroidal_mode": 3}
    events = Events()
    d = OperatorDesigner(FakeLLM([json.dumps(payload), json.dumps(
        {"accept": True, "reasons": [], "risk": "medium"})]), "w", "c", events)
    proposal, verdict = d.propose("none")
    assert proposal and verdict["accept"] and events.rows[-1]["type"] == "structural_proposal"


def test_duplicate_is_rejected_without_second_critic_call():
    value = valid(); value["parameters"] = {"aspect_ratio": 8, "max_elongation": 3,
        "rotational_transform": .6, "mirror_ratio": .1, "n_field_periods": 2,
        "max_poloidal_mode": 3, "max_toroidal_mode": 3}
    payload = json.dumps(value)
    events = Events(); llm = FakeLLM([payload, json.dumps(
        {"accept": True, "reasons": [], "risk": "low"}), payload])
    d = OperatorDesigner(llm, "w", "c", events)
    assert d.propose("none")[0]
    assert d.propose("none")[0] is None


def test_invalid_writer_gets_one_bounded_schema_repair():
    value = valid(); value["parameters"] = {"aspect_ratio": 8, "max_elongation": 3,
        "rotational_transform": .6, "mirror_ratio": .1, "n_field_periods": 2,
        "max_poloidal_mode": 3, "max_toroidal_mode": 3}
    events = Events(); d = OperatorDesigner(FakeLLM(["not json", json.dumps(value),
        json.dumps({"accept": True, "reasons": [], "risk": "low"})]), "w", "c", events)
    assert d.propose("none")[0] is not None


def test_allowed_family_rejection_cannot_leak_stale_proposal(tmp_path):
    import json
    from core.ledger import Ledger
    from core.structural_llm import OperatorDesigner
    class Bad:
        def chat(self, *args, role=None, **kwargs):
            if role == "structural_writer":
                return json.dumps({"family":"nae_axis","version":"x","parameters":{
                    "aspect_ratio":8,"max_elongation":3,"rotational_transform":.5,
                    "mirror_ratio":.1,"n_field_periods":2,"max_poloidal_mode":3,
                    "max_toroidal_mode":3,"torsion":.1},"mechanism":"independent",
                    "expected_l_gain_fraction":.1,"expected_constraint_effects":{
                    "aspect":"x","iota":"x","qi":"x","mirror":"x","elongation":"x"},
                    "novelty_mechanism":"new","kill_criterion":"cheap",
                    "falsifiable_prediction":"test"})
            return json.dumps({"accept":True,"reasons":[],"risk":"low"})
    d=OperatorDesigner(Bad(),"m","m",Ledger(tmp_path)); d.allowed_families={"ellipse"}
    proposal, verdict=d.propose("evidence")
    assert proposal is None and not verdict["accept"]
