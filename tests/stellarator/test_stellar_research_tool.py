import json

import pytest

from evoharness.tasks.stellar_p2.workflows.evaluate import (_signed, _verify_signature,
                                                audit_candidate)


def test_open_ended_candidate_audit_preserves_methods_but_blocks_bank_and_io():
    good = """def solve(fm, rng):
    boundary = fm.seed_nae(aspect_ratio=8, max_elongation=3,
        rotational_transform=.6, mirror_ratio=.1, n_field_periods=2,
        max_poloidal_mode=3, max_toroidal_mode=3)
    return boundary
"""
    assert audit_candidate(good) == []
    assert "public-bank access forbidden: seed_bank" in audit_candidate(
        "def solve(fm, rng): return fm.seed_bank(0)")
    assert "forbidden call: open" in audit_candidate(
        "def solve(fm, rng):\n    open('/tmp/x')\n    return {}")
    assert "forbidden import: subprocess" in audit_candidate(
        "import subprocess\ndef solve(fm, rng): return {}")
    assert "forbidden import: builtins" in audit_candidate(
        "import builtins\ndef solve(fm, rng): return builtins.open('/tmp/x')")
    assert "embedded raw boundary coefficients" in audit_candidate(
        "def solve(fm, rng): return {'r_cos': [[1]], 'z_sin': [[0]]}")[0]
    numbers = ",".join(str(i) for i in range(24))
    assert any("large numeric strings" in error for error in audit_candidate(
        f"def solve(fm, rng):\n    x = '{numbers}'\n    return fm.seed_nae()"))


def test_candidate_must_define_solve():
    assert audit_candidate("x = 1") == ["candidate must define solve(fm, rng)"]


def test_physics_artifact_signature_fails_closed_on_tampering(tmp_path):
    record = _signed({"candidate_hash": "abc", "score": .5}, tmp_path)
    _verify_signature(record, tmp_path)
    record["score"] = .7
    with pytest.raises(RuntimeError, match="signature"):
        _verify_signature(record, tmp_path)
    assert len(json.loads(json.dumps(record))["artifact_signature"]) == 64
