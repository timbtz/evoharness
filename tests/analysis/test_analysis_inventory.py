"""Evidence export must preserve failures and never leak raw model conversations."""
import json

from evoharness.analysis.build_run_inventory import clean, scan_jsonl


def test_inventory_counts_invalid_rows_without_exporting_prompts(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text('\n'.join([
        json.dumps({"type": "run_start", "config": {"task": "circles", "seed": 2, "secret": "hidden"}, "experiment_spec": {}}),
        json.dumps({"type": "llm_call", "model": "writer", "messages": ["private conversation"]}),
        json.dumps({"type": "candidate", "accepted": True, "code": "private code"}),
        '{"truncated":',
        '[]',
        json.dumps({"type": "run_end", "private": float('-inf'), "stop_reason": "budget", "render": "large payload"}),
    ]))
    result = scan_jsonl(path, tmp_path)
    assert result["rows"] == 6
    assert result["malformed_rows"] == 1
    assert result["non_object_rows"] == 1
    assert result["accepted_candidates"] == 1
    assert result["has_run_end"] is True
    assert result["summary"]["private"] is None
    exported = json.dumps(result, allow_nan=False)
    assert all(value not in exported for value in ("hidden", "private conversation", "private code", "large payload"))


def test_partial_run_is_not_reported_as_finished(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"type":"llm_call","model":"m","ts":12}\n')
    result = scan_jsonl(path, tmp_path)
    assert result["has_run_end"] is False
    assert "summary" not in result
    assert result["models"] == {"m": 1}
    assert len(result["sha256"]) == 64


def test_nonfinite_values_are_explicit_missing_data():
    assert clean({"a": [float('nan'), float('inf'), -1.5]}) == {"a": [None, None, -1.5]}
