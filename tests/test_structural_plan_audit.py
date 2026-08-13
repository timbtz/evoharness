from experiments.structural_plan_audit import audit


def test_plan_audit_is_fail_closed_and_large_run_is_ready_only_without_blockers():
    result = audit()
    assert result["counts"].get("invalid", 0) == 0
    live = next(r for r in result["requirements"] if r["id"] == "large-live")
    assert result["large_run_ready"] == (not live["blockers_remaining"])
    assert live["blockers_remaining"] == []
