from evoharness.tasks.stellar_p2.discovery.reproduce import command, verify


def test_reproduction_fails_closed_on_source_drift():
    spec = {"source_sha256": {"tasks/stellar_p2/discovery/boundaries.py": "wrong"},
            "operators": None}
    assert verify(spec) == [
        "source fingerprint differs: tasks/stellar_p2/discovery/boundaries.py"]


def test_reproduction_command_preserves_fidelity_gate(tmp_path):
    spec = {"seed": 1, "per_arm": 2, "retries": 1, "promote": 3,
            "max_jobs": 4, "lf": True, "operators": None}
    cmd = command(spec, tmp_path)
    assert "--lf" in cmd and cmd[cmd.index("--max-jobs") + 1] == "4"
