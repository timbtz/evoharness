from experiments.structural_archive_tournament import distance, select


def b(nfp=2, x=.1):
    return {"n_field_periods": nfp, "r_cos": [[0, 1, 0], [x, 0, x]],
            "z_sin": [[0, 0, 0], [-x, 0, x]]}


def test_archive_control_selection_is_behaviorally_distinct():
    rows = [{"key": "a", "p2": .6, "boundary": b(x=.1)},
            {"key": "copy", "p2": .59, "boundary": b(x=.101)},
            {"key": "other", "p2": .5, "boundary": b(3, .1)}]
    assert [r["key"] for r in select(rows, 2, .003)] == ["a", "other"]
    assert distance(rows[0]["boundary"], rows[2]["boundary"]) == float("inf")
