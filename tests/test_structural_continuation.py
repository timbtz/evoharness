from experiments.structural_continuation import accept_band


def test_continuation_requires_material_effect():
    base = {"objective_L": 10, "feasibility": .5}
    assert accept_band(base, {"objective_L": 10.21, "feasibility": .5}) == \
        (True, "material L gain")
    assert accept_band(base, {"objective_L": 10.01, "feasibility": .39}) == \
        (True, "material feasibility repair")
    assert not accept_band(base, {"objective_L": 10.01, "feasibility": .49})[0]
