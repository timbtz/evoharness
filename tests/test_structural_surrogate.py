from core.structural_discovery import ConstructionSurrogate


def test_surrogate_uses_observations_but_never_promotes_without_screen():
    keys = ("aspect_ratio", "n_field_periods")
    surrogate = ConstructionSurrogate(keys)
    surrogate.add({"aspect_ratio": 8, "n_field_periods": 2}, 12.0, .1)
    take, prediction = surrogate.acquisition({"aspect_ratio": 8.1, "n_field_periods": 2}, 11.0)
    assert prediction.neighbors == 1
    assert prediction.objective_l == 12.0
    assert take
