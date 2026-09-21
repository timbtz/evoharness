from __future__ import annotations

import numpy as np
import pytest

from evoharness.tasks.stellar_p2.discovery.operators import (ModeBand, activate_band, expand_grid,
                                    geometry_screen, truncate)


def boundary():
    return {"n_field_periods": 2,
            "r_cos": [[0, 1, .02], [.1, 0, .1], [.01, .02, .01]],
            "z_sin": [[0, 0, .02], [-.1, 0, .1], [-.01, .02, .01]]}


def test_geometry_screen_checks_symmetry_and_finiteness():
    assert geometry_screen(boundary()) == (True, "ok")
    bad = boundary(); bad["r_cos"][0][0] = .1
    assert not geometry_screen(bad)[0]
    bad = boundary(); bad["z_sin"][1][1] = float("nan")
    assert not geometry_screen(bad)[0]


def test_expand_preserves_coefficients_and_center_alignment():
    b, x = boundary(), expand_grid(boundary(), 4, 3)
    assert np.asarray(x["r_cos"]).shape == (5, 7)
    assert np.array_equal(np.asarray(x["r_cos"])[:3, 2:5], np.asarray(b["r_cos"]))


def test_truncation_is_idempotent_and_keeps_grid():
    once = truncate(boundary(), 1, 0)
    assert truncate(once, 1, 0) == once
    assert np.asarray(once["r_cos"]).shape == (3, 3)


def test_band_activation_is_deterministic_and_only_adds_shell():
    band = ModeBand(3, 2, .01, 17)
    a, b = activate_band(boundary(), band), activate_band(boundary(), band)
    assert a == b and geometry_screen(a)[0]
    assert np.array_equal(np.asarray(a["r_cos"])[:3, 1:4],
                          np.asarray(boundary()["r_cos"]))
    with pytest.raises(ValueError):
        activate_band(boundary(), ModeBand(3, 2, .2, 1))
