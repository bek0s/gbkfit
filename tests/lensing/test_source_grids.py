"""
Tests of the grids of planes behind lenses (gbkfit.lensing.SourceGrid):
the box of the rays with a margin, and pixels finer than those of the
image where it stretches the plane.
"""

import numpy as np
import pytest

from gbkfit.config import ConfigError
from gbkfit.grid import sky_positions
from gbkfit.lensing import (
    MARGIN, MAX_STRETCH, MIN_MARGIN_STEPS, STRETCH_QUANTILE, SourceGrid,
    source_step)


ORIGIN = (150.0, 2.0)


def centres(grid):
    x, y = sky_positions(grid)
    return x[0], y[:, 0]


def test_an_automatic_extent_is_the_box_of_the_rays_and_a_margin():
    x = np.array([-1.0, 0.5, 2.0])
    y = np.array([0.2, 1.4, 0.8])
    grid = SourceGrid().layout(x, y, 0.1, ORIGIN)
    along_x, along_y = centres(grid)
    assert grid.coords.rval == ORIGIN and grid.coords.rota == 0
    np.testing.assert_allclose(np.diff(along_x), 0.1)
    # (centred on the box, covering it and its margins)
    np.testing.assert_allclose(along_x.mean(), 0.5)
    np.testing.assert_allclose(along_y.mean(), 0.8)
    assert along_x[0] <= -1.0 - MARGIN * 3 and along_x[-1] >= 2.0 + MARGIN * 3
    assert along_x[0] > -1.0 - MARGIN * 3 - 0.1
    # (a margin of at least some pixels, about a narrow box)
    narrow = SourceGrid().layout(
        np.array([0.0, 0.01]), np.array([0.0, 0.0]), 0.1, ORIGIN)
    assert narrow.size[1] == 2 * MIN_MARGIN_STEPS + 1


def test_an_extent_given_is_the_span_of_the_centres_of_the_pixels():
    options = SourceGrid(step=0.25, extent=[-1, 1, 0, 0.5])
    grid = options.layout(np.zeros(1), np.zeros(1), 0.25, ORIGIN)
    along_x, along_y = centres(grid)
    np.testing.assert_allclose(along_x, np.arange(-1, 1.01, 0.25))
    np.testing.assert_allclose(along_y, [0, 0.25, 0.5])
    assert options.dump() == dict(step=0.25, extent=[-1.0, 1.0, 0.0, 0.5])
    assert SourceGrid().dump() == dict(step='auto', extent='auto')


def test_the_options_of_source_grids():
    with pytest.raises(ValueError, match="x0 < x1"):
        SourceGrid(extent=[1, -1, 0, 1])
    with pytest.raises(ValueError, match="a list of 4 values of type float"):
        SourceGrid(extent=[0, 1, 0])
    with pytest.raises((ValueError, ConfigError)):
        SourceGrid(step=0)
    with pytest.raises((ValueError, ConfigError)):
        SourceGrid(step='fine')


def test_the_automatic_step_follows_the_stretch_of_the_image():
    # From the pixels of the image (no stretch, or a squeeze) to
    # MAX_STRETCH times finer, of a high quantile of the stretch: a
    # critical curve, of infinite stretch at a few rays, does not set it
    assert source_step(0.1, np.array([0.5, 1.0])) == 0.1
    assert source_step(0.1, np.full(10, 4.0)) == pytest.approx(0.025)
    stretched = np.full(100, 2.0)
    stretched[:round(100 * (1 - STRETCH_QUANTILE)) - 1] = np.inf
    assert source_step(0.1, stretched) == pytest.approx(0.05)
    assert source_step(0.1, np.full(10, np.inf)) \
        == pytest.approx(0.1 / MAX_STRETCH)
