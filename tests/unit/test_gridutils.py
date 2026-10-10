"""
Tests for the grids of pixels with world coordinates.
"""

import astropy.wcs
import numpy as np
import pytest

from gbkfit.utils import fitsutils, gridutils
from gbkfit.utils.parseutils import ConfigError


def celestial_wcs(crval, crpix, rota):
    """A celestial WCS of 1 arcsec pixels, east to the left, rotated."""
    wcs = astropy.wcs.WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crval = crval
    wcs.wcs.crpix = crpix
    wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]
    angle = np.radians(rota)
    wcs.wcs.pc = [
        [np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    return wcs


def test_grids_are_placed_by_their_sky_positions():
    # Two grids of different reference positions and rotations: the pixel
    # on the second of each pixel of the first is that of astropy
    wcs_a = celestial_wcs([150.0, 50.0], [20, 15], 30)
    wcs_b = celestial_wcs([150.004, 50.002], [5, 8], -10)
    grid_a = gridutils.Grid(
        (40, 30), fitsutils.coords_from_wcs('a', wcs_a), None)
    grid_b = gridutils.Grid(
        (25, 20), fitsutils.coords_from_wcs('b', wcs_b), None)
    j, i = np.mgrid[0:30, 0:40]
    expected = wcs_b.world_to_pixel(wcs_a.pixel_to_world(i, j))
    np.testing.assert_allclose(
        gridutils.pixels_on(grid_a, grid_b), expected, atol=1e-6)


def test_grids_must_overlap():
    grid = gridutils.make_grid((10, 10), rval=(150.0, 2.0))
    gridutils.check_overlap(grid, grid, 'the image')
    far = gridutils.make_grid((10, 10), rval=(150.0, 2.1))
    with pytest.raises(RuntimeError, match="the image does not cover"):
        gridutils.check_overlap(grid, far, 'the image')


@pytest.mark.parametrize('options, message', [
    (dict(size=(10, 0)), "positive integers"),
    (dict(size=(10, 2.5)), "positive integers"),
    (dict(size=(10, 10), step=(1, 1, 1)), "step has 3 values"),
    (dict(size=(10, 10), step=-1), "step must be positive"),
    (dict(size=(10, 10, 5), spectral_axis=3), "spectral axis cannot be 3"),
    (dict(size=(10, 10), rest='1 GHz'), "has no rest")])
def test_make_grid_errors(options, message):
    with pytest.raises(ConfigError, match=message):
        gridutils.make_grid(**options)
