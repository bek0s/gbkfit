import numpy as np
import pytest

from gbkfit.region import (
    ApertureCircle, ApertureEllipse, ApertureField, AperturePolygon,
    ApertureRectangle, RegionsApertures, RegionsBins, regions_parser)
from gbkfit.math import overlap
from gbkfit.utils import gridutils


def _pixels(size):
    j, i = np.mgrid[0:size, 0:size]
    return i, j


def test_ellipse_pixel_overlaps_are_exact():
    i, j = _pixels(20)
    # A circle of radius 3, an ellipse rotated by 30 degrees, and a small
    # circle inside one pixel
    circle = overlap.ellipse_pixel_overlaps(
        np.eye(2) / 3, -np.array([5.2, 4.7]) / 3, i, j)
    np.testing.assert_allclose(circle.sum(), np.pi * 9, rtol=1e-12)
    angle = np.radians(30)
    axes = np.array([
        [np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    ellipse = axes / np.array([[4], [1.5]])
    areas = overlap.ellipse_pixel_overlaps(
        ellipse, -ellipse @ [10.3, 9.9], i, j)
    np.testing.assert_allclose(areas.sum(), np.pi * 6, rtol=1e-12)
    small = overlap.ellipse_pixel_overlaps(
        np.eye(2) / 0.2, -np.array([3.1, 3.05]) / 0.2, i, j)
    np.testing.assert_allclose(small[3, 3], np.pi * 0.04, rtol=1e-12)
    # A pixel inside the circle is whole
    np.testing.assert_allclose(circle[5, 5], 1, rtol=1e-12)


def test_polygon_pixel_overlaps_are_exact():
    i, j = _pixels(20)
    # A concave polygon (an L), in either order, and a polygon with a
    # slot narrower than a pixel
    shape_l = np.array([
        [2.3, 2.2], [9.7, 2.2], [9.7, 4.1], [4.4, 4.1], [4.4, 8.9],
        [2.3, 8.9]])
    for vertices in (shape_l, shape_l[::-1]):
        areas = overlap.polygon_pixel_overlaps(vertices, i, j)
        np.testing.assert_allclose(areas.sum(), 24.14, rtol=1e-12)
        np.testing.assert_allclose(areas[3, 3], 1, rtol=1e-12)
        # Pixel (3, 2) spans y from 1.5 to 2.5, the polygon from 2.2
        np.testing.assert_allclose(areas[2, 3], 0.3, rtol=1e-12)
    slot = np.array([
        [1.1, 1.1], [15.2, 1.1], [15.2, 6.3], [8.05, 6.3], [8.05, 2.0],
        [7.95, 2.0], [7.95, 6.3], [1.1, 6.3]])
    areas = overlap.polygon_pixel_overlaps(slot, i, j)
    np.testing.assert_allclose(
        areas.sum(), abs(overlap.polygon_area(slot)), rtol=1e-12)
    np.testing.assert_allclose(areas[4, 8], 0.9, rtol=1e-12)


def test_polygon_and_ellipse_overlaps_agree():
    # A circle as a polygon of many vertices
    i, j = _pixels(12)
    angles = np.linspace(0, 2 * np.pi, 20000, endpoint=False)
    vertices = np.stack([5.2 + 3 * np.cos(angles), 4.7 + 3 * np.sin(angles)], 1)
    np.testing.assert_allclose(
        overlap.polygon_pixel_overlaps(vertices, i, j),
        overlap.ellipse_pixel_overlaps(
            np.eye(2) / 3, -np.array([5.2, 4.7]) / 3, i, j),
        atol=1e-7)


def _grid(size=(40, 30), step=(0.5, 0.25), rota=0):
    return gridutils.make_grid(size, step, rota=rota)


@pytest.mark.parametrize('aperture, area', [
    (ApertureCircle(1, -0.5, 1.5), np.pi * 1.5 ** 2),
    (ApertureEllipse(-1, 0.5, 2, 1, posa=40), np.pi * 2),
    (ApertureRectangle(0.3, 0, 4, 0.6, posa=-25), 2.4),
    (AperturePolygon([[0, 0], [3, 0], [3, 1], [1, 1], [1, 2], [0, 2]]), 4),
    (ApertureField(), 20 * 7.5)])
@pytest.mark.parametrize('rota', [0, 33])
def test_aperture_overlaps_cover_its_area(aperture, area, rota):
    # The fractions of the pixels add up to the area of the aperture (on
    # the sky, the pixels are 0.5 x 0.25 arcsec)
    grid = _grid(rota=rota)
    pixels, fractions = aperture.overlaps(grid)
    assert np.all((fractions > 0) & (fractions <= 1 + 1e-12))
    np.testing.assert_allclose(fractions.sum() * 0.125, area, rtol=1e-12)


@pytest.mark.parametrize('rota, pixel', [
    (0, (20, 19)), (90, (24, 15)), (-90, (16, 15))])
def test_apertures_are_on_the_sky(rota, pixel):
    # A small circle 2 arcsec north of the reference pixel (20, 15) of a
    # grid of 0.5 arcsec pixels: with rota 0 the +y axis points north,
    # with rota 90 the +x axis does, and with rota -90 the -x axis
    grid = gridutils.make_grid((40, 30), 0.5, rpix=(20, 15), rota=rota)
    pixels, fractions = ApertureCircle(0, 2, 0.1).overlaps(grid)
    assert pixels.tolist() == [pixel[1] * 40 + pixel[0]]


def test_apertures_must_be_inside_the_grid():
    grid = _grid()
    with pytest.raises(RuntimeError, match="not inside the grid"):
        ApertureCircle(9, 0, 1.5).overlaps(grid)
    with pytest.raises(RuntimeError, match="aperture 1: .*not inside"):
        RegionsApertures([
            ApertureCircle(0, 0, 1), ApertureCircle(9, 0, 1.5)]).weights(grid)
    # Also when it is far outside (it failed with "negative dimensions")
    with pytest.raises(RuntimeError, match="not inside the grid"):
        ApertureCircle(50, 0, 1.5).overlaps(grid)


def test_regions_of_apertures():
    grid = _grid()
    apertures = [ApertureCircle(1, -0.5, 1.5), ApertureField()]
    regions = RegionsApertures(apertures)
    assert regions.nregions() == 2
    assert regions.grid() is None
    weights = regions.weights(grid)
    assert weights.shape == (2, 40 * 30)
    np.testing.assert_allclose(
        weights.sum(axis=1), [np.pi * 1.5 ** 2 / 0.125, 40 * 30])
    # Round trip through the configuration
    info = regions_parser.dump(regions)
    loaded = regions_parser.load(info)
    assert regions_parser.dump(loaded) == info
    assert (loaded.weights(grid) != weights).nnz == 0


def test_regions_of_bins(tmp_path):
    index = np.full((6, 8), -1)
    index[1:3, 1:4] = 0
    index[3:5, 2:7] = 1
    index[0, 7] = 2
    regions = RegionsBins(index, step=0.5)
    assert regions.nregions() == 3
    grid = regions.grid()
    assert grid.size == (8, 6)
    weights = regions.weights(grid).toarray()
    assert weights.sum(axis=1).tolist() == [6, 10, 1]
    np.testing.assert_array_equal(
        weights[1].reshape(6, 8), index == 1)
    # The bins are on their own grid only
    with pytest.raises(RuntimeError, match="defined on the grid"):
        regions.weights(gridutils.make_grid((8, 6)))
    # Round trip through a file
    info = regions_parser.dump(regions, prefix=str(tmp_path / ''))
    loaded = regions_parser.load(dict(type='bins', file=info['file']))
    np.testing.assert_array_equal(loaded.index(), index)
    np.testing.assert_allclose(loaded.grid().coords.step, (0.5, 0.5))
    assert loaded.grid().coords.rpix == grid.coords.rpix


@pytest.mark.parametrize('index, message', [
    (np.array([[0, 2], [-1, 0]]), "have no pixels: \\[1\\]"),
    (np.array([[0, 0.5], [-1, 0]]), "must be integers"),
    (np.full((2, 2), -1), "no pixel is in a bin"),
    (np.zeros((2, 2, 2)), "must be an image")])
def test_bins_are_checked(index, message):
    with pytest.raises(RuntimeError, match=message):
        RegionsBins(index)


def test_region_sums(driver):
    # The native sums of the regions of each channel of a cube equal the
    # product of the weights and the cube
    from gbkfit.observation.observables._regions import RegionSumsPlan
    grid = _grid()
    weights = RegionsApertures([
        ApertureCircle(1, -0.5, 1.5),
        ApertureRectangle(0.3, 0, 4, 0.6, posa=-25),
        ApertureField()]).weights(grid)
    cube = np.random.default_rng(1).random((7, 30, 40)).astype(np.float32)
    plan = RegionSumsPlan(weights, driver, np.float32)
    d_cube = driver.mem_copy_h2d(cube)
    d_out = driver.mem_alloc_d((7, 3), np.float32)
    plan.evaluate(d_cube, d_out)
    expected = (weights @ cube.reshape(7, -1).T).T
    np.testing.assert_allclose(
        driver.mem_copy_d2h(d_out), expected, rtol=1e-5)


def test_aperture_position_angle_is_north_through_east():
    # A thin rectangle at the position angle 45 lies to the north-east
    # (+y, and -x: east is to the left) and the south-west of its centre
    grid = _grid((9, 9), (1, 1))
    weights = RegionsApertures([ApertureRectangle(0, 0, 6, 0.2, 45)]).weights(
        grid).toarray().reshape(9, 9)
    # (rows are y, columns x, the centre is pixel 4)
    assert weights[4 + 2, 4 - 2] > 0 and weights[4 - 2, 4 + 2] > 0
    assert weights[4 + 2, 4 + 2] == 0 and weights[4 - 2, 4 - 2] == 0
