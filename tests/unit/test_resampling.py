"""
Tests of the resampling of the residuals of data on grids of pixels, for
bootstraps (see Dataset.resample_blocks and Dataset.resample_flips).
"""

import numpy as np
import pytest

from gbkfit.dataset import (
    Data, DatasetPixelBrightness, DatasetPixelMoments, DatasetPixelSpectra)
from gbkfit.utils.parseutils import ConfigError


# A grid of 7 x 5 pixels of 1 arcsec, centred on its middle pixel
SHAPE = (5, 7)
GRID = dict(step=1, rpix=(3, 2))


def residuals(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    """Residuals of a shape, each different, the first masked (NaN)."""
    r = np.random.default_rng(seed).normal(size=shape)
    r.flat[0] = np.nan
    return r


def moments() -> DatasetPixelMoments:
    """Moment maps 0 and 1 of the grid."""
    return DatasetPixelMoments(
        {order: Data(np.ones(SHAPE)) for order in (0, 1)}, **GRID)


def kept(moved: np.ndarray, r: np.ndarray) -> np.ndarray:
    """The moved residuals where they and the residuals are finite."""
    return np.where(np.isfinite(moved) & np.isfinite(r), moved, r)


def test_blocks_redraw_whole_blocks_of_residuals():
    r = residuals(SHAPE)
    drawn = moments().resample_blocks(
        dict(moment0=r, moment1=10 * r), [2, 3], np.random.default_rng(1))
    # Masked values stay masked; the others take residuals of the data,
    # each block from one block of them, the same in every item
    assert np.array_equal(np.isnan(drawn['moment0']), np.isnan(r))
    np.testing.assert_array_equal(drawn['moment1'], 10 * drawn['moment0'])
    where = {value: index for index, value in np.ndenumerate(r)}
    origins = {}
    for (j, i), value in np.ndenumerate(drawn['moment0']):
        if np.isfinite(value):
            source_j, source_i = where[value]
            origin = (source_j - j % 3, source_i - i % 2)
            assert origins.setdefault((j // 3, i // 2), origin) == origin


def test_blocks_refused_without_room():
    r = dict(moment0=residuals(SHAPE), moment1=residuals(SHAPE))
    with pytest.raises(ConfigError, match='make the blocks smaller'):
        moments().resample_blocks(r, [8, 2], np.random.default_rng())
    with pytest.raises(ConfigError, match='each of x and y'):
        moments().resample_blocks(r, [2, 2, 2], np.random.default_rng())


@pytest.mark.parametrize('posa', [0, 90])
def test_flips_of_a_whole_map(posa):
    # One block of the whole map: unchanged, rotated by 180 degrees, or
    # mirrored about the major axis (north at posa 0) or the minor one;
    # the velocity changes sign where the sides swap. The pixel mirrored
    # onto the masked one keeps its own residual.
    r0, r1 = residuals(SHAPE, 1), residuals(SHAPE, 2)
    mirrors = [
        lambda r: r, lambda r: r[::-1, ::-1],
        lambda r: r[:, ::-1], lambda r: r[::-1, :]]
    if posa == 90:
        mirrors[2:] = mirrors[3], mirrors[2]
    flips = [
        (kept(mirror(r0), r0), kept(sign * mirror(r1), r1))
        for mirror, sign in zip(mirrors, [1, -1, 1, -1])]
    seen = set()
    for seed in range(20):
        drawn = moments().resample_flips(
            dict(moment0=r0, moment1=r1), {}, [7, 5], [0, 0], posa,
            np.random.default_rng(seed))
        matches = [
            n for n, (f0, f1) in enumerate(flips)
            if np.array_equal(drawn['moment0'], f0, equal_nan=True)
            and np.array_equal(drawn['moment1'], f1, equal_nan=True)]
        assert len(matches) == 1
        seen.add(matches[0])
    assert seen == {0, 1, 2, 3}


def test_flips_mirror_the_channels_about_the_lines_of_the_model():
    # Lines of the model at the channels 20 + 2 (j - 2), rising along the
    # major axis (north): the sides swap in y, about channel 20
    shape = (41, 5, 5)
    k, j, _ = np.indices(shape)
    model = np.exp(-0.5 * ((k - 20 - 2 * (j - 2)) / 2) ** 2)
    cube = DatasetPixelSpectra(
        Data(np.ones(shape)), step=(1, 1, 10), rpix=(2, 2, 20))
    r = residuals(shape)
    flips = [kept(f, r) for f in (
        r, r[::-1, ::-1, ::-1], r[:, :, ::-1], r[::-1, ::-1, :])]
    seen = set()
    for seed in range(20):
        drawn = cube.resample_flips(
            dict(spectra=r), dict(spectra=model), [5, 5, 41], [0, 0], 0,
            np.random.default_rng(seed))['spectra']
        matches = [
            n for n, f in enumerate(flips)
            if np.array_equal(drawn, f, equal_nan=True)]
        assert len(matches) == 1
        seen.add(matches[0])
    assert seen == {0, 1, 2, 3}


def test_images_resample_too():
    r = residuals(SHAPE)
    image = DatasetPixelBrightness(Data(np.ones(SHAPE)), **GRID)
    rng = np.random.default_rng(3)
    for drawn in (
            image.resample_blocks(dict(brightness=r), [2, 2], rng),
            image.resample_flips(
                dict(brightness=r), {}, [1, 1], [0, 0], 30, rng)):
        assert np.array_equal(
            np.isnan(drawn['brightness']), np.isnan(r))
