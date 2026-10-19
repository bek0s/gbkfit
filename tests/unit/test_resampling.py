"""
Tests of the resampling of the residuals of data on grids of pixels, for
bootstraps (see Dataset.resample_blocks).
"""

import numpy as np
import pytest

from gbkfit.dataset import (
    Data, DatasetPixelBrightness, DatasetPixelMoments, DatasetPixelSpectra,
    DatasetSlitlessSpectra, DatasetSlitSpectra)
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


@pytest.mark.parametrize('dataset, key, size', [
    (DatasetPixelBrightness(Data(np.ones(SHAPE))), 'brightness', [2, 2]),
    (DatasetPixelSpectra(Data(np.ones((6, *SHAPE)))), 'spectra', [2, 2, 3]),
    (DatasetSlitSpectra(Data(np.ones((6, 7)))), 'spectra', [3, 2]),
    (DatasetSlitlessSpectra(Data(np.ones(SHAPE))), 'spectra', [2, 2]),
])
def test_grids_of_every_kind_redraw_in_blocks(dataset, key, size):
    dataset.check_blocks(size)
    r = residuals(dataset.shape())
    drawn = dataset.resample_blocks(
        {key: r}, size, np.random.default_rng(3))[key]
    assert np.array_equal(np.isnan(drawn), np.isnan(r))
    assert np.isin(drawn[np.isfinite(r)], r).all()
    with pytest.raises(ConfigError, match='positive integer for each of'):
        dataset.check_blocks(size + [1])
