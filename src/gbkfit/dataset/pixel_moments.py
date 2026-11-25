"""
The dataset of moments of the spectra on a grid of pixels: moment maps.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from gbkfit.config import (
    ConfigProblem, DataFile, check_options, config_path, split_data_file)
from gbkfit.grid import Rota, Rpix, Rval, Step
from gbkfit.io import coords_from_header, header_shape
from gbkfit.moments import MOMENT_ITEMS, moment_orders, split_moments
from . import _detail, _grids
from .base import DatasetGrid, NoiseCorrelation, cleaned
from .deferred import Deferred
from .data import Data, DataFiles


__all__ = [
    'DatasetPixelMoments'
]


class DatasetPixelMoments(DatasetGrid):
    """
    Moments of the spectra on a grid of pixels: moment maps. They are
    maps of central moments of some of the orders 0 to 7, as the data
    items moment0 to moment7, and of the coefficients h3 and h4 of a
    Gauss-Hermite series, as the items h3 and h4 (see
    _detail.moment_items).

    Its configuration has those data items (the files of each: data,
    mask, error; see from_files), the world coordinates and the footprint,
    one for all the maps (see DatasetGrid).

    Parameters
    ----------
    moments : Mapping
        The map of the central moment of each order, of shape (ny, nx).
    hermite : Mapping, optional
        The map of the coefficient of each order (3 and 4) of a
        Gauss-Hermite series; with moments, at least one map.
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the grid (see Coords); their
        defaults are those of make_grid.
    rota : float, optional
        The rotation of the grid on the sky (see Coords).
    footprint : ndarray or Deferred, optional
        Where the instrument looked, of shape (ny, nx) (see DatasetGrid).
    """

    # Moment maps have no spectral axis
    ndim = 2
    spectral_axis = None

    # The axes of its blocks and windows (see resample_blocks and
    # screen_outliers)
    block_axes = ('x', 'y')

    config_titles = _detail.MOMENT_TITLES

    config_constructors = ('from_files',)

    @staticmethod
    def type() -> str:
        return 'pixel_moments'

    @staticmethod
    def type_title() -> str:
        return 'Pixel moments (maps)'

    @classmethod
    def from_files(
            cls,
            moment0: DataFiles | None = None,
            moment1: DataFiles | None = None,
            moment2: DataFiles | None = None,
            moment3: DataFiles | None = None,
            moment4: DataFiles | None = None,
            moment5: DataFiles | None = None,
            moment6: DataFiles | None = None,
            moment7: DataFiles | None = None,
            h3: DataFiles | None = None,
            h4: DataFiles | None = None,
            step: Step | None = None,
            rpix: Rpix | None = None,
            rval: Rval | None = None,
            rota: Rota | None = None,
            outliers: Mapping[str, Any] | None = None,
            footprint: DataFile | None = None
    ) -> 'DatasetPixelMoments':
        """
        Read the dataset from FITS files, with the world coordinates of the
        files of the moments unless given; they must agree.

        Parameters
        ----------
        moment0, ..., moment7, h3, h4 : DataFiles, optional
            The files of the map of each moment given (at least one), and
            their options (e.g. their unit; by default, that of the
            header).
        step, rpix, rval : float or Sequence of float, optional
            The world coordinates of the grid (see Coords); by
            default, those of the headers. Either rpix or rval can be
            given, and the other comes from the headers (see
            coords_from_header).
        rota : float, optional
            The rotation of the grid on the sky; by default, that of the
            headers.
        outliers : Mapping, optional
            Screen the data for outliers by the options size and nsigma
            (see screen_outliers), which are masked.
        footprint : str or tuple, optional
            The file of where the instrument looked, one for all the maps
            (see DatasetGrid).

        Returns
        -------
        DatasetPixelMoments
            The dataset.
        """
        check_options(DatasetPixelMoments.from_files, locals())
        files = _detail.moment_options(locals())
        items, coords = _detail.read_moments(
            files, rpix, rval, _grids.stated_options(step, rpix, rval))
        for name, item in items.items():
            with config_path(name):
                _grids.check_axes(item.shape(), cls.ndim)
        dataset = cls(
            *split_moments(items),
            **_grids.grid_coords(coords, step, rota),
            footprint=_grids.read_footprint(
                footprint, next(iter(files.values())).data))
        return cleaned(dataset, outliers)

    @classmethod
    def file_problems(
            cls, options: Mapping[str, Any]) -> list[ConfigProblem]:
        # (of the files of each map, each on the pixels of the first map,
        # and the footprint on them)
        problems = []
        first = None
        for name in MOMENT_ITEMS:
            if (files := options.get(name)) is None:
                continue
            found, model = _grids.item_file_problems(files, cls.ndim, {})
            problems += [
                problem._replace(path=[name, *problem.path])
                for problem in found]
            if model is None or len(header_shape(model)) != cls.ndim:
                continue
            try:
                coords = coords_from_header(
                    split_data_file(files.data)[0], model,
                    rpix=options.get('rpix'), rval=options.get('rval'))
            except ValueError:
                continue
            map_ = (name, header_shape(model), coords)
            if first is None:
                first, first_data = map_, files.data
            else:
                problems += _grids.problems_of(
                    _detail.check_on_first, *map_, *first)
        if first is not None:
            problems += _grids.footprint_problems(
                options.get('footprint'), first_data, {})
        return problems

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return _grids.dump_grid_dataset(self, prefix, dump_path, overwrite)

    def __init__(
            self,
            moments: Mapping[int, Data],
            hermite: Mapping[int, Data] | None = None,
            step: Step | None = None,
            rpix: Rpix | None = None,
            rval: Rval | None = None,
            rota: Rota | None = None,
            footprint: np.ndarray | Deferred | None = None
    ):
        check_options(DatasetPixelMoments.__init__, locals())
        super().__init__(
            _detail.moment_items(moments, hermite), step, rpix, rval, rota,
            footprint=footprint)

    def orders(self) -> tuple[int, ...]:
        """Return the orders of the moments."""
        return moment_orders(self)

    def noise_correlation(self, key: str) -> NoiseCorrelation | None:
        """
        Return the correlation of the noise of a data item on the
        separations of its pixels (see Dataset.noise_correlation).
        """
        return _grids.grid_noise_correlation(self, key)
