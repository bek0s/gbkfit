"""
The dataset of moments of the spectra in regions of the sky (e.g.
Voronoi bins, fibres).
"""

from collections.abc import Mapping, Sequence
from typing import Any

import astropy.io.fits
import numpy as np

from gbkfit.config import FilePath, check_options, config_path
from gbkfit.io import read_table
from gbkfit.region import Regions, regions_parser
from . import _detail, _outliers, _resampling
from .base import Dataset
from .data import Data, DataColumns, DataFiles, dump_data


__all__ = [
    'DatasetRegionMoments'
]


# The axes of its blocks of residuals (see resample_blocks)
_BLOCK_AXES = ('the regions',)


class DatasetRegionMoments(Dataset):
    """
    Moments of the spectra in regions of the sky (see Regions; e.g.
    Voronoi bins, fibres): central moments of some of the orders 0 to 7, as
    the data items moment0 to moment7, and the coefficients h3 and h4 of a
    Gauss-Hermite series, as the items h3 and h4 (see
    _detail.moment_items).

    Its configuration has the options of from_files (the regions, and the
    files of the data items), or those of from_table (the regions, a table
    file, and the columns of the data items).

    Parameters
    ----------
    moments : Mapping
        The central moment of each order, a vector of one value for each
        region.
    regions : Regions
        The regions.
    hermite : Mapping, optional
        The coefficient of each order (3 and 4) of a Gauss-Hermite series,
        as moments; with moments, at least one.

    Raises
    ------
    ValueError
        If the moments do not have a value for each region.
    """

    ndim = 1

    config_constructors = ('from_files', 'from_table')

    @staticmethod
    def type() -> str:
        return 'region_moments'

    @classmethod
    def from_files(
            cls,
            regions: Regions,
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
            outliers: Mapping[str, Any] | None = None
    ) -> 'DatasetRegionMoments':
        """
        Read the dataset from FITS files, each a vector of one value for
        each region.

        Parameters
        ----------
        regions : Regions
            The regions.
        moment0, ..., moment7, h3, h4 : DataFiles, optional
            The files of each moment given (at least one),
            and their options (e.g. their unit; by default, that of the
            header).
        outliers : Mapping, optional
            Screen the data for outliers by the options size and nsigma
            (see screen_outliers), which are masked.

        Returns
        -------
        DatasetRegionMoments
            The dataset.
        """
        check_options(DatasetRegionMoments.from_files, locals())
        items, _ = _detail.read_moments(_detail.moment_options(locals()))
        moments, hermite = _detail.split_moments(items)
        dataset = cls(moments, regions, hermite)
        return _detail.cleaned(dataset, outliers)

    @classmethod
    def from_table(
            cls,
            file: FilePath,
            regions: Regions,
            format: str | None = None,
            moment0: DataColumns | None = None,
            moment1: DataColumns | None = None,
            moment2: DataColumns | None = None,
            moment3: DataColumns | None = None,
            moment4: DataColumns | None = None,
            moment5: DataColumns | None = None,
            moment6: DataColumns | None = None,
            moment7: DataColumns | None = None,
            h3: DataColumns | None = None,
            h4: DataColumns | None = None,
            outliers: Mapping[str, Any] | None = None
    ) -> 'DatasetRegionMoments':
        """
        Read the dataset from the columns of a table of one row for each
        region, in their order (e.g. the V, sigma, h3 and h4 of the bins of
        pPXF).

        Parameters
        ----------
        file : str
            The table file: any that astropy reads (see
            read_table: e.g. FITS, ECSV or CSV).
        regions : Regions
            The regions.
        format : str, optional
            The format of the file, if astropy cannot tell it.
        moment0, ..., moment7, h3, h4 : DataColumns, optional
            The columns of each moment given (at least one).
        outliers : Mapping, optional
            Screen the data for outliers by the options size and nsigma
            (see screen_outliers), which are masked.

        Returns
        -------
        DatasetRegionMoments
            The dataset.
        """
        check_options(DatasetRegionMoments.from_table, locals())
        table = read_table(file, format)
        items = {}
        for name, columns in _detail.moment_options(locals()).items():
            with config_path(name):
                items[name] = _detail.table_item(table, columns)
        if not items:
            raise ValueError("a dataset needs at least one moment")
        moments, hermite = _detail.split_moments(items)
        dataset = cls(moments, regions, hermite)
        return _detail.cleaned(dataset, outliers)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        def write(filename: str, array: np.ndarray) -> None:
            astropy.io.fits.writeto(filename, array, overwrite=overwrite)
        info = dict(
            type=self.type(),
            regions=regions_parser.dump(
                self._regions, prefix=prefix, dump_path=dump_path,
                overwrite=overwrite))
        for key, item in self.items():
            info[key] = dump_data(
                item, _detail.item_filenames(prefix, key), write, dump_path)
        return info

    def __init__(
            self,
            moments: Mapping[int, Data],
            regions: Regions,
            hermite: Mapping[int, Data] | None = None
    ):
        check_options(DatasetRegionMoments.__init__, locals())
        super().__init__(_detail.moment_items(moments, hermite))
        if self.shape() != (regions.nregions(),):
            raise ValueError(
                f"the moments have {self.shape()[0]} values, but there are "
                f"{regions.nregions()} regions")
        self._regions = regions

    def regions(self) -> Regions:
        """Return the regions."""
        return self._regions

    def orders(self) -> tuple[int, ...]:
        """Return the orders of the moments."""
        return _detail.moment_orders(self)

    def check_blocks(self, size: Sequence[float]) -> None:
        """
        Raise ValueError unless the size is of blocks of regions: a positive
        integer for each of the regions (see resample_blocks).
        """
        _resampling.block_size(size, _BLOCK_AXES)

    def resample_blocks(
            self,
            residuals: Mapping[str, np.ndarray],
            size: Sequence[float],
            rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """
        Return residuals of its moments redrawn in blocks of regions (see
        Dataset.resample_blocks), of the size (regions): runs of consecutive
        regions, in their order, so of 1 region unless their order follows the
        sky (e.g. fibres in a row). The noise that neighbouring regions share
        (e.g. bins smaller than the beam) is then not kept.
        """
        return _resampling.resample_blocks(
            residuals, size, _BLOCK_AXES, rng)

    def screen_outliers(
            self, size: Sequence[float], nsigma: float
    ) -> 'DatasetRegionMoments':
        """
        Return a copy of it with the outliers of its moments masked (see
        Dataset.screen_outliers): of the values of the regions nearest each, by
        their centres (see Regions.centres), of the size (number of nearest
        regions).
        """
        nnearest, _ = _outliers.region_size(
            size, self._regions.nregions(), False)
        nsigma = _outliers.check_nsigma(nsigma)
        centres = self._regions.centres()
        return self.with_data({
            key: np.where(_outliers.region_outliers(
                item.data()[None], centres, nnearest, 1, nsigma)[0],
                np.nan, item.data())
            for key, item in self.items()})
