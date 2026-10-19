from collections.abc import Mapping, Sequence
from typing import Any

import astropy.io.fits
import numpy as np

from gbkfit.region import Regions, regions_parser
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _detail, _resampling
from .base import Dataset
from .data import Data, FitsFile, dump_data


__all__ = [
    'DatasetRegionMoments'
]


# The axes of its blocks of residuals (see resample_blocks)
_BLOCK_AXES = ('the regions',)


class DatasetRegionMoments(Dataset):
    """
    Moments of the spectra in regions of the sky (see Regions; e.g.
    Voronoi bins, fibres): some of the orders 0 to 7, as the data items
    moment0 to moment7.

    Its configuration has the regions, and the data items moment0 to
    moment7 (the files of each: data, mask, error; see from_files).

    Parameters
    ----------
    moments : Mapping
        The moments of each order, a vector of one value for each region;
        at least one.
    regions : Regions
        The regions.

    Raises
    ------
    ConfigError
        If the moments do not have a value for each region.
    """

    ndim = 1

    @staticmethod
    def type() -> str:
        return 'region_moments'

    @classmethod
    def load(
            cls, info: dict[str, Any], prefix: str = ''
    ) -> 'DatasetRegionMoments':
        info = dict(info)
        parseutils.load_option_and_update_info(
            regions_parser, info, 'regions', required=True, prefix=prefix)
        files = _detail.pop_moment_options(info, prefix)
        return _detail.load_with_files(cls, info, **files)

    @classmethod
    def from_files(
            cls,
            moments: Mapping[int, FitsFile],
            regions: Regions,
            masks: Mapping[int, FitsFile] | None = None,
            errors: Mapping[int, FitsFile | float] | None = None,
            units: Mapping[int, str] | None = None,
            beams: Mapping[int, Mapping[str, Any]] | None = None
    ) -> 'DatasetRegionMoments':
        """
        Read the dataset from FITS files, each a vector of one value for
        each region.

        Parameters
        ----------
        moments : Mapping
            The file of the moments of each order: a filename, or a
            filename and the HDU.
        regions : Regions
            The regions.
        masks : Mapping, optional
            The files of the masks of some of the moments.
        errors : Mapping, optional
            The files of the errors of some of the moments, or one error
            for all the values of each.
        units : Mapping, optional
            The units of some of the moments; by default, those of the
            headers (see Data.from_files).
        beams : Mapping, optional
            The options of the beams of some of the moments (see Beam); by
            default, those of the headers.

        Returns
        -------
        DatasetRegionMoments
            The dataset.
        """
        items, _ = _detail.read_moments(moments, masks, errors, units, beams)
        return cls(items, regions)

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

    def __init__(self, moments: Mapping[int, Data], regions: Regions):
        super().__init__(_detail.moment_items(moments))
        if self.shape() != (regions.nregions(),):
            raise ConfigError(
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
        Raise ConfigError unless the size is of blocks of regions: a positive
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
