from typing import Any

import astropy.io.fits
import numpy as np

from gbkfit.region import Regions, regions_parser
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _detail
from .base import Dataset
from .data import Data, dump_data, load_data


__all__ = [
    'DatasetRegionMoments'
]


class DatasetRegionMoments(Dataset):
    """
    Moments of the spectra in regions of the sky (see Regions; e.g.
    Voronoi bins, fibres): some of the orders 0 to 7, as the data items
    moment0 to moment7.

    Parameters
    ----------
    regions : Regions
        The regions.
    moment0, ..., moment7 : Data, optional
        The moments, each a vector of one value for each region; at least
        one.

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
        # (the data files of the moments are vectors)
        parseutils.load_option_and_update_info(
            regions_parser, info, 'regions', required=True, prefix=prefix)
        for name in [f'moment{i}' for i in range(8)]:
            # (an item that is null is absent)
            if info.get(name) is not None:
                with parseutils.config_path(name):
                    info[name] = load_data(info[name], prefix)[0]
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

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
            regions: Regions,
            moment0: Data | None = None,
            moment1: Data | None = None,
            moment2: Data | None = None,
            moment3: Data | None = None,
            moment4: Data | None = None,
            moment5: Data | None = None,
            moment6: Data | None = None,
            moment7: Data | None = None
    ):
        moments = (
            moment0, moment1, moment2, moment3, moment4, moment5, moment6,
            moment7)
        super().__init__({
            f'moment{order}': moment for order, moment in enumerate(moments)
            if moment is not None})
        if self.shape() != (regions.nregions(),):
            raise ConfigError(
                f"the moments have {self.shape()[0]} values, but there are "
                f"{regions.nregions()} regions")
        self._regions = regions

    def regions(self) -> Regions:
        """Return the regions."""
        return self._regions
