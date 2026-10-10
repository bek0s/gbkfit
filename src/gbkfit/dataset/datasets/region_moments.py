import astropy.io.fits

from gbkfit.dataset.base import Dataset
from gbkfit.dataset.data import Data, dump_data, load_data
from gbkfit.region import Regions, regions_parser
from gbkfit.utils import parseutils
from . import _detail


__all__ = [
    'DatasetRegionMoments'
]


class DatasetRegionMoments(Dataset):
    """
    Moments of the spectra in regions of the sky (see Regions; e.g.
    Voronoi bins, fibres): the data items moment0 to moment7, each a vector of
    one value for each region.
    """

    ndim = 1

    @staticmethod
    def type():
        return 'region_moments'

    @classmethod
    def load(cls, info, prefix=''):
        """The data files of the moments are vectors (one value per region)."""
        parseutils.load_option_and_update_info(
            regions_parser, info, 'regions', required=True, prefix=prefix)
        for name in [f'moment{i}' for i in range(8)]:
            # (an item that is null is absent)
            if info.get(name) is not None:
                with parseutils.config_path(name):
                    info[name] = load_data(info[name], prefix)[0]
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self, prefix='', dump_path=True, overwrite=False):
        def write(filename, array):
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
        """The moments of the given orders, in the given regions."""
        moments = (
            moment0, moment1, moment2, moment3, moment4, moment5, moment6,
            moment7)
        super().__init__({
            f'moment{order}': moment for order, moment in enumerate(moments)
            if moment is not None})
        if self.shape() != (regions.nregions(),):
            raise RuntimeError(
                f"the moments have {self.shape()[0]} values, but there are "
                f"{regions.nregions()} regions")
        self._regions = regions

    def regions(self) -> Regions:
        return self._regions
