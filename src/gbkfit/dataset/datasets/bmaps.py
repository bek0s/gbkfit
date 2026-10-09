import astropy.io.fits

from gbkfit.dataset.core import Dataset
from gbkfit.dataset.data import Data, dump_data, load_data
from gbkfit.dataset.regions import Regions, regions_parser
from gbkfit.utils import parseutils
from . import _detail


__all__ = [
    'DatasetBMaps'
]


class DatasetBMaps(Dataset):
    """
    Moments of the spectra in regions of the sky (see Regions; e.g.
    Voronoi bins, fibres): the data items mmap0 to mmap7, each a vector of
    one value for each region.
    """

    _ndim = 1

    @staticmethod
    def type():
        return 'bmaps'

    @classmethod
    def load(cls, info, prefix=''):
        """The data files of the moments are vectors (one value per region)."""
        desc = parseutils.make_typed_desc(cls, 'dataset')
        parseutils.load_option_and_update_info(
            regions_parser, info, 'regions', True, False, prefix=prefix)
        for name in [f'mmap{i}' for i in range(8)]:
            if name in info:
                with parseutils.config_path(name):
                    info[name] = load_data(info[name], prefix)[0]
        return cls(**parseutils.parse_options_for_callable(
            info, desc, cls.__init__))

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
            mmap0: Data | None = None,
            mmap1: Data | None = None,
            mmap2: Data | None = None,
            mmap3: Data | None = None,
            mmap4: Data | None = None,
            mmap5: Data | None = None,
            mmap6: Data | None = None,
            mmap7: Data | None = None
    ):
        """The moments of the given orders, in the given regions."""
        mmaps = (mmap0, mmap1, mmap2, mmap3, mmap4, mmap5, mmap6, mmap7)
        super().__init__({
            f'mmap{order}': mmap for order, mmap in enumerate(mmaps)
            if mmap is not None})
        if self.shape() != (regions.nregions(),):
            raise RuntimeError(
                f"the moments have {self.shape()[0]} values, but there are "
                f"{regions.nregions()} regions")
        self._regions = regions

    def regions(self) -> Regions:
        return self._regions
