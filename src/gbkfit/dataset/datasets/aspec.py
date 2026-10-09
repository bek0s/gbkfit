from numbers import Real

import astropy.units

from gbkfit.dataset.core import Dataset
from gbkfit.dataset.data import Data, dump_data, load_data
from gbkfit.dataset.regions import Regions, regions_parser
from gbkfit.utils import fitsutils, parseutils
from . import _detail


__all__ = [
    'DatasetASpec'
]


class DatasetASpec(Dataset):
    """
    Spectra in regions of the sky (see Regions; e.g. fibres, apertures,
    bins): one data item, 'aspec', of shape (nchannels, nregions) (FITS:
    the regions along x, the velocity along y), with the world coordinates
    of the spectral axis.
    """

    _ndim = 2

    @staticmethod
    def type():
        return 'aspec'

    @classmethod
    def load(cls, info, prefix=''):
        """
        The options of its one data item are given flat. The world
        coordinates of the spectral axis are those of the header of the
        data file, or of the options step, rpix, rval and rest.
        """
        desc = parseutils.make_typed_desc(cls, 'dataset')
        parseutils.load_option_and_update_info(
            regions_parser, info, 'regions', True, False, prefix=prefix)
        step, rpix, rval, rest = (info.pop(key, None) for key in (
            'step', 'rpix', 'rval', 'rest'))
        item = {k: info.pop(k) for k in ('data', 'mask', 'error') if k in info}
        aspec, coords = load_data(item, prefix, rpix, rval, rest)
        info.update(
            aspec=aspec,
            step=coords.step[1] if step is None else step,
            rpix=coords.rpix[1],
            rval=coords.rval[1],
            rest=coords.rest)
        return cls(**parseutils.parse_options_for_callable(
            info, desc, cls.__init__))

    def dump(self, prefix='', dump_path=True, overwrite=False):
        spectral = self._spectral_grid.coords

        def write(filename, array):
            fitsutils.write_spectra(filename, array, spectral, overwrite)
        item = dump_data(
            self['aspec'], _detail.item_filenames(prefix, 'aspec'), write,
            dump_path)
        return dict(
            type=self.type(),
            regions=regions_parser.dump(
                self._regions, prefix=prefix, dump_path=dump_path,
                overwrite=overwrite),
            step=spectral.step[0],
            rpix=spectral.rpix[0],
            rval=spectral.rval[0],
            rest=None if spectral.rest is None else str(spectral.rest)) | item

    def __init__(
            self,
            aspec: Data,
            regions: Regions,
            step: Real = 1,
            rpix: Real | None = None,
            rval: Real = 0,
            rest: str | astropy.units.Quantity | None = None
    ):
        """
        step, rpix, rval and rest are the world coordinates of the
        spectral axis (see fitsutils.Coords): the channel width (km/s), the
        reference channel (by default the centre), its velocity (km/s), and
        the rest wavelength or frequency of the velocities.
        """
        super().__init__(dict(aspec=aspec))
        nchannels, nregions = aspec.shape()
        if nregions != regions.nregions():
            raise RuntimeError(
                f"the spectra are of {nregions} regions, but there are "
                f"{regions.nregions()} regions")
        self._regions = regions
        self._spectral_grid = fitsutils.make_grid(
            (nchannels,), step, rpix, rval, 0, spectral_axis=0, rest=rest)

    def regions(self) -> Regions:
        return self._regions

    def spectral_grid(self) -> fitsutils.Grid:
        """The grid of the spectral axis (one axis)."""
        return self._spectral_grid
