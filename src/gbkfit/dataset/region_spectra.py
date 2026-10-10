from typing import Any

import astropy.units
import numpy as np

from gbkfit.region import Regions, regions_parser
from gbkfit.utils import fitsutils, gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _detail
from .base import Dataset
from .data import Data, dump_data, load_data


__all__ = [
    'DatasetRegionSpectra'
]


class DatasetRegionSpectra(Dataset):
    """
    Spectra in regions of the sky (see Regions; e.g. fibres, apertures,
    bins), with the world coordinates of their spectral axis.

    Its configuration has the options of its one data item beside its own
    (e.g. data, error and step). The world coordinates of the spectral
    axis come from the header of the data file unless given.

    Parameters
    ----------
    spectra : Data
        The spectra, of shape (nchannels, nregions) (in FITS, the regions
        along x and the velocity along y).
    regions : Regions
        The regions.
    step : float, optional
        The width of the channels (km/s).
    rpix : float, optional
        The reference channel; by default, the centre.
    rval : float, optional
        The velocity of the reference channel (km/s).
    rest : str or Quantity, optional
        The rest of the spectral axis (see gridutils.make_rest).

    Raises
    ------
    ConfigError
        If the spectra are not of one region each.
    """

    ndim = 2

    @staticmethod
    def type() -> str:
        return 'region_spectra'

    @classmethod
    def load(
            cls, info: dict[str, Any], prefix: str = ''
    ) -> 'DatasetRegionSpectra':
        parseutils.load_option_and_update_info(
            regions_parser, info, 'regions', required=True, prefix=prefix)
        step, rpix, rval, rest = (info.pop(key, None) for key in (
            'step', 'rpix', 'rval', 'rest'))
        item = {k: info.pop(k) for k in ('data', 'mask', 'error') if k in info}
        # (the velocity is along FITS y, the axis 1)
        spectra, coords = load_data(item, prefix, rpix, rval, rest, 1)
        info.update(
            spectra=spectra,
            step=coords.step[1] if step is None else step,
            rpix=coords.rpix[1],
            rval=coords.rval[1],
            rest=coords.rest)
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        spectral = self._spectral_grid.coords

        def write(filename: str, array: np.ndarray) -> None:
            fitsutils.write_spectra(filename, array, spectral, overwrite)
        item = dump_data(
            self['spectra'], _detail.item_filenames(prefix, 'spectra'), write,
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
            spectra: Data,
            regions: Regions,
            step: float = 1,
            rpix: float | None = None,
            rval: float = 0,
            rest: str | astropy.units.Quantity | None = None
    ):
        super().__init__(dict(spectra=spectra))
        nchannels, nregions = spectra.shape()
        if nregions != regions.nregions():
            raise ConfigError(
                f"the spectra are of {nregions} regions, but there are "
                f"{regions.nregions()} regions")
        self._regions = regions
        self._spectral_grid = gridutils.make_grid(
            (nchannels,), step, rpix, rval, 0, spectral_axis=0, rest=rest)

    def regions(self) -> Regions:
        """Return the regions."""
        return self._regions

    def spectral_grid(self) -> gridutils.Grid:
        """Return the grid of the spectral axis (one axis)."""
        return self._spectral_grid
