from typing import Any

import astropy.units
import numpy as np

from gbkfit.region import Regions, regions_parser
from gbkfit.utils import fitsutils, gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _detail
from .base import Dataset
from .data import Data, FitsFile, dump_data


__all__ = [
    'DatasetRegionSpectra'
]


class DatasetRegionSpectra(Dataset):
    """
    Spectra in regions of the sky (see Regions; e.g. fibres, apertures,
    bins), with the world coordinates of their spectral axis.

    Its configuration has the regions, and the options of from_files: the
    files of its one data item (data, mask, error) and the world
    coordinates of its spectral axis.

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
        info = dict(info)
        parseutils.load_option_and_update_info(
            regions_parser, info, 'regions', required=True, prefix=prefix)
        files = _detail.pop_item_files(info, prefix)
        return _detail.load_with_files(cls, info, **files)

    @classmethod
    def from_files(
            cls,
            data: FitsFile,
            regions: Regions,
            mask: FitsFile | None = None,
            error: FitsFile | float | None = None,
            step: float | None = None,
            rpix: float | None = None,
            rval: float | None = None,
            rest: str | astropy.units.Quantity | None = None
    ) -> 'DatasetRegionSpectra':
        """
        Read the dataset from FITS files (the regions along x, the velocity
        along y), with the world coordinates of the spectral axis of the
        file of its values unless given.

        Parameters
        ----------
        data : str or tuple
            The file of the spectra: a filename, or a filename and the HDU.
        regions : Regions
            The regions.
        mask : str or tuple, optional
            The file of the mask.
        error : str or tuple or float, optional
            The file of the errors, or one error for all the values.
        step, rpix, rval : float, optional
            The world coordinates of the spectral axis (see the class); by
            default, those of the header. Either rpix or rval can be given,
            and the other comes from the header (see fitsutils.read_data).
        rest : str or Quantity, optional
            The rest of the spectral axis (see gridutils.make_rest); by
            default, that of the header.

        Returns
        -------
        DatasetRegionSpectra
            The dataset.
        """
        # (the velocity is along FITS y, the axis 1)
        spectra, coords = Data.from_files(
            data, mask, error, rpix, rval, rest, 1)
        return cls(
            spectra, regions,
            step=coords.step[1] if step is None else step,
            rpix=coords.rpix[1], rval=coords.rval[1], rest=coords.rest)

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
