from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gbkfit.region import Regions, regions_parser
from gbkfit.utils import fitsutils, gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from gbkfit.utils.typeutils import FitsFile
from . import _detail, _outliers, _resampling
from .base import Dataset
from .data import Data, dump_data


__all__ = [
    'DatasetRegionSpectra'
]


# The axes of its blocks of residuals (see resample_blocks)
_BLOCK_AXES = ('the regions', 'the channels')


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
    spectral : SpectralAxis, optional
        The spectral axis (see gridutils.make_cube_grid): its world
        coordinates, the convention of its velocities, its rest and its
        medium; the data give its size.

    Raises
    ------
    ConfigError
        If the spectra are not of one region each.
    """

    ndim = 2

    # (the velocity along FITS y)
    spectral_axis = 1

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
        files = _detail.pop_item_options(info, prefix)
        return _detail.load_with_files(cls, info, **files)

    @classmethod
    def from_files(
            cls,
            data: FitsFile,
            regions: Regions,
            mask: FitsFile | None = None,
            error: FitsFile | float | None = None,
            unit: str | None = None,
            beam: Mapping[str, Any] | None = None,
            spectral: gridutils.SpectralAxis | None = None
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
        unit : str, optional
            The unit of the values; by default, that of the header (see
            Data.from_files).
        beam : Mapping, optional
            The options of their beam (see Beam); by default, that of the
            header.
        spectral : SpectralAxis, optional
            The world coordinates of the spectral axis, the convention of
            its velocities, its rest and its medium, given instead of
            those of the header (either rpix or rval can be given, and the
            other comes from the header; see fitsutils.read_data); not its
            size, which the data give.

        Returns
        -------
        DatasetRegionSpectra
            The dataset.
        """
        references = _detail.file_references(1, None, None, spectral)
        spectra, coords = Data.from_files(
            data, mask, error, references['rpix'], references['rval'],
            references['rest'], cls.spectral_axis, references['medium'],
            unit, beam)
        return cls(
            spectra, regions,
            _detail.cube_coords(coords, None, None, spectral)['spectral'])

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        coords = self._spectral_grid.coords

        def write(filename: str, array: np.ndarray) -> None:
            fitsutils.write_spectra(filename, array, coords, overwrite)
        item = dump_data(
            self['spectra'], _detail.item_filenames(prefix, 'spectra'), write,
            dump_path)
        return dict(
            type=self.type(),
            regions=regions_parser.dump(
                self._regions, prefix=prefix, dump_path=dump_path,
                overwrite=overwrite),
            # (the convention of the velocities is in the header of the file)
            spectral={
                key: value for key, value in gridutils.SpectralAxis.of(
                    self._spectral_grid).dump().items()
                if key not in ('size', 'convention')}) | item

    def __init__(
            self,
            spectra: Data,
            regions: Regions,
            spectral: gridutils.SpectralAxis | None = None
    ):
        super().__init__(dict(spectra=spectra))
        nchannels, nregions = spectra.shape()
        if nregions != regions.nregions():
            raise ConfigError(
                f"the spectra are of {nregions} regions, but there are "
                f"{regions.nregions()} regions")
        self._regions = regions
        spectral = spectral or gridutils.SpectralAxis()
        if spectral.size() not in (None, nchannels):
            raise ConfigError(
                f"the spectra have {nchannels} channels; the spectral axis "
                f"has {spectral.size()}")
        self._spectral_grid = gridutils.make_cube_grid(
            (), None, None, None, None,
            spectral.replaced(gridutils.SpectralAxis(size=nchannels)))

    def regions(self) -> Regions:
        """Return the regions."""
        return self._regions

    def spectral_grid(self) -> gridutils.Grid:
        """Return the grid of the spectral axis (one axis)."""
        return self._spectral_grid

    def check_blocks(self, size: Sequence[float]) -> None:
        """
        Raise ConfigError unless the size is of blocks of regions and channels:
        a positive integer for each of the regions and the channels (see
        resample_blocks).
        """
        _resampling.block_size(size, _BLOCK_AXES)

    def resample_blocks(
            self,
            residuals: Mapping[str, np.ndarray],
            size: Sequence[float],
            rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """
        Return residuals of its spectra redrawn in blocks of regions and
        channels (see Dataset.resample_blocks), of the size (regions,
        channels): runs of consecutive regions, in their order, so of 1 region
        unless their order follows the sky (e.g. fibres in a row). The noise
        that neighbouring regions share (e.g. bins smaller than the beam) is
        then not kept.
        """
        return _resampling.resample_blocks(
            residuals, size, _BLOCK_AXES, rng)

    def screen_outliers(
            self, size: Sequence[float], nsigma: float
    ) -> 'DatasetRegionSpectra':
        """
        Return a copy of it with the outliers of its spectra masked (see
        Dataset.screen_outliers): of the values of the region and of the
        regions nearest it, by their centres (see Regions.centres), in a window
        of channels centred on each, of the size (number of nearest regions,
        channels).
        """
        nnearest, window = _outliers.region_size(
            size, self._regions.nregions(), True)
        nsigma = _outliers.check_nsigma(nsigma)
        spectra = self['spectra'].data()
        return self.with_data(dict(spectra=np.where(
            _outliers.region_outliers(
                spectra, self._regions.centres(), nnearest, window, nsigma),
            np.nan, spectra)))
