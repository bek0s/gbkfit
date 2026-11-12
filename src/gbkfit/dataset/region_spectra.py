"""
The dataset of spectra in regions of the sky (e.g. fibres, apertures,
bins).
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gbkfit.config import DataFile, check_options
from gbkfit.grid import Grid, SpectralAxis, make_cube_grid
from gbkfit.instrument import LSF
from gbkfit.io import ErrorKind, MaskKind, write_spectra
from . import _grids, _outliers
from .base import Dataset, NoiseCorrelation, cleaned
from .data import Data, dump_data, item_filenames
from .regions import Regions, regions_parser


__all__ = [
    'DatasetRegionSpectra'
]


class DatasetRegionSpectra(Dataset):
    """
    Spectra in regions of the sky (e.g. fibres, apertures, bins), with
    the world coordinates of their spectral axis. The regions are those of
    Regions.

    Its configuration has the regions, and the options of from_files: the
    files of its one data item (data, mask, error) and the world
    coordinates of its spectral axis.

    Its noise can be correlated along the spectral axis (e.g. by the
    resampling of the spectra), not between regions (see
    noise_correlation).

    Parameters
    ----------
    spectra : Data
        The spectra, of shape (nchannels, nregions) (in FITS, the regions
        along x and the velocity along y).
    regions : Regions
        The regions.
    spectral : SpectralAxis, optional
        The spectral axis (see make_cube_grid): its world
        coordinates, the convention of its velocities, its rest and its
        medium; the data give its size.

    Raises
    ------
    ValueError
        If the spectra are not of one region each.
    """

    ndim = 2

    # (the velocity along FITS y)
    spectral_axis = 1

    # The axes of its blocks of residuals (see resample_blocks): runs of
    # consecutive regions, in their order, so of 1 region unless their
    # order follows the sky (e.g. fibres in a row), and of channels. The
    # noise that neighbouring regions share (e.g. bins smaller than the
    # beam) is then not kept.
    block_axes = ('the regions', 'the channels')

    config_constructors = ('from_files',)

    @staticmethod
    def type() -> str:
        return 'region_spectra'

    @staticmethod
    def type_title() -> str:
        return 'Region spectra (bins, fibres)'

    @classmethod
    def from_files(
            cls,
            data: DataFile,
            regions: Regions,
            mask: DataFile | None = None,
            error: DataFile | float | None = None,
            unit: str | None = None,
            beam: Mapping[str, Any] | None = None,
            spectral: SpectralAxis | None = None,
            spectral_correlation: LSF | None = None,
            outliers: Mapping[str, Any] | None = None,
            mask_whole_spectra: bool = False,
            error_kind: ErrorKind = 'sigma',
            mask_kind: MaskKind = 'valid'
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
            other comes from the header; see coords_from_header); not its
            size, which the data give.
        spectral_correlation : LSF, optional
            The correlation of the noise along the spectral axis (see
            Data).
        outliers : Mapping, optional
            Screen the data for outliers by the options size and nsigma
            (see screen_outliers), which are masked.
        mask_whole_spectra : bool, optional
            Whether every spectrum with a masked value is masked whole,
            but for the channels masked in every spectrum (see
            mask_whole_spectra), after the outliers.
        error_kind : str, optional
            What the file of the errors holds: 'sigma' (standard
            deviations), 'variance' or 'inverse_variance' (see
            read_arrays).
        mask_kind : str, optional
            What the file of the mask marks with values other than 0:
            'valid' (by default) or 'invalid' values (see read_arrays).

        Returns
        -------
        DatasetRegionSpectra
            The dataset.
        """
        check_options(DatasetRegionSpectra.from_files, locals())
        references = _grids.file_references(1, None, None, spectral)
        spectra, coords = Data.from_files(
            data, mask, error, references['rpix'], references['rval'],
            references['rest'], cls.spectral_axis, references['medium'],
            unit, beam, spectral_correlation=spectral_correlation,
            velocity=references['velocity'], error_kind=error_kind,
            mask_kind=mask_kind, spectral_type=references['spectral_type'])
        dataset = cls(
            spectra, regions,
            _grids.cube_coords(coords, None, None, spectral)['spectral'])
        return cleaned(dataset, outliers, mask_whole_spectra)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        coords = self._spectral_grid.coords

        def write(filename: str, array: np.ndarray) -> None:
            write_spectra(filename, array, coords, overwrite=overwrite)
        item = dump_data(
            self['spectra'], item_filenames(prefix, 'spectra'), write,
            dump_path, f'{prefix}spectra_', overwrite)
        return dict(
            type=self.type(),
            regions=regions_parser.dump(
                self._regions, prefix=prefix, dump_path=dump_path,
                overwrite=overwrite),
            # (the convention of the velocities is in the header of the file)
            spectral={
                key: value for key, value in SpectralAxis.of(
                    self._spectral_grid).dump().items()
                if key not in ('size', 'convention')}) | item

    def __init__(
            self,
            spectra: Data,
            regions: Regions,
            spectral: SpectralAxis | None = None
    ):
        check_options(DatasetRegionSpectra.__init__, locals())
        super().__init__(dict(spectra=spectra))
        nchannels, nregions = spectra.shape()
        if nregions != regions.nregions():
            raise ValueError(
                f"the spectra are of {nregions} regions, but there are "
                f"{regions.nregions()} regions")
        self._regions = regions
        spectral = spectral or SpectralAxis()
        if spectral.size() not in (None, nchannels):
            raise ValueError(
                f"the spectra have {nchannels} channels; the spectral axis "
                f"has {spectral.size()}")
        self._spectral_grid = make_cube_grid(
            (), None, None, None, None,
            spectral.replaced(SpectralAxis(size=nchannels)))

    def regions(self) -> Regions:
        """Return the regions."""
        return self._regions

    def spectral_grid(self) -> Grid:
        """Return the grid of the spectral axis (one axis)."""
        return self._spectral_grid

    def mask_whole_spectra(self) -> 'DatasetRegionSpectra':
        """
        Return a copy of it with the spectrum of each region masked whole
        where it masks any of its channels, but for the channels it masks
        in every region (see Dataset.mask_whole_spectra).
        """
        return self.with_data(dict(
            spectra=_grids.whole_spectra(self['spectra'])))

    def noise_correlation(self, key: str) -> NoiseCorrelation | None:
        """
        Return the correlation of the noise of a data item on the
        separations of its channels (see Dataset.noise_correlation): its
        regions, which are not pixels of the sky, have none.
        """
        return _grids.spectra_noise_correlation(
            self, key, self._spectral_grid)

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
