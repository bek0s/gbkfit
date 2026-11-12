"""
The dataset of spectra on a grid of pixels: a spectral cube.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from gbkfit.config import DataFile, check_options
from gbkfit.grid import SpectralAxis
from gbkfit.instrument import LSF, PSF
from gbkfit.io import ErrorKind, MaskKind
from . import _grids
from .base import DatasetGrid, NoiseCorrelation, cleaned
from .data import Data


__all__ = [
    'DatasetPixelSpectra'
]


class DatasetPixelSpectra(DatasetGrid):
    """
    Spectra on a grid of pixels: a spectral cube.

    Its configuration has the options of from_files: the files of its one
    data item (data, mask, error) and its world coordinates.

    Parameters
    ----------
    spectra : Data
        The cube, of shape (nz, ny, nx): x, y and the spectral axis.
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the axes on the sky (x and y; see
        Coords); their defaults are those of make_grid.
    rota : float, optional
        The rotation of the grid on the sky (see Coords).
    spectral : SpectralAxis, optional
        The spectral axis (see make_cube_grid): its world
        coordinates, the convention of its velocities, its rest and its
        medium; the data give its size.
    """

    # The axes of a spectral cube: x, y and the spectral axis
    ndim = 3
    spectral_axis = 2

    # The axes of its blocks and windows (see resample_blocks and
    # screen_outliers): of pixels and channels
    block_axes = ('x', 'y', 'the channels')

    config_constructors = ('from_files',)

    @staticmethod
    def type() -> str:
        return 'pixel_spectra'

    @staticmethod
    def type_title() -> str:
        return 'Pixel spectra (cube)'

    @classmethod
    def from_files(
            cls,
            data: DataFile,
            mask: DataFile | None = None,
            error: DataFile | float | None = None,
            unit: str | None = None,
            beam: Mapping[str, Any] | None = None,
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None,
            spectral: SpectralAxis | None = None,
            spatial_correlation: PSF | None = None,
            spectral_correlation: LSF | None = None,
            outliers: Mapping[str, Any] | None = None,
            mask_whole_spectra: bool = False,
            error_kind: ErrorKind = 'sigma',
            mask_kind: MaskKind = 'valid',
            footprint: DataFile | None = None
    ) -> 'DatasetPixelSpectra':
        """
        Read the dataset from FITS files, with the world coordinates of the
        file of its values unless given.

        Parameters
        ----------
        data : str or tuple
            The file of the values: a filename, or a filename and the HDU.
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
        step, rpix, rval : float or Sequence of float, optional
            The world coordinates of the axes on the sky (see
            Coords); by default, those of the header. Either rpix
            or rval can be given, and the other comes from the header (see
            coords_from_header).
        rota : float, optional
            The rotation of the grid on the sky; by default, that of the
            header.
        spectral : SpectralAxis, optional
            The world coordinates of the spectral axis, the convention of
            its velocities, its rest and its medium, given instead of
            those of the header (as those of the axes on the sky; the
            medium of the header, see coords_from_header); not its size,
            which the data give.
        spatial_correlation : PSF, optional
            The correlation of the noise on the sky (see Data).
        spectral_correlation : LSF, optional
            The correlation of the noise along the spectral axis.
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
        footprint : str or tuple, optional
            The file of where the instrument looked (see Data).

        Returns
        -------
        DatasetPixelSpectra
            The dataset.
        """
        check_options(DatasetPixelSpectra.from_files, locals())
        references = _grids.file_references(2, rpix, rval, spectral)
        item, coords = Data.from_files(
            data, mask, error, references['rpix'], references['rval'],
            references['rest'], cls.spectral_axis, references['medium'],
            unit, beam, spatial_correlation, spectral_correlation,
            references['velocity'], error_kind, footprint,
            spectral_type=references['spectral_type'], mask_kind=mask_kind)
        dataset = cls(
            item, **_grids.cube_coords(coords, step, rota, spectral))
        return cleaned(dataset, outliers, mask_whole_spectra)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return _grids.flatten_single_item(
            _grids.dump_grid_dataset(self, prefix, dump_path, overwrite),
            'spectra')

    def __init__(
            self,
            spectra: Data,
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None,
            spectral: SpectralAxis | None = None
    ):
        check_options(DatasetPixelSpectra.__init__, locals())
        super().__init__(
            dict(spectra=spectra), step, rpix, rval, rota, spectral)

    def mask_whole_spectra(self) -> 'DatasetPixelSpectra':
        """
        Return a copy of it with each spectrum of its cube masked whole
        where it masks any of its channels, but for the channels it masks
        in every spectrum (see Dataset.mask_whole_spectra).
        """
        return self.with_data(dict(
            spectra=_grids.whole_spectra(self['spectra'])))

    def noise_correlation(self, key: str) -> NoiseCorrelation | None:
        """
        Return the correlation of the noise of a data item on the
        separations of its pixels and channels (see
        Dataset.noise_correlation).
        """
        return _grids.grid_noise_correlation(self, key)
