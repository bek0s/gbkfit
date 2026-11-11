"""
The dataset of brightness on a grid of pixels: an image.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from gbkfit.config import DataFile, check_options
from gbkfit.instrument import PSF
from gbkfit.io import ErrorKind, MaskKind
from . import _grids
from .base import DatasetGrid, NoiseCorrelation, cleaned
from .data import Data


__all__ = [
    'DatasetPixelBrightness'
]


class DatasetPixelBrightness(DatasetGrid):
    """
    Brightness on a grid of pixels: an image.

    Its configuration has the options of from_files: the files of its one
    data item (data, mask, error) and its world coordinates.

    Parameters
    ----------
    brightness : Data
        The image, of shape (ny, nx).
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the grid (see Coords); their
        defaults are those of make_grid.
    rota : float, optional
        The rotation of the grid on the sky (see Coords).
    """

    # An image has no spectral axis
    ndim = 2
    spectral_axis = None

    # The axes of its blocks and windows (see resample_blocks and
    # screen_outliers)
    block_axes = ('x', 'y')

    config_constructors = ('from_files',)

    @staticmethod
    def type() -> str:
        return 'pixel_brightness'

    @staticmethod
    def type_title() -> str:
        return 'Pixel brightness (image)'

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
            spatial_correlation: PSF | None = None,
            outliers: Mapping[str, Any] | None = None,
            error_kind: ErrorKind = 'sigma',
            mask_kind: MaskKind = 'valid',
            footprint: DataFile | None = None
    ) -> 'DatasetPixelBrightness':
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
            The world coordinates of the grid (see Coords); by
            default, those of the header. Either rpix or rval can be given,
            and the other comes from the header (see coords_from_header).
        rota : float, optional
            The rotation of the grid on the sky; by default, that of the
            header.
        spatial_correlation : PSF, optional
            The correlation of the noise on the sky (see Data).
        outliers : Mapping, optional
            Screen the data for outliers by the options size and nsigma
            (see screen_outliers), which are masked.
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
        DatasetPixelBrightness
            The dataset.
        """
        check_options(DatasetPixelBrightness.from_files, locals())
        item, coords = Data.from_files(
            data, mask, error, rpix, rval, unit=unit, beam=beam,
            spatial_correlation=spatial_correlation, error_kind=error_kind,
            mask_kind=mask_kind, footprint=footprint)
        dataset = cls(item, **_grids.grid_coords(coords, step, rota))
        return cleaned(dataset, outliers)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return _grids.flatten_single_item(
            _grids.dump_grid_dataset(self, prefix, dump_path, overwrite),
            'brightness')

    def __init__(
            self,
            brightness: Data,
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None
    ):
        check_options(DatasetPixelBrightness.__init__, locals())
        super().__init__(dict(brightness=brightness), step, rpix, rval, rota)

    def noise_correlation(self, key: str) -> NoiseCorrelation | None:
        """
        Return the correlation of the noise of a data item on the
        separations of its pixels (see Dataset.noise_correlation).
        """
        return _grids.grid_noise_correlation(self, key)
