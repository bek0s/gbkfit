"""
The dataset of slitless spectra: an image of a grism (or prism)
exposure.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from gbkfit.config import DataFile, check_options
from gbkfit.io import ErrorKind, MaskKind
from . import _grids
from .base import DatasetGrid, cleaned
from .data import Data


__all__ = [
    'DatasetSlitlessSpectra'
]


class DatasetSlitlessSpectra(DatasetGrid):
    """
    Slitless spectra: an image of a grism (or prism) exposure, where the
    light of each wavelength lands away from its place on the sky. The
    trace of the disperser moves it (see SlitlessSpectra).

    Its grid is that of the detector, with the world coordinates of the
    undispersed light: a pixel is at the position on the sky whose light
    would land on it without the disperser.

    Its configuration has the options of from_files: the files of its one
    data item (data, mask, error) and its world coordinates.

    Parameters
    ----------
    spectra : Data
        The image, of shape (ny, nx).
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the grid (see Coords); their
        defaults are those of make_grid.
    rota : float, optional
        The rotation of the grid on the sky (see Coords).
    """

    # An image of dispersed light has no spectral axis: wavelength and
    # position share the axes of the detector
    ndim = 2
    spectral_axis = None

    # The axes of its blocks and windows (see resample_blocks and
    # screen_outliers): of pixels of the detector
    block_axes = ('x', 'y')

    config_constructors = ('from_files',)

    @staticmethod
    def type() -> str:
        return 'slitless_spectra'

    @staticmethod
    def type_title() -> str:
        return 'Slitless spectra (grism)'

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
            outliers: Mapping[str, Any] | None = None,
            error_kind: ErrorKind = 'sigma',
            mask_kind: MaskKind = 'valid',
            footprint: DataFile | None = None
    ) -> 'DatasetSlitlessSpectra':
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
        DatasetSlitlessSpectra
            The dataset.
        """
        check_options(DatasetSlitlessSpectra.from_files, locals())
        item, coords = Data.from_files(
            data, mask, error, rpix, rval, unit=unit, beam=beam,
            error_kind=error_kind, mask_kind=mask_kind, footprint=footprint)
        dataset = cls(item, **_grids.grid_coords(coords, step, rota))
        return cleaned(dataset, outliers)

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
            rota: float | None = None
    ):
        check_options(DatasetSlitlessSpectra.__init__, locals())
        super().__init__(dict(spectra=spectra), step, rpix, rval, rota)
