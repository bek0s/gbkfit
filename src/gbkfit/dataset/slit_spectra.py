"""
The dataset of spectra along a long slit.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from gbkfit.config import DataFile, check_options
from gbkfit.grid import SpectralAxis
from gbkfit.io import ErrorKind, MaskKind
from . import _grids
from .base import DatasetGrid, cleaned
from .data import Data


__all__ = [
    'DatasetSlitSpectra'
]


class DatasetSlitSpectra(DatasetGrid):
    """
    Spectra along a long slit.

    Its configuration has the options of from_files: the files of its one
    data item (data, mask, error) and its world coordinates.

    Parameters
    ----------
    spectra : Data
        The spectra, of shape (nz, nx): the position along the slit and
        the spectral axis.
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the axes on the sky (the position along
        the slit; see Coords); their defaults are those of
        make_grid.
    rota : float, optional
        The rotation of the grid on the sky (see Coords).
    spectral : SpectralAxis, optional
        The spectral axis (see make_cube_grid): its world
        coordinates, the convention of its velocities, its rest and its
        medium; the data give its size.
    """

    # The axes of a long-slit spectrum: the position along the slit and
    # the spectral axis
    ndim = 2
    spectral_axis = 1

    # The axes of its blocks and windows (see resample_blocks and
    # screen_outliers): of positions along the slit and channels
    block_axes = ('the position along the slit', 'the channels')

    config_constructors = ('from_files',)

    @staticmethod
    def type() -> str:
        return 'slit_spectra'

    @staticmethod
    def type_title() -> str:
        return 'Slit spectra (long slit)'

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
            outliers: Mapping[str, Any] | None = None,
            error_kind: ErrorKind = 'sigma',
            mask_kind: MaskKind = 'valid',
            footprint: DataFile | None = None
    ) -> 'DatasetSlitSpectra':
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
        DatasetSlitSpectra
            The dataset.
        """
        check_options(DatasetSlitSpectra.from_files, locals())
        references = _grids.file_references(1, rpix, rval, spectral)
        item, coords = Data.from_files(
            data, mask, error, references['rpix'], references['rval'],
            references['rest'], cls.spectral_axis, references['medium'],
            unit, beam, velocity=references['velocity'],
            error_kind=error_kind, mask_kind=mask_kind, footprint=footprint,
            spectral_type=references['spectral_type'])
        dataset = cls(
            item, **_grids.cube_coords(coords, step, rota, spectral))
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
            rota: float | None = None,
            spectral: SpectralAxis | None = None
    ):
        check_options(DatasetSlitSpectra.__init__, locals())
        super().__init__(
            dict(spectra=spectra), step, rpix, rval, rota, spectral)
