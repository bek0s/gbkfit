from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import check_options
from gbkfit.utils.typeutils import FitsFile
from . import _detail, _outliers, _resampling
from .base import Dataset
from .data import Data


__all__ = [
    'DatasetSlitSpectra'
]


# The axes of its blocks and windows (see resample_blocks and
# screen_outliers)
_AXES = ('the position along the slit', 'the channels')


class DatasetSlitSpectra(Dataset):
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
        the slit; see gridutils.Coords); their defaults are those of
        gridutils.make_grid.
    rota : float, optional
        The rotation of the grid on the sky (see gridutils.Coords).
    spectral : SpectralAxis, optional
        The spectral axis (see gridutils.make_cube_grid): its world
        coordinates, the convention of its velocities, its rest and its
        medium; the data give its size.
    """

    # The axes of a long-slit spectrum: the position along the slit and
    # the spectral axis
    ndim = 2
    spectral_axis = 1

    config_constructors = ('from_files',)

    @staticmethod
    def type() -> str:
        return 'slit_spectra'

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'DatasetSlitSpectra':
        return _detail.load_with_files(cls, dict(info), cls.ndim - 1)

    @classmethod
    def from_files(
            cls,
            data: FitsFile,
            mask: FitsFile | None = None,
            error: FitsFile | float | None = None,
            unit: str | None = None,
            beam: Mapping[str, Any] | None = None,
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None,
            spectral: gridutils.SpectralAxis | None = None,
            outliers: Mapping[str, Any] | None = None
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
            gridutils.Coords); by default, those of the header. Either rpix
            or rval can be given, and the other comes from the header (see
            fitsutils.read_data).
        rota : float, optional
            The rotation of the grid on the sky; by default, that of the
            header.
        spectral : SpectralAxis, optional
            The world coordinates of the spectral axis, the convention of
            its velocities, its rest and its medium, given instead of
            those of the header (as those of the axes on the sky; the
            medium of the header, see fitsutils.read_data); not its size,
            which the data give.
        outliers : Mapping, optional
            Screen the data for outliers by the options size and nsigma
            (see screen_outliers), which are masked.

        Returns
        -------
        DatasetSlitSpectra
            The dataset.
        """
        check_options(DatasetSlitSpectra.from_files, locals())
        references = _detail.file_references(1, rpix, rval, spectral)
        item, coords = Data.from_files(
            data, mask, error, references['rpix'], references['rval'],
            references['rest'], cls.spectral_axis, references['medium'],
            unit, beam)
        dataset = cls(
            item, **_detail.cube_coords(coords, step, rota, spectral))
        return _detail.cleaned(dataset, outliers)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return _detail.flatten_single_item(
            _detail.dump_grid_dataset(self, prefix, dump_path, overwrite),
            'spectra')

    def __init__(
            self,
            spectra: Data,
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None,
            spectral: gridutils.SpectralAxis | None = None
    ):
        check_options(DatasetSlitSpectra.__init__, locals())
        super().__init__(dict(spectra=spectra))
        self._grid = _detail.cube_grid(
            self, step, rpix, rval, rota, spectral)

    def grid(self) -> gridutils.Grid:
        """Return the grid of the positions and channels."""
        return self._grid

    def check_blocks(self, size: Sequence[float]) -> None:
        """
        Raise ConfigError unless the size is of blocks of positions along the
        slit and channels: a positive integer for each of the position along
        the slit and the channels (see resample_blocks).
        """
        _resampling.block_size(size, _AXES)

    def resample_blocks(
            self,
            residuals: Mapping[str, np.ndarray],
            size: Sequence[float],
            rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """
        Return residuals of its spectra redrawn in blocks of positions along
        the slit and channels (see Dataset.resample_blocks), of the size
        (position along the slit, channels).
        """
        return _resampling.resample_blocks(
            residuals, size, _AXES, rng)

    def screen_outliers(
            self, size: Sequence[float], nsigma: float
    ) -> 'DatasetSlitSpectra':
        """
        Return a copy of it with the outliers of its spectra masked (see
        Dataset.screen_outliers): of the other values of a window of the size
        (position along the slit, channels) about each.
        """
        return _outliers.screen_grid(self, size, _AXES, nsigma)
