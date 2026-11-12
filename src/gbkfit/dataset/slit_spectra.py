"""
The dataset of spectra along a long slit.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from gbkfit.config import DataFile, check_options
from gbkfit.grid import SpectralAxis, x_axis_pa, x_axis_rota
from gbkfit.instrument import LSF
from gbkfit.io import ErrorKind, MaskKind
from . import _grids
from .base import DatasetGrid, NoiseCorrelation, cleaned
from .data import Data


__all__ = [
    'DatasetSlitSpectra'
]


class DatasetSlitSpectra(DatasetGrid):
    """
    Spectra along a long slit.

    Its configuration has the options of from_files: the files of its one
    data item (data, mask, error) and its world coordinates.

    The slit is oriented on the sky by its position angle, slit_pa: that
    of its +x axis, the way the positions along it increase, as the PA of
    a slit is quoted (the rotation of its grid, the position angle of +y,
    across the slit, is 90 degrees more; see x_axis_rota).

    Its noise can be correlated along the spectral axis (e.g. by the
    resampling of the spectra), not along the slit (see
    noise_correlation).

    Parameters
    ----------
    spectra : Data
        The spectra, of shape (nz, nx): the position along the slit and
        the spectral axis.
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the axes on the sky (the position along
        the slit; see Coords); their defaults are those of
        make_grid.
    slit_pa : float, optional
        The position angle of the slit (degrees, north through east), of
        its +x axis; -90 by default (+x to the west, as the x axis of an
        image of north up and east to the left).
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
            slit_pa: float | None = None,
            spectral: SpectralAxis | None = None,
            spectral_correlation: LSF | None = None,
            outliers: Mapping[str, Any] | None = None,
            mask_whole_spectra: bool = False,
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
        slit_pa : float, optional
            The position angle of the slit (see DatasetSlitSpectra); by
            default, that of the x axis of the header (of files with an
            axis along the slit and none across it, -90: their world
            coordinates do not orient the slit on the sky).
        spectral : SpectralAxis, optional
            The world coordinates of the spectral axis, the convention of
            its velocities, its rest and its medium, given instead of
            those of the header (as those of the axes on the sky; the
            medium of the header, see coords_from_header); not its size,
            which the data give.
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
            unit, beam, spectral_correlation=spectral_correlation,
            velocity=references['velocity'], error_kind=error_kind,
            mask_kind=mask_kind, footprint=footprint,
            spectral_type=references['spectral_type'])
        options = _grids.cube_coords(coords, step, None, spectral)
        # (the slit along the x axis of the files, unless its angle is given)
        rota = options.pop('rota')
        dataset = cls(
            item, **options,
            slit_pa=x_axis_pa(rota) if slit_pa is None else slit_pa)
        return cleaned(dataset, outliers, mask_whole_spectra)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        options = _grids.dump_grid_dataset(
            self, prefix, dump_path, overwrite)
        # (oriented by its slit, in place of the rotation of its grid)
        options = dict(
            ('slit_pa', self._slit_pa) if key == 'rota' else (key, value)
            for key, value in options.items())
        return _grids.flatten_single_item(options, 'spectra')

    def __init__(
            self,
            spectra: Data,
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            slit_pa: float | None = None,
            spectral: SpectralAxis | None = None
    ):
        check_options(DatasetSlitSpectra.__init__, locals())
        super().__init__(
            dict(spectra=spectra), step, rpix, rval,
            None if slit_pa is None else x_axis_rota(slit_pa), spectral)
        self._slit_pa = x_axis_pa(self.grid().coords.rota) \
            if slit_pa is None else float(slit_pa)

    def slit_pa(self) -> float:
        """Return the position angle of the slit (see the class)."""
        return self._slit_pa

    def mask_whole_spectra(self) -> 'DatasetSlitSpectra':
        """
        Return a copy of it with the spectrum of each position along the
        slit masked whole where it masks any of its channels, but for the
        channels it masks at every position (see
        Dataset.mask_whole_spectra).
        """
        return self.with_data(dict(
            spectra=_grids.whole_spectra(self['spectra'])))

    def noise_correlation(self, key: str) -> NoiseCorrelation | None:
        """
        Return the correlation of the noise of a data item on the
        separations of its channels (see Dataset.noise_correlation): its
        positions along the slit, which have no pixels across it, have
        none.
        """
        return _grids.spectra_noise_correlation(self, key, self.grid())
