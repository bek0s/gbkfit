"""
The observation of spectra along a long slit.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias

import numpy as np
from annotated_types import Gt

from gbkfit.config import Unit, check_options
from gbkfit.dataset import Data, Dataset, DatasetSlitSpectra
from gbkfit.driver import Driver
from gbkfit.grid import (
    Grid, GridData, SpectralAxis, grid_options, x_axis_rota)
from gbkfit.instrument import Instrument
from gbkfit.model import Band, DataUnit, LightUnit, LineEmission
from ..depth import Depth
from ..foreground import Foreground
from ..plan import FluxScale, ModelData, ModelItem
from ..weights import Weights
from . import _model_cube
from ._cube_plan import CubePlan
from ._data_units import data_unit
from ._options import (
    Rota, Rpix, Rval, Size, Step, warn_unsmoothed_weights)
from .base import DType, ObservationSky, SkyOptions

if TYPE_CHECKING:
    from gbkfit.model import Scene
    from ..noise import Noise


__all__ = [
    'ObservationSlitSpectra'
]


# The width of a slit
_Width: TypeAlias = Annotated[float, Gt(0), Unit('arcsec')]


def _slit_extra(data: np.ndarray, grid: Grid) -> GridData:
    """
    Make an extra output on the low-resolution grid of ModelCube a slit
    (the position along the slit and the velocity; the slit is one pixel
    wide).
    """
    return GridData(data[:, 0, :], grid.coords.axes(0, 2), 1)


def _slit_grid_options(grid: Grid, sizes: bool = True) -> dict[str, Any]:
    """
    Return the options of the grid of a slit (see grid_options) but its
    rotation, which the position angle of the slit gives (see
    x_axis_rota).
    """
    options = grid_options(grid, sizes)
    del options['rota']
    return options


class ObservationSlitSpectra(ObservationSky):
    """
    Spectra along a long slit: the model, seen through the instrument,
    averaged across the slit at each position along it.

    The slit is oriented on the sky by its position angle, slit_pa, as
    that of its data (see DatasetSlitSpectra): the position angle of its
    +x axis, the way the positions along it increase.

    Its options are those below, then those of every observation on the
    sky (see ObservationSky). Its data give those of its grid and
    slit_pa, which are then optional; without data, size and spectral are
    required.

    Parameters
    ----------
    size : int or Sequence of int, optional
        The number of positions along the slit (one value).
    spectral : SpectralAxis, optional
        The spectral axis: its size, its world coordinates and the
        convention of its velocities (see make_cube_grid).
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the positions along the slit (see
        Coords; one value each); without data, a step of 1, rval 0, and
        rpix the centre by default.
    slit_pa : float, optional
        The position angle of the slit (degrees, north through east), of
        its +x axis; without data, -90 by default (+x to the west, as the
        x axis of an image of north up and east to the left). The rotation
        of its grid (see Coords), the position angle of +y, across the
        slit, is 90 degrees more (see x_axis_rota).
    slit_width : float, optional
        The width of the slit; by default, the step along it.
    sky_rval : Sequence of float, optional
        The RA and Dec (degrees) of the reference point of the slit (rpix
        along it), the origin of its frame (see ObservationSky.sky_rval);
        (0, 0) by default, as the grids of data without coordinates.
    smooth_weights : bool, optional
        Whether the weights of the model are smoothed by the PSF and the
        LSF, as the model is.

    Raises
    ------
    ValueError
        If the spectral axis is not valid (see make_cube_grid),
        slit_width is not positive, sky_rval is not two numbers, or an
        option of every observation on the sky is not valid (see
        ObservationSky).
    """

    dataset_class = DatasetSlitSpectra

    emission = LineEmission
    orbit_stars = None

    # The axes of a long-slit spectrum: the position along the slit
    # and the spectral axis
    spectral_axis = 1

    @staticmethod
    def type() -> str:
        return 'slit_spectra'

    @staticmethod
    def type_title() -> str:
        return 'Slit spectra (long slit)'

    config_required_without = dict(dataset=('size', 'spectral'))
    defaults_without_data = dict(step=(1,), rval=(0,), slit_pa=-90)

    @classmethod
    def _options_from_data(
            cls, dataset: DatasetSlitSpectra, options: dict[str, Any]
    ) -> dict[str, Any]:
        # (oriented by the slit of the data, not the rotation of its grid)
        return _slit_grid_options(dataset.grid()) \
            | dict(slit_pa=dataset.slit_pa())

    @classmethod
    def options_from_data(cls, dataset: Dataset) -> tuple[str, ...]:
        # (the orientation of the slit, in place of the rotation of the
        # grid)
        return tuple(
            'slit_pa' if option == 'rota' else option
            for option in super().options_from_data(dataset))

    def _dump_grid(self) -> dict[str, Any]:
        # (the orientation of the slit, in place of the rotation of the
        # grid, at its place)
        return dict(
            ('slit_pa', self._slit_pa) if key == 'rota' else (key, value)
            for key, value in super()._dump_grid().items())

    def _dump_options(
            self, prefix: str, dump_path: bool, overwrite: bool
    ) -> dict[str, Any]:
        return dict(
            slit_width=self._slit_width,
            sky_rval=self._sky_rval,
            smooth_weights=self._smooth_weights)

    def _kind_options(self) -> dict[str, Any]:
        return _slit_grid_options(self.grid()) | dict(
            slit_pa=self._slit_pa, slit_width=self._slit_width,
            sky_rval=self._sky_rval, smooth_weights=self._smooth_weights)

    def __init__(
            self,
            size: Size | None = None,
            spectral: SpectralAxis | None = None,
            step: Step | None = None,
            rpix: Rpix | None = None,
            rval: Rval | None = None,
            slit_pa: Rota | None = None,
            slit_width: _Width | None = None,
            sky_rval: Rval = (0, 0),
            smooth_weights: bool = False,
            foreground: Foreground | None = None,
            instrument: Instrument | None = None,
            galaxies: Sequence[str] | None = None,
            components: Sequence[str] | None = None,
            lines: Sequence[str] | None = None,
            band: str | Band | None = None,
            pixel_weights: Weights | None = None,
            flux_scale: str | None = None,
            offsets: Mapping[str, bool] | None = None,
            dataset: Dataset | None = None,
            oversampling: Sequence[int] | None = None,
            depth: Depth | None = None,
            dtype: DType | None = None,
            fft_dtype: DType = 'float64',
            name: str | None = None,
            unit: str | None = None
    ):
        check_options(ObservationSlitSpectra.__init__, locals())
        options = self._options(dataset, dict(
            size=size, spectral=spectral, step=step, rpix=rpix, rval=rval,
            slit_pa=slit_pa))
        self._slit_pa = float(options['slit_pa'])
        grid = self._make_grid(
            options['size'], options['step'], options['rpix'],
            options['rval'], x_axis_rota(self._slit_pa),
            options['spectral'])
        if slit_width is None:
            slit_width = grid.coords.step[0]
        sky_rval = tuple(sky_rval)
        if len(sky_rval) != 2:
            raise ValueError(
                f"sky_rval is the RA and Dec of the reference point of the "
                f"slit; it is {sky_rval}")
        self._sky_rval = float(sky_rval[0]), float(sky_rval[1])
        self._slit_width = slit_width
        self._smooth_weights = smooth_weights
        super().__init__(
            grid, foreground, instrument, galaxies, components, lines, band,
            pixel_weights, flux_scale, offsets, dataset, oversampling, depth,
            dtype, fft_dtype, name, unit)

    def slit_pa(self) -> float:
        """Return the position angle of the slit (see the class)."""
        return self._slit_pa

    def slit_width(self) -> float:
        """Return the width of the slit."""
        return self._slit_width

    def sky_rval(self) -> tuple[float, float]:
        return self._sky_rval

    def keys(self) -> tuple[str, ...]:
        return ('spectra',)

    def make_dataset(self, items: Mapping[str, Data]) -> DatasetSlitSpectra:
        return DatasetSlitSpectra(
            items['spectra'], **_slit_grid_options(self.grid(), sizes=False),
            slit_pa=self._slit_pa)

    def _data_unit(
            self,
            dataset: Dataset | None,
            light: LightUnit | None
    ) -> DataUnit | None:
        # (a pixel spans the step along the slit and the width of the slit)
        return data_unit(
            dataset, 'spectra', abs(self.step()[0]) * self._slit_width,
            False, light)

    def light_items(self) -> tuple[str, ...]:
        return ('spectra',)

    def flux_scales(self) -> dict[str, FluxScale]:
        # (each spectrum runs along the first axis, at each position along
        # the slit)
        return super().flux_scales() \
            | dict(per_spectrum=FluxScale(('spectra',), 0, self.size()[0]))

    def linear_in_light(self) -> bool:
        # (its spectra of the light, through its instrument: linear)
        return True

    def model_can_have_missing_values(self) -> bool:
        return False

    def _check_options(self, options: SkyOptions) -> None:
        warn_unsmoothed_weights(self._smooth_weights, options.instrument)

    def _plan(self, driver: Driver, scene: 'Scene') -> 'SlitSpectraPlan':
        options = self._sky_options
        # The cube of a slit is one pixel (the slit width) across the slit,
        # sampled as finely as the pixels along it (the model is the mean
        # over the slit)
        size, coords = self._grid.size, self._grid.coords
        step, rpix, rval = coords.step, coords.rpix, coords.rval
        oversampling = options.oversampling
        across = oversampling[0] * max(1, round(self._slit_width / step[0]))
        model_cube = _model_cube.ModelCube(
            Grid((size[0], 1, size[1]), coords._replace(
                step=(step[0], self._slit_width, step[1]),
                rpix=(rpix[0], 0, rpix[1]), rval=(rval[0], 0, rval[1])), 2),
            oversampling=(oversampling[0], across, oversampling[1]),
            primary_beam=options.instrument.primary_beam(),
            psf=options.instrument.psf(),
            lsf=options.instrument.lsf(),
            smooth_weights=self._smooth_weights,
            dtype=options.dtype,
            fft_dtype=options.fft_dtype)
        return SlitSpectraPlan(model_cube, driver, scene, options)


class SlitSpectraPlan(CubePlan):

    def _model_extra(self, value: GridData) -> np.ndarray:
        # The slit has no position on the sky, so the extra outputs of the
        # model, which are on the sky, are plain arrays
        return value.data

    def has_exact_errors(self) -> bool:
        return True

    def _exact_errors(
            self, noise: 'Noise', recorded: np.ndarray
    ) -> dict[str, np.ndarray]:
        return dict(
            spectra=noise.sigma(self._model_cube.grid(), recorded)[:, 0, :])

    # The slit has no position on the sky, so the extra outputs of the
    # high-resolution grid are plain arrays
    _extra_lo = staticmethod(_slit_extra)
    _extra_high_res = staticmethod(_model_cube.plain_extra)

    def _measure(self) -> ModelData:
        plan = self._model_cube_plan
        weights = plan.weights()
        return dict(spectra=ModelItem(
            plan.cube()[:, 0, :],
            weights=None if weights is None else weights[:, 0, :]))
