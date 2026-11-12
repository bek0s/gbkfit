"""
The observable of spectra along a long slit.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias

import numpy as np
from annotated_types import Gt

from gbkfit.config import Unit, check_options
from gbkfit.dataset import Data, Dataset, DatasetSlitSpectra
from gbkfit.driver import Driver
from gbkfit.grid import (
    Grid, GridData, SpectralAxis, grid_options, x_axis_rota)
from gbkfit.model import DataUnit, LightUnit, LineEmission
from . import _model_cube
from ._cube_plan import CubeObservablePlan
from ._data_units import data_unit
from ._options import (
    Rota, Rpix, Rval, Size, Step, warn_unsmoothed_weights)
from .base import (
    FluxScale, ModelData, ModelItem, Observable, ObservationSetup)

if TYPE_CHECKING:
    from gbkfit.model import Scene
    from ..noise import Noise


__all__ = [
    'SlitSpectra'
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


class SlitSpectra(Observable):
    """
    Spectra along a long slit: the model, seen through the instrument,
    averaged across the slit at each position along it.

    The slit is oriented on the sky by its position angle, slit_pa, as
    that of its data (see DatasetSlitSpectra): the position angle of its
    +x axis, the way the positions along it increase.

    Its configuration has the options below, but those of the grid (and
    slit_pa) when it has data, which give them (see from_data).

    Parameters
    ----------
    size : int or Sequence of int
        The number of positions along the slit (one value).
    spectral : SpectralAxis
        The spectral axis: its size, its world coordinates and the
        convention of its velocities (see make_cube_grid).
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the positions along the slit (see
        Coords; one value each); rpix is the centre by default.
    slit_pa : float, optional
        The position angle of the slit (degrees, north through east), of
        its +x axis; -90 by default (+x to the west, as the x axis of an
        image of north up and east to the left). The rotation of its grid
        (see Coords), the position angle of +y, across the slit, is 90
        degrees more (see x_axis_rota).
    slit_width : float, optional
        The width of the slit; by default, the step along it.
    sky_rval : Sequence of float, optional
        The RA and Dec (degrees) of the reference point of the slit (rpix
        along it), the origin of its frame (see Observable.sky_rval);
        (0, 0) by default, as the grids of data without coordinates.
    smooth_weights : bool, optional
        Whether the weights of the model are smoothed by the PSF and the
        LSF, as the model is.

    Raises
    ------
    ValueError
        If the spectral axis is not valid (see make_cube_grid),
        slit_width is not positive, or sky_rval is not two numbers.
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

    @classmethod
    def from_data(
            cls,
            dataset: DatasetSlitSpectra,
            slit_width: _Width | None = None,
            sky_rval: Rval = (0, 0),
            smooth_weights: bool = False
    ) -> 'SlitSpectra':
        """
        Make the observable of the spectra of a slit, on their grid.

        Parameters
        ----------
        dataset : DatasetSlitSpectra
            The spectra.
        slit_width, sky_rval, smooth_weights : optional
            As those of SlitSpectra.

        Returns
        -------
        SlitSpectra
            The observable.

        Raises
        ------
        TypeError
            If the dataset is not of slit spectra.
        ValueError
            If the options are not valid.
        """
        check_options(SlitSpectra.from_data, locals())
        # (oriented by the slit of the data, not the rotation of its grid)
        options = grid_options(dataset.grid())
        del options['rota']
        return cls(
            **options, slit_pa=dataset.slit_pa(), slit_width=slit_width,
            sky_rval=sky_rval, smooth_weights=smooth_weights)

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

    def __init__(
            self,
            size: Size,
            spectral: SpectralAxis,
            step: Step = (1,),
            rpix: Rpix | None = None,
            rval: Rval = (0,),
            slit_pa: Rota = -90,
            slit_width: _Width | None = None,
            sky_rval: Rval = (0, 0),
            smooth_weights: bool = False
    ):
        check_options(SlitSpectra.__init__, locals())
        super().__init__(
            size, step, rpix, rval, x_axis_rota(slit_pa), spectral)
        self._slit_pa = float(slit_pa)
        if slit_width is None:
            slit_width = self.step()[0]
        sky_rval = tuple(sky_rval)
        if len(sky_rval) != 2:
            raise ValueError(
                f"sky_rval is the RA and Dec of the reference point of the "
                f"slit; it is {sky_rval}")
        self._sky_rval = float(sky_rval[0]), float(sky_rval[1])
        self._slit_width = slit_width
        self._smooth_weights = smooth_weights

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
        options = grid_options(self.grid(), sizes=False)
        del options['rota']
        return DatasetSlitSpectra(
            items['spectra'], **options, slit_pa=self._slit_pa)

    def data_unit(
            self,
            dataset: Dataset | None,
            light: LightUnit | None = None
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

    def check_observation(self, setup: ObservationSetup) -> None:
        warn_unsmoothed_weights(self._smooth_weights, setup.instrument)

    def plan(
            self,
            driver: Driver,
            scene: 'Scene',
            setup: ObservationSetup
    ) -> 'SlitSpectraPlan':
        # The cube of a slit is one pixel (the slit width) across the slit,
        # sampled as finely as the pixels along it (the model is the mean
        # over the slit)
        size, coords = self._grid.size, self._grid.coords
        step, rpix, rval = coords.step, coords.rpix, coords.rval
        oversampling = setup.oversampling
        across = oversampling[0] * max(1, round(self._slit_width / step[0]))
        model_cube = _model_cube.ModelCube(
            Grid((size[0], 1, size[1]), coords._replace(
                step=(step[0], self._slit_width, step[1]),
                rpix=(rpix[0], 0, rpix[1]), rval=(rval[0], 0, rval[1])), 2),
            oversampling=(oversampling[0], across, oversampling[1]),
            primary_beam=setup.instrument.primary_beam(),
            psf=setup.instrument.psf(),
            lsf=setup.instrument.lsf(),
            smooth_weights=self._smooth_weights,
            dtype=setup.dtype,
            fft_dtype=setup.fft_dtype)
        return SlitSpectraPlan(model_cube, driver, scene, setup)


class SlitSpectraPlan(CubeObservablePlan):

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
