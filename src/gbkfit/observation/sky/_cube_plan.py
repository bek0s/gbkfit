"""
The base plan of the observations on the sky made from a ModelCube
(see CubePlan), and what it draws (see SkyDrawing).
"""

import abc
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from gbkfit.driver import Driver
from gbkfit.grid import Grid, GridData
from gbkfit.model import Extras, GalaxyState
from gbkfit.params import ParamValues
from .._scene import SeenScenes
from ..depth import plan_scenes
from ..weights import plan_weights
from ._model_cube import ExtraFunction, ModelCube, cube_extra
from ..plan import HostModelData, ModelData, ModelItem, ObservationPlan

if TYPE_CHECKING:
    from ..noise import Noise
    from .base import SkyOptions


class SkyDrawing(NamedTuple):
    """
    What the plan of an observation on the sky draws (see
    ObservationSky.plan): its scenes (see SeenScenes), the values of the
    parameters at which what is laid out once is laid out (see
    ParamSpace.reference; the grids of its planes behind lenses), if
    given, and how to find the pixels of the grid of its model cube whose
    light its data measure (see ObservationSky._measured_pixels), from
    which the rays through the lenses are traced.
    """
    scenes: SeenScenes
    reference: ParamValues | None
    measured: Callable[[], np.ndarray]


class CubePlan(ObservationPlan):
    """
    The evaluation of an observation made from a ModelCube: the plan of the
    cube, the plan of the scenes (what an observation sees) along its line
    of sight on its high-resolution grid (see _model_grid; those of the
    planes behind lenses on grids of their own, lensed into it, see
    plan_scenes), that of the pixel weights of the data, if any, which
    fill the weight cube of the ModelCube, and the name of the parameter
    of the flux scale of the observation, if one value (see
    ObservationSky), which scales the flux of the cube.

    It evaluates the model on the cube, records and masks it (see
    ModelCubePlan.evaluate), and measures the data items from its
    low-resolution cube (_measure, of each kind). Its mock data are
    measured from the recorded low-resolution cube plus noise, masked as
    the model (see _measure_draw), after the model without noise is
    measured once, which raises if it failed (e.g. its moments).

    Parameters
    ----------
    model_cube : ModelCube
        The cube of the model.
    driver : Driver
        The driver it is evaluated on.
    drawing : SkyDrawing
        What is seen.
    options : SkyOptions
        The options of the observation (see ObservationSky._plan).

    Raises
    ------
    ValueError
        If the pixel weights are not in the plane of a disk of the
        galaxies seen (see weights_geometry), or the scenes cannot be
        planned (see plan_scenes).
    """

    def __init__(
            self,
            model_cube: ModelCube,
            driver: Driver,
            drawing: SkyDrawing,
            options: 'SkyOptions'
    ):
        dtype = options.dtype
        scenes = drawing.scenes
        self._driver = driver
        self._dtype = dtype
        self._model_cube = model_cube
        self._flux_scale_param = options.flux_scale_param
        # (the unit of the light of the model, if known)
        data_unit = scenes.data_unit()
        self._light_unit = None if data_unit is None \
            else data_unit.cube_unit()
        self._model_cube_plan = model_cube_plan = model_cube.plan(
            driver, has_weights=options.pixel_weights is not None)
        grid = self._model_grid()
        self._weights_plan = None
        if options.pixel_weights is not None:
            self._weights_plan = plan_weights(
                options.pixel_weights, scenes.seen,
                model_cube_plan.high_res_grid().spatial(), driver, dtype)
        # (the smoothing of the spectra checks the resolution of the light;
        # an image has none)
        smoothing = None if grid.spectral_axis is None \
            else model_cube.spectral_smoothing()
        self._model_plan = plan_scenes(
            scenes, options.depth, options.source_grid, driver, grid, dtype,
            smoothing,
            lambda: model_cube_plan.footprint(drawing.measured()),
            drawing.reference)

    def _model_grid(self) -> Grid:
        """
        Return the grid the model is evaluated on: the high-resolution
        grid.
        """
        return self._model_cube_plan.high_res_grid()

    def _model_extra(
            self, value: GridData
    ) -> GridData | np.ndarray:
        """Return an extra output of the model as this observation gives it."""
        return value

    def evaluate(
            self,
            params: ParamValues,
            galaxies: Sequence[GalaxyState],
            out_extra: Extras | None
    ) -> ModelData:
        model_extra = None if out_extra is None else Extras()
        self._evaluate_model(params, galaxies, model_extra)
        self._model_cube_plan.evaluate(
            out_extra, self._extra_lo, self._extra_high_res,
            self._flux_scale(params), self._light_unit)
        if model_extra:
            for extra in model_extra:
                out_extra.add(
                    extra.name, self._model_extra(extra.value), extra.title,
                    extra.description, **extra.ids)
        return self._measure()

    def mock(
            self,
            params: ParamValues,
            galaxies: Sequence[GalaxyState],
            noise: 'Noise',
            rng: np.random.Generator,
            ndraws: int
    ) -> tuple[list[HostModelData], dict[str, np.ndarray] | None]:
        # The recorded data, without noise, are kept on the host
        self._evaluate_model(params, galaxies, None)
        model_cube_plan = self._model_cube_plan
        model_cube_plan.record(self._flux_scale(params))
        recorded = self._driver.mem_copy_d2h(model_cube_plan.cube())
        # (a model that failed raises here, not in the draws)
        self._measure()
        draws = [self._mock_once(recorded, noise, rng) for _ in range(ndraws)]
        return draws, self._exact_errors(noise, recorded)

    def has_exact_errors(self) -> bool:
        return False

    def _exact_errors(
            self, noise: 'Noise', recorded: np.ndarray
    ) -> dict[str, np.ndarray] | None:
        """
        Return the exact errors of mock data of the recorded
        low-resolution cube without noise (on the host), if known (see
        has_exact_errors); None here.
        """
        return None

    # How the extra outputs of the low- and high-resolution cubes are made
    # from their data and grids (see ModelCubePlan.evaluate); kinds whose
    # model data are the low-resolution cubes (their outputs hold them)
    # have none of those (None)
    _extra_lo: ExtraFunction | None = staticmethod(cube_extra)
    _extra_high_res: ExtraFunction = staticmethod(cube_extra)

    @abc.abstractmethod
    def _measure(self) -> ModelData:
        """Return the model data measured from the low-resolution cube."""

    def _measure_draw(self) -> ModelData:
        """
        Return the model data measured from the low-resolution cube of a
        mock draw (see _mock_once): here as _measure. Kinds whose
        measurement of a model checks that it did not fail (e.g. moments,
        see MomentsPlan.evaluate) measure a draw as data, whose noise may
        blank values on purpose (see Noise).
        """
        return self._measure()

    def _mock_once(
            self,
            recorded: np.ndarray,
            noise: 'Noise',
            rng: np.random.Generator
    ) -> HostModelData:
        """
        Return the model data measured from the recorded low-resolution
        cube (on the host) plus a draw of the noise, masked as the model,
        on the host (see mock). Kinds whose noise goes in elsewhere (e.g.
        slitless spectra, in the image) override this.
        """
        model_cube = self._model_cube
        noisy = recorded + noise.draw(
            rng, model_cube.grid(), model_cube.psf(), recorded)
        self._driver.mem_copy_h2d(
            noisy.astype(self._dtype), self._model_cube_plan.cube())
        # (the data are smoothed with their noise)
        self._model_cube_plan.smooth()
        return self._to_host(self._measure_draw())

    def _to_host(self, data: ModelData) -> HostModelData:
        """Return copies of the data and masks of model data on the host."""
        return {
            key: ModelItem(
                self._driver.mem_copy_d2h(item.data),
                None if item.mask is None
                else self._driver.mem_copy_d2h(item.mask))
            for key, item in data.items()}

    def _flux_scale(self, params: ParamValues) -> float | None:
        """Return the flux scale of the observation, if one value."""
        return None if self._flux_scale_param is None \
            else params[self._flux_scale_param]

    def _evaluate_model(
            self,
            params: ParamValues,
            galaxies: Sequence[GalaxyState],
            model_extra: Extras | None
    ) -> None:
        """
        Evaluate the model on the high-resolution cube, and the weights of
        the data, if any.
        """
        model_cube_plan = self._model_cube_plan
        if self._weights_plan is not None:
            self._weights_plan.evaluate(
                params, model_cube_plan.high_res_weights())
        # The model adds to its cube, so clear it
        cube = model_cube_plan.high_res_cube()
        self._driver.mem_fill(cube, 0)
        self._model_plan.evaluate(params, galaxies, cube, model_extra)
