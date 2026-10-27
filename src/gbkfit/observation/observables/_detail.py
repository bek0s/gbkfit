"""
Helpers shared by the observables.
"""

import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import astropy.units
import numpy as np

from gbkfit.dataset import Dataset
from gbkfit.driver import Driver
from gbkfit.instrument import Instrument
from gbkfit.model import DataUnit, Extras, GalaxyState
from gbkfit.params import ParamValues
from gbkfit.region import Regions
from gbkfit.utils import gridutils, parseutils, unitutils
from gbkfit.utils.parseutils import ConfigError
from ..weights import plan_weights
from ._dcube import DCube, cube_extra
from .base import HostModelData, ModelData, Observable, ObservablePlan

if TYPE_CHECKING:
    from gbkfit.model import Scene
    from ..line_of_sight import LineOfSight
    from ..foreground import Foreground
    from ..noise import Noise
    from ..weights import Weights


# The options of the spatial grid of the cube of an observable of data
# in regions (see spatial_grid_of_regions)
SPATIAL_OPTIONS = ('size', 'step', 'rpix', 'rval', 'rota')


def load_observable(
        cls: type[Observable],
        info: dict[str, Any],
        dataset: Dataset | None,
        ndim: int
) -> Observable:
    """
    Load an observable of class cls from its configuration: with its
    __init__, or with its from_data and the given dataset, whose options
    it must not repeat. The options of the grid with one value per axis
    (on the sky, ndim of them) are made lists, and its spectral axis is
    loaded (see gridutils.SpectralAxis).
    """
    parseutils.sanitize_dimensional_options(info, dict(
        size=int, step=float, rpix=float, rval=float), ndim)
    parseutils.load_option_and_update_info(
        gridutils.spectral_axis_parser, info, 'spectral')
    if dataset is None:
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))
    require_no_options_from_data(cls, info, dataset)
    return cls.from_data(dataset, **parseutils.parse_options_for_callable(
        info, cls.from_data, ignore_params=['dataset']))


def grid_options(grid: gridutils.Grid) -> dict[str, Any]:
    """
    Return the options of the grid of an observable (size, step, rpix,
    rval and rota, of the axes on the sky, and spectral, its spectral axis
    after them, if any) of the given grid.
    """
    coords = grid.coords
    axes = slice(None) if grid.spectral_axis is None \
        else slice(grid.spectral_axis)
    options = dict(
        size=grid.size[axes], step=coords.step[axes], rpix=coords.rpix[axes],
        rval=coords.rval[axes], rota=coords.rota)
    if grid.spectral_axis is not None:
        options.update(spectral=gridutils.SpectralAxis.of(grid))
    return options


def dump_grid(grid: gridutils.Grid) -> dict[str, Any]:
    """Return the configuration of the grid of an observable."""
    options = grid_options(grid)
    if 'spectral' in options:
        options['spectral'] = options['spectral'].dump()
    return options


def dataset_grid_options(grid: gridutils.Grid) -> dict[str, Any]:
    """
    Return the options of the grid of a dataset on the grid of an
    observable: its grid options but the sizes, which the data give.
    """
    options = grid_options(grid)
    del options['size']
    if 'spectral' in options:
        options['spectral'] = gridutils.SpectralAxis(**{
            key: value for key, value in options['spectral'].dump().items()
            if key != 'size'})
    return options


def require_no_options_from_data(
        cls: type[Observable], info: dict[str, Any], dataset: Dataset
) -> None:
    """
    Raise ConfigError if the options of an observable of class cls (info)
    have one that its data give (see Observable.options_from_data).
    """
    given = [
        key for key in cls.options_from_data(dataset)
        if info.get(key) is not None]
    if given:
        desc = parseutils.make_typed_desc(cls, 'observable')
        raise ConfigError(
            f"the data of {desc} give its options {given}; remove them (the "
            f"world coordinates of data can be set in the data options)")


def without_options_from_data(
        observable: Observable,
        info: dict[str, Any],
        dataset: Dataset | None
) -> dict[str, Any]:
    """
    Return the options of an observable (info, as dumped) without those
    its data give (see Observable.options_from_data), if it has data.
    """
    if dataset is None:
        return info
    options = observable.options_from_data(dataset)
    return {k: v for k, v in info.items() if k not in options}


def check_median_size(median_size: int | None) -> None:
    """
    Raise ConfigError unless the size of a median filter (see
    DCube.median_size) is an odd number of pixels, at least 3 (a window
    centred on each pixel), or None.
    """
    if median_size is not None and not (
            isinstance(median_size, int) and median_size >= 3
            and median_size % 2 == 1):
        raise ConfigError(
            f"median_size must be an odd number of pixels, at least 3; it "
            f"is {median_size!r}")


def check_positive(name: str, value: float) -> None:
    """Raise ConfigError unless the value of an option is positive."""
    if not value > 0:
        raise ConfigError(f"{name} must be positive; it is {value}")


def warn_unsmoothed_weights(
        smooth_weights: bool, instrument: Instrument
) -> None:
    """
    Warn if the weights are to be smoothed, but the instrument has no PSF
    or LSF to smooth them with.
    """
    if smooth_weights and instrument.psf() is None \
            and instrument.lsf() is None:
        parseutils.warn(
            "smooth_weights is true, but the instrument has no PSF or LSF "
            "to smooth the weights with")


def spatial_grid_of_regions(
        regions: Regions,
        size: Sequence[int] | None,
        step: Sequence[float] | None,
        rpix: Sequence[float] | None,
        rval: Sequence[float] | None,
        rota: float | None
) -> gridutils.Grid:
    """
    Return the spatial grid of the cube of an observable of data in
    regions: that of the regions if they have one (bins), when the grid
    options must not be given; else (apertures) the grid of the options
    (see gridutils.make_grid), of which size is required.
    """
    grid = regions.grid()
    given = [
        key for key, value in zip(
            SPATIAL_OPTIONS, (size, step, rpix, rval, rota))
        if value is not None]
    if grid is not None and given:
        raise ConfigError(
            f"the regions are on a grid, which is that of the model; "
            f"remove the options {given}")
    if grid is None:
        if size is None:
            raise ConfigError(
                "the regions are on the sky; the size of the grid of the "
                "model is required")
        grid = gridutils.make_grid(size, step, rpix, rval, rota)
    return grid


def spatial_options_from_regions(regions: Regions) -> tuple[str, ...]:
    """
    Return the options of the spatial grid of an observable of data in
    regions that the regions give (see spatial_grid_of_regions).
    """
    return SPATIAL_OPTIONS if regions.grid() is not None else ()


def dump_spatial_grid(
        observable: Observable, regions: Regions
) -> dict[str, Any]:
    """
    Return the options of the spatial grid of an observable of data in
    regions, unless the regions give it.
    """
    if regions.grid() is not None:
        return {}
    return dict(
        size=observable.size()[:2],
        step=observable.step()[:2],
        rpix=observable.rpix()[:2],
        rval=observable.rval()[:2],
        rota=observable.rota())


def data_unit(
        dataset: Dataset | None,
        key: str | None,
        pixel: float,
        integrated: bool
) -> DataUnit | None:
    """
    Return the unit of the light of a dataset (see Observable.data_unit):
    that of its item of the given key (None: it has none), with its beam,
    whose pixels have the given solid angle (arcsec^2), and whose light
    is integrated over the lines or not; None without data, or without an
    item of light.
    """
    if dataset is None or key is None:
        return None
    data = dataset[key]
    beam = data.beam()
    return DataUnit(
        data.unit(), pixel,
        None if beam is None else beam.area(),
        None if beam is None else beam.spectral, integrated)


def region_data_unit(
        dataset: Dataset | None, key: str | None, integrated: bool
) -> DataUnit | None:
    """
    Return the unit of the light of a dataset of regions (see data_unit),
    whose values are fluxes: the sums of the surface brightness of the
    model over their pixels, times the area of each (see flux_weights),
    so the model takes the unit per arcsec^2 (see flux_per_area).
    """
    result = data_unit(dataset, key, 1.0, integrated)
    if result is None or result.unit is None:
        return result
    return result._replace(
        unit=flux_per_area(result.unit), beam=None, beam_spectral=None)


def flux_per_area(unit: astropy.units.UnitBase) -> astropy.units.UnitBase:
    """
    Return the unit of light per arcsec^2 whose sums over the area of
    regions are data of the given unit, a flux (e.g. Jy, 1e-20
    erg/s/cm2/Angstrom, Jy km/s). Raise ConfigError if the unit is per
    beam or per solid angle (the light of a pixel, not of a region).
    """
    if unitutils.spatial_unit(unit) != astropy.units.dimensionless_unscaled:
        raise ConfigError(
            f"the data of regions are fluxes, sums over their pixels: their "
            f"unit is one of flux (e.g. 'Jy', '1e-20 erg/s/cm2/Angstrom', "
            f"'Jy km/s'); {unitutils.format_unit(unit)!r} is not")
    return unit / astropy.units.arcsec ** 2


class DCubePlanBase(ObservablePlan):
    """
    The evaluation of an observable made from a DCube: the plan of the cube,
    the plan of the scene (what an observation sees) along its line of
    sight on its high-res grid (see _model_grid), that of the pixel
    weights of the data, if any, which fill the weight cube of the DCube,
    and the name of the parameter of the flux scale of the observation, if
    one value (see ObservationSky), which scales the flux of the cube.

    It evaluates the model on the cube, records and masks it (see
    DCubePlan.evaluate), and measures the data items from its low-res
    cube (_measure, of each observable). Its mock data are measured from
    the recorded low-res cube plus noise, masked as the model.

    Raises
    ------
    ConfigError
        If there are pixel weights and a lens (not supported yet), or the
        pixel weights are not in the plane of a disk of the scene.
    """

    def __init__(
            self,
            dcube: DCube,
            driver: Driver,
            scene: 'Scene',
            line_of_sight: 'LineOfSight',
            foreground: 'Foreground',
            dtype: np.dtype,
            pixel_weights: 'Weights | None',
            flux_scale: str | None
    ):
        self._driver = driver
        self._dtype = dtype
        self._dcube = dcube
        self._flux_param = flux_scale
        self._dcube_plan = dcube.plan(
            driver, has_weights=pixel_weights is not None)
        grid = self._model_grid()
        self._weights_plan = None
        if pixel_weights is not None:
            self._weights_plan = plan_weights(
                pixel_weights, scene,
                self._dcube_plan.scratch_grid().spatial(), driver, dtype)
        # With a lens, the model is evaluated on the source plane, into a
        # cube of its own, which is lensed into the high-res cube
        self._lens_plan = None
        self._source_cube = None
        if foreground.lens() is not None:
            if pixel_weights is not None:
                raise ConfigError("a lens does not support pixel_weights yet")
            self._lens_plan = foreground.plan_lens(
                driver, self._dcube_plan.scratch_grid(), dtype)
            grid = self._lens_plan.source_grid(grid)
            nz = self._dcube_plan.scratch_dcube().shape[0]
            self._source_cube = driver.mem_alloc_d(
                (nz,) + grid.size[:2][::-1], dtype)
        # (the smoothing of the spectra checks the resolution of the light;
        # an image has none)
        smoothing = None if grid.spectral_axis is None \
            else dcube.spectral_smoothing()
        self._model_plan = line_of_sight.plan_scene(
            scene, driver, grid, dtype, smoothing)

    def _model_grid(self) -> gridutils.Grid:
        """Return the grid the model is evaluated on: the high-res grid."""
        return self._dcube_plan.scratch_grid()

    def _model_extra(
            self, value: gridutils.GridData
    ) -> gridutils.GridData | np.ndarray:
        """Return an extra output of the model as this observable gives it."""
        return value

    def evaluate(
            self,
            params: ParamValues,
            galaxies: Sequence[GalaxyState],
            out_extra: Extras | None
    ) -> ModelData:
        model_extra = None if out_extra is None else Extras()
        self._evaluate_model(params, galaxies, model_extra)
        self._dcube_plan.evaluate(
            out_extra, self._extra_lo, self._extra_hi,
            self._flux_scale(params))
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
    ) -> list[HostModelData]:
        # The recorded data, without noise, are kept on the host
        self._evaluate_model(params, galaxies, None)
        dcube_plan = self._dcube_plan
        dcube_plan.record(self._flux_scale(params))
        recorded = self._driver.mem_copy_d2h(dcube_plan.dcube())
        draws = [self._mock_once(recorded, noise, rng) for _ in range(ndraws)]
        return draws, self._exact_errors(noise, recorded)

    def has_exact_errors(self) -> bool:
        return False

    def _exact_errors(
            self, noise: 'Noise', recorded: np.ndarray
    ) -> dict[str, np.ndarray] | None:
        """
        Return the exact errors of mock data of the recorded low-res cube
        without noise (on the host), if known (see has_exact_errors);
        None here.
        """
        return None

    # How the extra outputs of the low- and high-res cubes are made from
    # their data and grids (see DCubePlan.evaluate)
    _extra_lo = staticmethod(cube_extra)
    _extra_hi = staticmethod(cube_extra)

    @abc.abstractmethod
    def _measure(self) -> ModelData:
        """Return the model data measured from the low-res cube."""

    def _mock_once(
            self, recorded: np.ndarray, noise: 'Noise', rng: np.random.Generator
    ) -> HostModelData:
        """
        Return the model data measured from the recorded low-res cube (on
        the host) plus a draw of the noise, masked as the model, on the
        host (see mock). Observables whose noise goes in elsewhere (e.g.
        slitless spectra, in the image) override this.
        """
        dcube = self._dcube
        noisy = recorded + noise.draw(
            rng, dcube.grid(), dcube.psf(), recorded)
        self._driver.mem_copy_h2d(
            noisy.astype(self._dtype), self._dcube_plan.dcube())
        # (the data are smoothed with their noise)
        self._dcube_plan.smooth()
        return self._to_host(self._measure())

    def _to_host(self, data: ModelData) -> HostModelData:
        """Return copies of the data and masks of model data on the host."""
        return {
            key: dict(
                d=self._driver.mem_copy_d2h(item['d']),
                m=None if item['m'] is None
                else self._driver.mem_copy_d2h(item['m']))
            for key, item in data.items()}

    def _flux_scale(self, params: ParamValues) -> float | None:
        """Return the flux scale of the observation, if one value."""
        return None if self._flux_param is None \
            else params[self._flux_param]

    def _evaluate_model(
            self,
            params: ParamValues,
            galaxies: Sequence[GalaxyState],
            model_extra: Extras | None
    ) -> None:
        """
        Evaluate the model on the high-res cube (or on the source plane of
        a lens, lensed into it), and the weights of the data, if any.
        """
        dcube_plan = self._dcube_plan
        if self._weights_plan is not None:
            self._weights_plan.evaluate(params, dcube_plan.scratch_wcube())
        # The model adds to its cube, so clear it
        cube = dcube_plan.scratch_dcube() if self._lens_plan is None \
            else self._source_cube
        self._driver.mem_fill(cube, 0)
        self._model_plan.evaluate(params, galaxies, cube, model_extra)
        if self._lens_plan is not None:
            self._lens_plan.evaluate(
                params, cube, dcube_plan.scratch_dcube())
