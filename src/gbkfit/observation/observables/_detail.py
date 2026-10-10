"""
Helpers shared by the observables.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import astropy.units
import numpy as np

from gbkfit.dataset import Dataset
from gbkfit.driver import Driver
from gbkfit.instrument import Instrument
from gbkfit.model.base import GModel, Selection
from gbkfit.region import Regions
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from ._dcube import DCube, ExtraFunction
from .base import Observable, ObservablePlan

if TYPE_CHECKING:
    from ..foreground import Foreground


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
    (ndim of them) are made lists.
    """
    parseutils.sanitize_dimensional_options(info, dict(
        size=int, step=float, rpix=float, rval=float), ndim)
    if dataset is None:
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))
    require_no_options_from_data(cls, info, dataset)
    return cls.from_data(dataset, **parseutils.parse_options_for_callable(
        info, cls.from_data, ignore_params=['dataset']))


def grid_options(grid: gridutils.Grid) -> dict[str, Any]:
    """
    Return the options of the grid of an observable (size, step, rpix,
    rval, rota, and rest with a spectral axis) of the given grid.
    """
    coords = grid.coords
    options = dict(
        size=grid.size, step=coords.step, rpix=coords.rpix, rval=coords.rval,
        rota=coords.rota)
    if grid.spectral_axis is not None:
        options.update(rest=coords.rest)
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


def check_mask_options(mask_cutoff: float | None, mask_apply: bool) -> None:
    """
    Raise ConfigError if the mask cutoff is negative, or if the mask is
    applied without one (with masking disabled).
    """
    if mask_cutoff is not None and mask_cutoff < 0:
        raise ConfigError(
            f"mask_cutoff must not be negative; it is {mask_cutoff}")
    if mask_apply and mask_cutoff is None:
        raise ConfigError(
            "mask_apply needs a mask_cutoff, but masking is disabled "
            "(mask_cutoff is null)")


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


def dump_rest(rest: astropy.units.Quantity | None) -> str | None:
    """Return the option of the rest of a spectral axis."""
    return None if rest is None else str(rest)


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


class DCubePlanBase(ObservablePlan):
    """
    The evaluation of an observable made from a DCube: the plan of the cube,
    and the plan of the gmodel (of the given selection) on its high-res
    grid (see _gmodel_grid).
    """

    def __init__(
            self,
            dcube: DCube,
            driver: Driver,
            gmodel: GModel,
            foreground: 'Foreground',
            dtype: np.dtype,
            selection: Selection
    ):
        self._driver = driver
        self._dtype = dtype
        self._dcube = dcube
        self._dcube_plan = dcube.plan(driver, gmodel.has_weights())
        grid = self._gmodel_grid()
        # With a lens, the gmodel is evaluated on the source plane, into a
        # cube of its own, which is lensed into the high-res cube
        self._lens_plan = None
        self._source_cube = None
        lens = foreground.lens()
        if lens is not None:
            if gmodel.has_weights():
                raise ConfigError(
                    "a lens does not support gmodels with weights (wtraits) "
                    "yet")
            self._lens_plan = lens.plan(
                driver, self._dcube_plan.scratch_grid(), dtype)
            grid = self._lens_plan.source_grid(grid)
            nz = self._dcube_plan.scratch_dcube().shape[0]
            self._source_cube = driver.mem_alloc_d(
                (nz,) + grid.size[:2][::-1], dtype)
        self._gmodel_plan = gmodel.plan(
            driver, grid, gmodel.has_weights(), dtype, selection)

    def _gmodel_grid(self) -> gridutils.Grid:
        """Return the grid the gmodel is evaluated on: the high-res grid."""
        return self._dcube_plan.scratch_grid()

    def _gmodel_extra(
            self, value: gridutils.GridData
    ) -> gridutils.GridData | np.ndarray:
        """Return an extra output of the gmodel as this observable gives it."""
        return value

    def _evaluate_cube(
            self,
            params: dict[str, float | np.ndarray],
            out_extra: dict[str, Any] | None,
            extra_lo: ExtraFunction,
            extra_hi: ExtraFunction
    ) -> None:
        """
        Evaluate the gmodel on the high-res cube (or on the source plane of
        a lens, lensed into it), then convolve, downscale and mask it into
        the low-res cube (see DCubePlan.evaluate).
        """
        dcube_plan = self._dcube_plan
        # The gmodel adds to its cube, so clear it
        cube = dcube_plan.scratch_dcube() if self._lens_plan is None \
            else self._source_cube
        self._driver.mem_fill(cube, 0)
        gmodel_extra = None if out_extra is None else {}
        self._gmodel_plan.evaluate(
            params, cube, dcube_plan.scratch_wcube(), gmodel_extra)
        if self._lens_plan is not None:
            self._lens_plan.evaluate(cube, dcube_plan.scratch_dcube())
        dcube_plan.evaluate(out_extra, extra_lo, extra_hi)
        if gmodel_extra:
            out_extra.update(
                {f'gmodel_{k}': self._gmodel_extra(v)
                 for k, v in gmodel_extra.items()})
