from gbkfit.utils import fitsutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from .core import ObservablePlan


__all__ = [
    'SPATIAL_OPTIONS',
    'dump_rest',
    'dump_spatial_grid',
    'load_observable_common',
    'spatial_grid_of_regions',
    'spatial_options_from_regions',
    'require_no_options_from_data',
    'without_options_from_data',
    'DCubePlanBase'
]


# The options of the spatial grid of the cube of an observable of data
# in regions (see spatial_grid_of_regions)
SPATIAL_OPTIONS = ('size', 'step', 'rpix', 'rval', 'rota')


def spatial_grid_of_regions(regions, size, step, rpix, rval, rota):
    """
    The spatial grid of the cube of an observable of data in regions:
    that of the regions if they have one (bins), when the grid options
    must not be given; else (apertures) the grid of the options (see
    fitsutils.make_grid), of which size is required.
    """
    grid = regions.grid()
    given = [
        key for key, value in zip(
            SPATIAL_OPTIONS, (size, step, rpix, rval, rota))
        if value is not None]
    if grid is not None and given:
        raise RuntimeError(
            f"the regions are on a grid, which is that of the model; "
            f"remove the options {given}")
    if grid is None:
        if size is None:
            raise RuntimeError(
                "the regions are on the sky; the size of the grid of the "
                "model is required")
        grid = fitsutils.make_grid(size, step, rpix, rval, rota)
    return grid


def spatial_options_from_regions(regions):
    """
    The options of the spatial grid of an observable of data in regions
    that the regions give (see spatial_grid_of_regions).
    """
    return SPATIAL_OPTIONS if regions.grid() is not None else ()


def dump_rest(rest):
    """The option of the rest of a spectral axis (see fitsutils.Coords)."""
    return None if rest is None else str(rest)


def dump_spatial_grid(observable, regions):
    """
    The options of the spatial grid of an observable of data in regions,
    unless the regions give it.
    """
    if regions.grid() is not None:
        return {}
    return dict(
        size=observable.size()[:2],
        step=observable.step()[:2],
        rpix=observable.rpix()[:2],
        rval=observable.rval()[:2],
        rota=observable.rota())


def require_no_options_from_data(cls, info, dataset):
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


def without_options_from_data(observable, info, dataset):
    """
    The options of an observable (info, as dumped) without those its data
    give (see Observable.options_from_data), if it has data.
    """
    if dataset is None:
        return info
    options = observable.options_from_data(dataset)
    return {k: v for k, v in info.items() if k not in options}


def load_observable_common(cls, info, ndim, dataset, expected_dataset_cls):
    """
    The options of an observable of class cls, with the grid of the given
    dataset (if any; and the rest of its spectral axis), which must be of
    the expected class. The observable must not be given the grid too.
    """
    desc = parseutils.make_typed_desc(cls, 'observable')
    if dataset is not None:
        if not isinstance(dataset, expected_dataset_cls):
            expected_desc = parseutils.make_typed_desc(
                expected_dataset_cls, 'dataset')
            provided_desc = parseutils.make_typed_desc(
                dataset.__class__, 'dataset')
            raise RuntimeError(
                f"{desc} is not compatible with the supplied dataset "
                f"and cannot be used to describe its properties; "
                f"expected dataset type: {expected_desc}; "
                f"provided dataset type: {provided_desc}")
        require_no_options_from_data(cls, info, dataset)
        grid = dataset.grid()
        info.update(
            size=grid.size,
            step=grid.coords.step,
            rpix=grid.coords.rpix,
            rval=grid.coords.rval,
            rota=grid.coords.rota)
        if cls._spectral_axis is not None:
            info.update(rest=grid.coords.rest)
    parseutils.sanitize_dimensional_options(info, dict(
        size=int, step=int | float, rpix=int | float, rval=int | float),
        ndim)
    return parseutils.parse_options_for_callable(info, desc, cls.__init__)


class DCubePlanBase(ObservablePlan):
    """
    The evaluation of an observable made from a DCube: the plan of the cube,
    and the plan of the gmodel on its high-res grid (see _gmodel_grid).
    """

    def __init__(self, dcube, driver, gmodel, dtype):
        self._driver = driver
        self._dtype = dtype
        self._dcube = dcube
        self._dcube_plan = dcube.plan(driver, gmodel.has_weights())
        self._gmodel_plan = gmodel.plan(
            driver, self._gmodel_grid(), gmodel.has_weights(), dtype)

    def _gmodel_grid(self):
        """The grid the gmodel is evaluated on: the high-res grid."""
        return self._dcube_plan.scratch_grid()

    def _gmodel_extra(self, value):
        """An extra output of the gmodel, as this observable gives it."""
        return value

    def _evaluate_cube(self, params, out_extra, extra_lo, extra_hi):
        """
        Evaluate the gmodel on the high-res cube, then convolve, downscale
        and mask it into the low-res cube (see DCubePlan.evaluate).
        """
        dcube_plan = self._dcube_plan
        # The gmodel adds to the cube, so clear it
        self._driver.mem_fill(dcube_plan.scratch_dcube(), 0)
        gmodel_extra = None if out_extra is None else {}
        self._gmodel_plan.evaluate(
            params, dcube_plan.scratch_dcube(), dcube_plan.scratch_wcube(),
            gmodel_extra)
        dcube_plan.evaluate(out_extra, extra_lo, extra_hi)
        if gmodel_extra:
            out_extra.update(
                {f'gmodel_{k}': self._gmodel_extra(v)
                 for k, v in gmodel_extra.items()})
