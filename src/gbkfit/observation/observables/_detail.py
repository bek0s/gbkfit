from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from .core import ObservablePlan


__all__ = [
    'load_observable_common',
    'require_no_options_from_data',
    'without_options_from_data',
    'DCubePlanBase'
]


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
    dataset (if any), which must be of the expected class. The observable
    must not be given the grid too.
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
