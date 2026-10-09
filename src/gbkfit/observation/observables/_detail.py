import numpy as np

from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from .core import ObservablePlan


__all__ = [
    'load_observable_common',
    'DCubePlanBase'
]


def load_observable_common(cls, info, ndim, dataset, expected_dataset_cls):
    """
    The options of an observable of class cls, with the grid of the given
    dataset (if any), which must be of the expected class. The grid
    options of the observable may repeat those of the data, but must not
    differ (the world coordinates of data are set in the data options).
    """
    desc = parseutils.make_typed_desc(cls, 'observable')
    parseutils.sanitize_dimensional_options(info, dict(
        size=int, step=int | float, rpix=int | float, rval=int | float),
        ndim)
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
        grid = dataset.grid()
        data_options = dict(
            size=grid.size,
            step=grid.coords.step,
            rpix=grid.coords.rpix,
            rval=grid.coords.rval,
            rota=grid.coords.rota)
        for key, value in data_options.items():
            given = info.get(key)
            if given is not None and not np.array_equal(given, value):
                raise ConfigError(
                    f"option '{key}' of {desc} is {given}, but the data "
                    f"have {value}; the grid comes from the data (their "
                    f"world coordinates can be set in the data options)")
        info.update(data_options)
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
