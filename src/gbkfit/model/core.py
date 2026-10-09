
import abc

import numpy as np

from gbkfit.utils import parseutils


__all__ = [
    'DModel',
    'GModel',
    'GModelImage',
    'GModelSCube',
    'dmodel_parser',
    'gmodel_parser'
]


class DModel(parseutils.TypedSerializable, abc.ABC):
    """
    A data model. A subclass declares the index of the spectral axis of
    its data (_spectral_axis, None if none), as the datasets do.
    """

    _spectral_axis: int | None

    @staticmethod
    @abc.abstractmethod
    def is_compatible(gmodel):
        pass

    def spectral_axis(self):
        return self._spectral_axis

    def __init__(self):
        # The driver and gmodel the dmodel was prepared for, and the plan
        # of the gmodel on the high-res grid of the dmodel
        self._driver = None
        self._gmodel = None
        self._gmodel_plan = None

    def npix(self):
        return int(np.prod(self.size()))

    @abc.abstractmethod
    def keys(self):
        pass

    @abc.abstractmethod
    def size(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def zero(self):
        pass

    def require_compatible(self, gmodel):
        """Raise RuntimeError unless the dmodel can evaluate the gmodel."""
        if not self.is_compatible(gmodel):
            dmodel_desc = parseutils.make_typed_desc(self.__class__, 'dmodel')
            gmodel_desc = parseutils.make_typed_desc(gmodel.__class__, 'gmodel')
            raise RuntimeError(
                f"{dmodel_desc} is not compatible with {gmodel_desc}")

    def _prepare(self, driver, gmodel):
        self.require_compatible(gmodel)
        self._driver = driver
        self._gmodel = gmodel
        try:
            self._prepare_impl(gmodel)
        except Exception:
            # Prepare again on the next evaluation
            self._driver = None
            self._gmodel = None
            self._gmodel_plan = None
            raise

    def evaluate(self, driver, gmodel, params, out_extra=None):
        if self._driver is not driver or self._gmodel is not gmodel:
            self._prepare(driver, gmodel)
        out_dmodel_extra = None if out_extra is None else {}
        out_gmodel_extra = None if out_extra is None else {}
        out = self._evaluate_impl(params, out_dmodel_extra, out_gmodel_extra)
        if out_dmodel_extra:
            out_extra.update(
                {f'dmodel_{k}': v for k, v in out_dmodel_extra.items()})
        if out_gmodel_extra:
            out_extra.update(
                {f'gmodel_{k}': v for k, v in out_gmodel_extra.items()})
        return out

    @abc.abstractmethod
    def _prepare_impl(self, gmodel):
        pass

    @abc.abstractmethod
    def _evaluate_impl(self, params, out_dmodel_extra, out_gmodel_extra):
        pass


class GModel(parseutils.TypedSerializable, abc.ABC):

    @abc.abstractmethod
    def pdescs(self):
        pass

    @abc.abstractmethod
    def has_weights(self):
        pass

    def constants(self):
        """
        Values that parameter expressions can use (e.g. the radial nodes
        of a disk), by name.
        """
        return {}

    @abc.abstractmethod
    def plan(self, driver, grid, has_weights, dtype):
        """
        The evaluation of the gmodel on the given driver, grid of its data
        (fitsutils.Grid: x and y, and the spectral axis of spectral cubes)
        and dtype, with spatial weights if has_weights (a GModelPlan).
        """
        pass


class GModelPlan(abc.ABC):
    """The evaluation of a gmodel on a driver, grid and dtype."""

    @abc.abstractmethod
    def evaluate(self, params, data, weights, out_extra):
        """
        Add the gmodel to data (the image or spectral cube on the grid of
        the plan), and weight the data weights (or None) with its spatial
        weights.
        """
        pass


class GModelImage(GModel, abc.ABC):
    """A gmodel evaluated into images."""


class GModelSCube(GModel, abc.ABC):
    """A gmodel evaluated into spectral cubes."""


dmodel_parser = parseutils.TypedParser(DModel)
gmodel_parser = parseutils.TypedParser(GModel)
