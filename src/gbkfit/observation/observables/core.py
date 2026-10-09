import abc

import numpy as np

from gbkfit.utils import fitsutils, parseutils


__all__ = [
    'Observable',
    'ObservablePlan'
]


class Observable(parseutils.TypedSerializable, abc.ABC):
    """
    What a dataset measures, made from a gmodel seen through an
    instrument: the form of the data (their items, see keys()) and their
    grid. A subclass declares the index of the spectral axis of its data
    (_spectral_axis, None if none), as the datasets do.
    """

    _spectral_axis: int | None

    @staticmethod
    @abc.abstractmethod
    def is_compatible(gmodel):
        pass

    def __init__(self, size, step, rpix, rval, rota):
        if rpix is None:
            rpix = tuple((np.array(size) / 2 - 0.5).tolist())
        self._grid = fitsutils.Grid(
            tuple(size),
            fitsutils.Coords(tuple(step), tuple(rpix), tuple(rval), rota),
            self._spectral_axis)

    def spectral_axis(self):
        return self._spectral_axis

    def grid(self) -> fitsutils.Grid:
        """The grid of the data."""
        return self._grid

    def size(self):
        return self._grid.size

    def step(self):
        return self._grid.coords.step

    def rpix(self):
        return self._grid.coords.rpix

    def rval(self):
        return self._grid.coords.rval

    def rota(self):
        return self._grid.coords.rota

    def zero(self):
        return self._grid.zero()

    def npix(self):
        return int(np.prod(self.size()))

    @abc.abstractmethod
    def keys(self):
        """The names of the data items (e.g. 'scube')."""
        pass

    def require_compatible(self, gmodel):
        """Raise RuntimeError unless the gmodel can be observed as this."""
        if not self.is_compatible(gmodel):
            observable_desc = parseutils.make_typed_desc(
                self.__class__, 'observable')
            gmodel_desc = parseutils.make_typed_desc(gmodel.__class__, 'gmodel')
            raise RuntimeError(
                f"{observable_desc} is not compatible with {gmodel_desc}")

    @abc.abstractmethod
    def plan(self, driver, gmodel, instrument, scale, dtype):
        """
        The evaluation of the gmodel as this observable, seen through the
        instrument, on the given driver and dtype, with the model
        oversampled scale times along each axis of the data (an
        ObservablePlan).
        """
        pass


class ObservablePlan(abc.ABC):
    """The evaluation of a gmodel as an observable."""

    @abc.abstractmethod
    def evaluate(self, params, out_extra):
        """
        The model data: for each data item (see Observable.keys), the
        (device) arrays of the data ('d'), mask ('m') and weights ('w'),
        or None. The extra outputs of the gmodel are prefixed 'gmodel_'.
        """
        pass
