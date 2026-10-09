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
    (_spectral_axis, None if none), as the datasets do. Its dump(data)
    leaves out the options that the given data give (see
    options_from_data); the files of its options (e.g. bins without
    data) are named with the prefix, as those of the datasets.
    """

    _spectral_axis: int | None

    # The class of the datasets this observable measures (declared by
    # each subclass)
    dataset_class: type

    @staticmethod
    @abc.abstractmethod
    def is_compatible(gmodel):
        pass

    def __init__(self, size, step, rpix, rval, rota, rest=None):
        """
        The grid of the data: its size and world coordinates (see
        fitsutils.Coords; rest is that of the spectral axis, if any).
        """
        if rpix is None:
            rpix = tuple((np.array(size) / 2 - 0.5).tolist())
        self._grid = fitsutils.make_grid(
            tuple(size), tuple(step), tuple(rpix), tuple(rval), rota,
            self._spectral_axis, rest)

    @classmethod
    def options_from_data(cls, dataset) -> tuple[str, ...]:
        """
        The options of the observable that its data (the given dataset)
        give, and that it must not be given too: here, its grid (and the
        rest of its spectral axis). Observables of other data override
        this.
        """
        grid = ('size', 'step', 'rpix', 'rval', 'rota')
        return grid + (('rest',) if cls._spectral_axis is not None else ())

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

    def rest(self):
        """The rest of the spectral axis (see fitsutils.Coords), or None."""
        return self._grid.coords.rest

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

    def require_matching(self, dataset):
        """
        Raise RuntimeError unless the dataset holds the data of this
        observable: it is of its dataset class, has its data items, and
        was measured where the observable is modelled.
        """
        desc = parseutils.make_typed_desc(self.__class__, 'observable')
        if not isinstance(dataset, self.dataset_class):
            dataset_desc = parseutils.make_typed_desc(
                dataset.__class__, 'dataset')
            raise RuntimeError(
                f"{desc} cannot be compared with {dataset_desc}")
        if set(dataset.keys()) != set(self.keys()):
            raise RuntimeError(
                f"{desc} has the data items {sorted(self.keys())}, but the "
                f"data have {sorted(dataset.keys())}")
        self._require_matching_coordinates(dataset)

    def _require_matching_coordinates(self, dataset):
        """
        Raise RuntimeError unless the data were measured on the grid of
        this observable. Observables of other data override this.
        """
        if dataset.grid() != self._grid:
            desc = parseutils.make_typed_desc(self.__class__, 'observable')
            raise RuntimeError(
                f"the data are on the grid {dataset.grid()}, but {desc} "
                f"is on the grid {self._grid}")

    def output(self, data):
        """
        An array of the form of the data of this observable (the model,
        its mask or weights, or a residual) as an output: here, on the
        grid of the observable with its world coordinates (a GridData).
        Observables of other data override this.
        """
        return fitsutils.GridData(
            data, self._grid.coords, self._grid.spectral_axis)

    @abc.abstractmethod
    def plan(
            self, driver, gmodel, foreground, instrument, scale, dtype,
            selection):
        """
        The evaluation of the gmodel as this observable, through the
        foreground and the instrument, on the given driver and dtype, with
        the model oversampled scale times along each axis of the data (an
        ObservablePlan): of what the selection of the gmodel has (see
        Selection).
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
