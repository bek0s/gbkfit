
import abc

import numpy as np

from gbkfit.dataset.data import Data
from gbkfit.utils import iterutils, parseutils


__all__ = [
    'Dataset',
    'dataset_parser'
]


class Dataset(parseutils.TypedSerializable, abc.ABC):
    """
    Named data items (see Data) of one shape, and where they were
    measured, which each kind of dataset describes (e.g. the grid of their
    pixels). A subclass declares the number of axes of its items (ndim).
    """

    ndim: int

    def __init__(self, items: dict[str, Data]):
        # At least one data item must be defined
        if not items:
            raise RuntimeError("dataset contains no data items")
        # All data items must be of the right type
        invalid = [k for k, v in items.items() if not isinstance(v, Data)]
        if invalid:
            raise RuntimeError(
                f"dataset contains invalid data items: {invalid}")
        # All data items must have the axes of the dataset
        for key, item in items.items():
            if item.ndim() != self.ndim:
                raise RuntimeError(
                    f"data item {key} has {item.ndim()} axes; expected "
                    f"{self.ndim} (axes of length 1 can be removed with "
                    f"gbkfit-cli prep)")
        # All data items must have the same shape
        shapes = {k: v.shape() for k, v in items.items()}
        if len(set(shapes.values())) > 1:
            raise RuntimeError(
                f"dataset contains data items of different shapes: {shapes}")
        self._items = dict(items)

    def __contains__(self, item):
        return item in self._items

    def __getitem__(self, item):
        return self._items[item]

    def __iter__(self):
        return iter(self._items)

    def items(self):
        return self._items.items()

    def keys(self):
        return self._items.keys()

    def values(self):
        return self._items.values()

    def get(self, item, default=None):
        return self._items.get(item, default)

    # All data items have the same shape and dtype (see __init__())
    def _first(self) -> Data:
        return next(iter(self.values()))

    def shape(self) -> tuple[int, ...]:
        """The shape of the arrays of the data items (numpy order)."""
        return self._first().shape()

    def dtype(self) -> np.dtype:
        return self._first().dtype()


class DatasetTypedParser(parseutils.TypedParser):

    def __init__(self):
        super().__init__(Dataset)

    def dump_many(self, x, *args, **kwargs):
        # Ensure that a unique prefix for each dataset is provided
        # in order to avoid datasets overwriting each other
        prefix = iterutils.listify(kwargs.get('prefix'), False)
        if len(x) != len(set(prefix)):
            raise RuntimeError(
                "when dumping multiple datasets,"
                "a unique prefix for each dataset must be provided")
        return super().dump_many(x, *args, **kwargs)


dataset_parser = DatasetTypedParser()
