import abc
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from gbkfit.utils import iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from .data import Data


__all__ = [
    'Dataset',
    'dataset_parser'
]


class Dataset(parseutils.TypedSerializable, Mapping[str, Data], abc.ABC):
    """
    Named data items (see Data) of one shape, and where they were
    measured, which each kind of dataset describes (e.g. the grid of their
    pixels).

    A dataset is a read-only mapping of the names of its items to the
    items. A subclass declares the number of axes of its items (ndim).

    Parameters
    ----------
    items : Mapping
        The data items, by name; at least one.

    Raises
    ------
    ConfigError
        If there are no items, or they are not of one shape with ndim axes.
    """

    ndim: int

    def __init__(self, items: Mapping[str, Data]):
        if not items:
            raise ConfigError("a dataset needs at least one data item")
        invalid = [k for k, v in items.items() if not isinstance(v, Data)]
        if invalid:
            raise ConfigError(f"the data items {invalid} are not Data")
        for key, item in items.items():
            if item.ndim() != self.ndim:
                raise ConfigError(
                    f"the data item {key} has {item.ndim()} axes; expected "
                    f"{self.ndim} (axes of length 1 can be removed with "
                    f"gbkfit-cli prep)")
        shapes = {k: v.shape() for k, v in items.items()}
        if len(set(shapes.values())) > 1:
            raise ConfigError(
                f"the data items must have one shape; their shapes are "
                f"{shapes}")
        self._items = dict(items)

    @abc.abstractmethod
    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        """
        Dump the dataset to its configuration, writing its arrays to files.

        Parameters
        ----------
        prefix : str, optional
            The start of the names of the files.
        dump_path : bool, optional
            Whether the configuration has the paths of the files, or only
            their names.
        overwrite : bool, optional
            Whether to overwrite existing files.

        Returns
        -------
        dict
            The options.
        """

    def __getitem__(self, name: str) -> Data:
        return self._items[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def _first(self) -> Data:
        """The first data item (all have the same shape and dtype)."""
        return next(iter(self.values()))

    def shape(self) -> tuple[int, ...]:
        """
        Return the shape of the arrays of the data items.

        Returns
        -------
        tuple of int
            The shape (numpy order).
        """
        return self._first().shape()

    def dtype(self) -> np.dtype:
        """
        Return the dtype of the arrays of the data items.

        Returns
        -------
        np.dtype
            The dtype.
        """
        return self._first().dtype()


class DatasetTypedParser(parseutils.TypedParser):
    """
    The parser of the datasets: the files of several datasets dumped
    together need a prefix each.
    """

    def __init__(self):
        super().__init__(Dataset)

    def dump_many(
            self, x: Sequence[Dataset], *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        prefix = iterutils.listify(kwargs.get('prefix'))
        if len(x) != len(set(prefix)):
            raise RuntimeError(
                "datasets dumped together need a different prefix each, so "
                "that their files do not overwrite each other")
        return super().dump_many(x, *args, **kwargs)


dataset_parser = DatasetTypedParser()
