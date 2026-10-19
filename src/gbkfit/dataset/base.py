import abc
import copy
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Self

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

    def with_data(self, data: Mapping[str, np.ndarray]) -> Self:
        """
        Return a copy with other values of its data items (e.g. those of a
        bootstrap; see Data.with_data), where they were measured.

        Parameters
        ----------
        data : Mapping
            The values of each of its items, of its shape.

        Returns
        -------
        Dataset
            The copy.

        Raises
        ------
        ConfigError
            If they are not the values of its items.
        """
        if set(data) != set(self._items):
            raise ConfigError(
                f"the values must be those of the data items "
                f"{list(self._items)}; they are of {list(data)}")
        dataset = copy.copy(self)
        dataset._items = {
            key: item.with_data(data[key])
            for key, item in self._items.items()}
        return dataset

    def screen_outliers(self, size: Sequence[float], nsigma: float) -> Self:
        """
        Return a copy of it with the outliers of its data items masked:
        the values x further from the median m of their neighbours than
        nsigma times a robust scatter s, |x - m| > nsigma s. The scatter is
        the larger of that of the neighbours, 1.4826 times the median of
        their absolute deviations from m (as the standard deviation of
        Gaussian noise), which keeps the values where the data vary fast,
        and that of the item, 1.4826 times the median of |x - m| over its
        values, which keeps those whose few neighbours agree by chance.
        The neighbours are those with values, the value itself not among
        them; a value with fewer than 3 is kept. Each item is screened by
        itself. The kinds of data
        that offer it say what the neighbours of a value are (e.g. the
        other pixels of a window about it); the others raise ConfigError.

        It is for values that no sky can give: e.g. failed fits of moment
        maps, hot pixels, cosmic rays, spikes of interference. Of data
        seen through a PSF, a window no wider than the beam keeps what is
        real (e.g. compact clumps), which spans the window; on a smooth
        gradient (e.g. the rotation in a velocity map), m is the value
        itself, and s grows with the gradient.

        Parameters
        ----------
        size : Sequence of float
            The size of the neighbourhoods (e.g. of the windows, in
            pixels, along each axis of an image).
        nsigma : float
            How far an outlier is, in units of the scatter.

        Returns
        -------
        Dataset
            The copy.

        Raises
        ------
        ConfigError
            If the kind of data does not offer it, the size is not of its
            neighbourhoods, or nsigma is not positive.
        """
        raise ConfigError(
            f"the data of {self.type()} cannot be screened for outliers")

    def check_blocks(self, size: Sequence[float]) -> None:
        """
        Raise ConfigError unless its data can be redrawn in blocks of a
        size (see resample_blocks): the kinds of data that offer it say
        what their blocks are; the others raise it.
        """
        raise ConfigError(
            f"the residuals of {self.type()} data cannot be redrawn in blocks")

    def resample_blocks(
            self,
            residuals: Mapping[str, np.ndarray],
            size: Sequence[float],
            rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """
        Return residuals of its data redrawn in blocks, for a bootstrap:
        cut into blocks of a size, each of which takes the residuals of a
        block drawn at random (with replacement) from the blocks with
        residuals in all of their values, in every item. The items share
        the draws, so that the correlations of their residuals remain.
        The kinds of data that offer it say what their blocks are (see
        check_blocks); the others raise ConfigError.

        Parameters
        ----------
        residuals : Mapping
            The residuals of each data item, of its shape: NaN where there
            are none (e.g. masked values, or values the model lacks).
        size : Sequence of float
            The size of the blocks (e.g. in pixels, along each axis of an
            image).
        rng : np.random.Generator
            The random numbers.

        Returns
        -------
        dict
            The redrawn residuals of each item, as new arrays: NaN where
            it has none.

        Raises
        ------
        ConfigError
            If the kind of data does not offer it, the size is not one of
            its blocks, or no block has residuals in all of its values.
        """
        raise ConfigError(
            f"the residuals of {self.type()} data cannot be redrawn in blocks")


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
