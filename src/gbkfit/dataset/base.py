"""
The base of the datasets: named data items of one shape, and where they
were measured (see Dataset; DatasetGrid, of data on a grid), and the
correlation of their noise (see NoiseCorrelation).
"""

import abc
import copy
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, NamedTuple, Self

import numpy as np

from gbkfit.config import (
    ConfigProblem, TypedParser, TypedSerializable, collect_problems,
    config_path, constructor_of, parse_options, parse_options_for_callable)
from gbkfit.grid import Grid, SpectralAxis
from gbkfit.utils import iterutils
from . import _grids, _outliers, _resampling
from .deferred import Deferred
from .data import Data, DataFiles


__all__ = [
    'Dataset',
    'DatasetGrid',
    'NoiseCorrelation',
    'check_files',
    'dataset_parser'
]


_log = logging.getLogger(__name__)


class NoiseCorrelation(NamedTuple):
    """
    The correlation of the noise of a data item on the separations of its
    values (see Dataset.noise_correlation): of two values dy and dx pixels
    apart on the sky, spatial[ny - 1 + dy, nx - 1 + dx], and dz channels
    apart along the spectral axis, spectral[nz - 1 + dz] (ny, nx and nz
    the numbers of pixels and channels of the data); 1 at a separation of
    0, and None along the axes along which it is not correlated.
    """
    spatial: np.ndarray | None
    spectral: np.ndarray | None


class Dataset(TypedSerializable, Mapping[str, Data], abc.ABC):
    """
    Named data items (see Data) of one shape, and where they were
    measured, which each kind of dataset describes (e.g. the grid of their
    pixels).

    A dataset is a read-only mapping of the names of its items to the
    items. A subclass declares the number of axes of its items (ndim),
    the index of their spectral axis, if any (spectral_axis, FITS order:
    the last, the first in numpy order), and the axes along which their
    residuals are cut into blocks for bootstraps, if they are
    (block_axes, FITS order; their names, for messages, e.g. ('x', 'y');
    see resample_blocks). Only data with a spectral axis can have noise
    correlated along it (see Data and noise_correlation).

    Parameters
    ----------
    items : Mapping
        The data items, by name; at least one.

    Raises
    ------
    ValueError
        If there are no items, they are not of one shape with ndim axes,
        or the data have no spectral axis and an item has a spectral
        correlation of its noise.
    """

    ndim: int

    spectral_axis: int | None = None

    # (None: not cut along its axes; see check_blocks)
    block_axes: tuple[str, ...] | None = None

    # The titles of the options of several kinds: those of the files of a
    # data item (see DataFiles), of the kinds of one item, of where the
    # instrument looked (see DatasetGrid), and those of the reading and
    # cleaning of the data
    config_titles = DataFiles.config_titles | dict(
        file='File', format='Format', regions='Regions',
        footprint='Footprint', spectral='Spectral axis',
        outliers='Outlier screening', mask_whole_spectra='Mask whole spectra')

    def __init__(self, items: Mapping[str, Data]):
        if not items:
            raise ValueError("a dataset needs at least one data item")
        invalid = [k for k, v in items.items() if not isinstance(v, Data)]
        if invalid:
            raise ValueError(f"the data items {invalid} are not Data")
        for key, item in items.items():
            if item.ndim() != self.ndim:
                raise ValueError(
                    f"the data item {key} has {item.ndim()} axes; expected "
                    f"{self.ndim} (axes of length 1 can be removed with "
                    f"gbkfit-cli prep)")
        shapes = {k: v.shape() for k, v in items.items()}
        if len(set(shapes.values())) > 1:
            raise ValueError(
                f"the data items must have one shape; their shapes are "
                f"{shapes}")
        if self.spectral_axis is None and (correlated := [
                key for key, item in items.items()
                if item.spectral_correlation() is not None]):
            raise ValueError(
                f"the noise of the data items {correlated} has a spectral "
                f"correlation, but {self.type()} data have no spectral axis")
        self._items = dict(items)

    @classmethod
    def file_problems(
            cls, options: Mapping[str, Any]) -> list[ConfigProblem]:
        """
        Return the problems of how the files of a dataset of the type fit
        together, from their headers alone, before their data are read
        (see check_files), each at the path of its option: of the options
        of its constructor of a configuration, as parsed (see
        config_constructors). A type whose files are checked only as they
        are read has none, as here.

        Parameters
        ----------
        options : Mapping
            The options.

        Returns
        -------
        list of ConfigProblem
            The problems.
        """
        return []

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
        ValueError
            If they are not the values of its items.
        """
        if set(data) != set(self._items):
            raise ValueError(
                f"the values must be those of the data items "
                f"{list(self._items)}; they are of {list(data)}")
        dataset = copy.copy(self)
        dataset._items = {
            key: item.with_data(data[key])
            for key, item in self._items.items()}
        return dataset

    def with_values(
            self,
            values: Callable[[str, Data], np.ndarray],
            what: str | None = None
    ) -> Self:
        """
        Return a copy with other values of its data items, each made of an
        item with its values, its mask and errors kept (see
        Data.with_data): at once if its values have been read, else as
        they are first used (see Data.then). The values that each loses
        are logged, if what it is is given (e.g. 'outliers masked').

        Parameters
        ----------
        values : Callable
            Makes the values of an item, as values(key, item).
        what : str, optional
            What the change is, in the log.

        Returns
        -------
        Dataset
            The copy.
        """
        def changed(key: str, item: Data) -> Data:
            def change(read: Data) -> Data:
                after = read.with_data(values(key, read))
                if what is not None:
                    lost = int(np.count_nonzero(read.mask())
                               - np.count_nonzero(after.mask()))
                    _log.info(f"{what} of {key}: {lost}")
                return after
            return item.then(change)
        dataset = copy.copy(self)
        dataset._items = {
            key: changed(key, item) for key, item in self._items.items()}
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
        itself. The kinds of data that offer it say what the neighbours of
        a value are (e.g. the other pixels of a window about it); the
        others raise ValueError.

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
        ValueError
            If the kind of data does not offer it, the size is not of its
            neighbourhoods, or nsigma is not positive.
        """
        raise ValueError(
            f"the data of {self.type()} cannot be screened for outliers")

    def check_blocks(self, size: Sequence[float]) -> None:
        """
        Raise ValueError unless its data can be redrawn in blocks of a
        size (see resample_blocks): a positive integer for each of the
        axes of its blocks (block_axes). The kinds of data whose blocks are
        not along their axes say what they are (e.g. cells of the uv
        plane); those without blocks raise it.
        """
        _resampling.block_size(size, self._required_block_axes())

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
        The blocks are along its axes (block_axes), unless the kind of
        data says otherwise (e.g. cells of the uv plane; see
        check_blocks); the kinds of data without blocks raise ValueError.

        Parameters
        ----------
        residuals : Mapping
            The residuals of each data item, of its shape: NaN where there
            are none (e.g. masked values, or values missing from the
            model).
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
        ValueError
            If the kind of data does not offer it, the size is not one of
            its blocks, or no block has residuals in all of its values.
        """
        return _resampling.resample_blocks(
            residuals, size, self._required_block_axes(), rng)

    def _required_block_axes(self) -> tuple[str, ...]:
        """
        Return the axes of its blocks (see block_axes), or raise
        ValueError if its residuals are not cut along its axes.
        """
        if self.block_axes is None:
            raise ValueError(
                f"the residuals of {self.type()} data cannot be redrawn in "
                f"blocks")
        return self.block_axes

    def noise_correlation(self, key: str) -> NoiseCorrelation | None:
        """
        Return the correlation of the noise of a data item on the
        separations of its values (see Data), or None if it has none. The
        kinds of data whose noise can be correlated say where (e.g. on the
        sky, for the pixels of an image); the others raise ValueError.

        Parameters
        ----------
        key : str
            The name of the data item.

        Returns
        -------
        NoiseCorrelation or None
            The correlation.

        Raises
        ------
        ValueError
            If the kind of data, or its axes, cannot have it.
        """
        item = self[key]
        if item.spatial_correlation() is None \
                and item.spectral_correlation() is None:
            return None
        raise ValueError(
            f"the noise of {self.type()} data cannot be correlated")

    def footprint_of(self, key: str) -> np.ndarray | None:
        """
        Return where the instrument looked (1 inside, 0 outside) on the
        values of a data item, as its dataset says (see DatasetGrid); None
        if it says nothing (everywhere).

        Parameters
        ----------
        key : str
            The name of the data item.

        Returns
        -------
        np.ndarray or None
            The footprint, of the shape of the item.
        """
        return None

    def mask_whole_spectra(self) -> Self:
        """
        Return a copy of it with each spectrum of its data items masked
        whole where they mask any of its values (the union of the masks
        of the channels), as correlated noise needs (see
        LikelihoodGauss). The channels they mask in every spectrum
        (e.g. a plane blank everywhere) are left out of the union: they
        stay masked, and mask no spectrum. The kinds of data of spectra
        offer it (of pixels, along a slit, or of regions); the others
        raise ValueError.

        Returns
        -------
        Dataset
            The copy.

        Raises
        ------
        ValueError
            If the kind of data does not offer it.
        """
        raise ValueError(
            f"the spectra of {self.type()} data cannot be masked whole")


class DatasetGrid(Dataset):
    """
    A dataset on a grid (see Grid): its items have a value in each cell
    (e.g. each pixel of an image, or each position along a slit and
    channel of long-slit spectra). The grid is of the shape of the items
    and of the world coordinates given, with a spectral axis, the last,
    if the kind of data has one (see spectral_axis).

    Its outliers are screened in windows along the axes of its grid, as
    its residuals are redrawn in blocks along them (see block_axes,
    screen_outliers and resample_blocks).

    Its footprint says where the instrument looked (e.g. the field of an
    IFU, whose corners are blank but were never observed): 0 outside,
    where the values of its items are masked too, and any other value
    inside (e.g. the number of fibres on each pixel, as CALIFA gives it),
    kept as 1; of the shape of
    its items, or, of a cube (two axes on the sky and a spectral axis), of
    its sky alone, for every channel. Of the masked values, those inside
    are non-detections, which an upper limit of their likelihood takes
    (see LikelihoodGauss), and those outside say nothing; without a
    footprint, every masked value is inside.

    Parameters
    ----------
    items : Mapping
        The data items, by name; at least one.
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the axes on the sky (see Coords); their
        defaults are those of make_grid.
    rota : float, optional
        The rotation of the grid on the sky (see Coords).
    spectral : SpectralAxis, optional
        The spectral axis of the kinds of data with one (see
        make_cube_grid): its world coordinates, the convention of its
        velocities, its rest and its medium; the data give its size.
    footprint : ndarray or Deferred, optional
        Where the instrument looked (0 outside); by default, everywhere.
        One read when first used (e.g. of a file opened from its header)
        masks the items as their values are read.

    Raises
    ------
    ValueError
        If the items are not valid (see Dataset), the spectral axis has
        another number of channels, the grid is not valid, or the
        footprint is not of the shape of the items or of their sky, or is
        not finite.
    """

    config_titles = dict(
        step='Pixel size', rpix='Reference pixel',
        rval='Reference position', rota='Rotation')

    # The axes of the grid (see Dataset), along which are both its
    # windows and its blocks
    block_axes: tuple[str, ...]

    def __init__(
            self,
            items: Mapping[str, Data],
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None,
            spectral: SpectralAxis | None = None,
            footprint: np.ndarray | Deferred | None = None
    ):
        super().__init__(items)
        self._footprint = None if footprint is None \
            else self._checked_footprint(footprint)
        if self._footprint is not None:
            self._items = {
                key: item.masked(self._footprint)
                for key, item in self._items.items()}
        if self.spectral_axis is None:
            self._grid = _grids.dataset_grid(self, step, rpix, rval, rota)
        else:
            self._grid = _grids.cube_grid(
                self, step, rpix, rval, rota, spectral)

    def _checked_footprint(
            self, footprint: np.ndarray | Deferred) -> Deferred:
        """
        Return a footprint as floats, 1 inside (where it is not 0) and 0
        outside, or raise ValueError if it is not of the shape of the items
        or of their sky (see DatasetGrid), or is not finite (one read when
        first used, as it is read).
        """
        shape = self.shape()
        shapes = [shape]
        if self.spectral_axis is not None and len(shape) == 3:
            shapes.append(shape[1:])
        given = np.shape(footprint) if not isinstance(footprint, Deferred) \
            else footprint.shape
        if given not in shapes:
            raise ValueError(
                f"the footprint must be of the shape of the data items"
                f"{' or of their sky' if len(shapes) > 1 else ''}, "
                f"{' or '.join(map(str, shapes))}; it is of {given}")
        if isinstance(footprint, Deferred):
            return Deferred(given, lambda: _inside(footprint.values()))
        return Deferred.of(_inside(np.asarray(footprint)))

    def grid(self) -> Grid:
        """
        Return the grid of the values of its items (e.g. of the pixels of
        an image).

        Returns
        -------
        Grid
            The grid.
        """
        return self._grid

    def footprint(self) -> np.ndarray | None:
        """
        Return where the instrument looked (1 inside, 0 outside), if
        given: of the shape of the items, or of their sky, of their dtype.
        """
        if self._footprint is None:
            return None
        return self._footprint.values().astype(self.dtype())

    def footprint_of(self, key: str) -> np.ndarray | None:
        if self._footprint is None:
            return None
        return np.broadcast_to(self.footprint(), self[key].shape())

    def screen_outliers(self, size: Sequence[float], nsigma: float) -> Self:
        """
        Return a copy of it with the outliers of its data items masked
        (see Dataset.screen_outliers): of the other values of a window
        about each, of the size given along each of its axes (block_axes;
        e.g. pixels along x and y of an image).
        """
        return _outliers.screen_grid(self, size, self.block_axes, nsigma)


def cleaned(
        dataset: Dataset,
        outliers: Mapping[str, Any] | None,
        mask_whole_spectra: bool = False
) -> Dataset:
    """
    Return a dataset with its outliers masked, if given the options size
    and nsigma (see Dataset.screen_outliers), and then every spectrum with
    a masked value masked whole, but for the channels masked in every
    spectrum, if asked (see Dataset.mask_whole_spectra); each logs how many
    values each item loses, as its values are made (see with_values).
    """
    if outliers is not None:
        with config_path('outliers'):
            options = parse_options(
                dict(outliers), required={'size', 'nsigma'})
            dataset = dataset.screen_outliers(
                options['size'], options['nsigma'])
    if mask_whole_spectra:
        dataset = dataset.mask_whole_spectra()
    return dataset


def _inside(footprint: np.ndarray) -> np.ndarray:
    """
    Return a footprint as floats, 1 inside (where it is not 0) and 0
    outside, or raise ValueError if it is not finite.
    """
    if not np.all(np.isfinite(footprint)):
        raise ValueError("the footprint must be finite")
    return (footprint != 0).astype(float)


class DatasetTypedParser(TypedParser):
    """
    The parser of the datasets: the files of several datasets dumped
    together need a prefix each.
    """

    def __init__(self):
        super().__init__(Dataset, family='dataset')

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


def check_files(config: Mapping[str, Any]) -> tuple[ConfigProblem, ...]:
    """
    Return the problems of how the files of a dataset fit together, from
    their headers alone, before their data are read (e.g. as it is edited;
    see Dataset.file_problems): of a dataset on a grid of pixels, that the
    data of its items have the axes of its type and are on one grid, and
    that the errors, the mask and the footprint are on the pixels of the
    data, as they are read. Its relative paths are of the directory of
    the configuration (see paths_relative_to).

    A file that cannot be read, or data whose header cannot be read in
    the orientation of the model, are problems of the files themselves
    (see describe_fits), not of these.

    Parameters
    ----------
    config : Mapping
        The configuration of the dataset (see dataset_parser).

    Returns
    -------
    tuple of ConfigProblem
        Its problems, each at its path in it (and an error of the
        configuration itself, if it has one).
    """
    with collect_problems() as problems:
        cls = dataset_parser.registered_class(config.get('type'))
        info = {key: value for key, value in config.items() if key != 'type'}
        func, _ = constructor_of(cls, info)
        problems += cls.file_problems(parse_options_for_callable(info, func))
    return tuple(problems)
