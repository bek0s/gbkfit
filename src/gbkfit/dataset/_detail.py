"""
Helpers shared by the datasets.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import astropy.units
import numpy as np

from gbkfit.utils import fitsutils, gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from .data import Data, FitsFile, dump_data, fits_file

if TYPE_CHECKING:
    from .base import Dataset


# The options of the files of a data item
_ITEM_FILES = ('data', 'mask', 'error')

# The orders of the moments that datasets can hold
_ORDERS = range(8)


def item_files(info: Any, prefix: str) -> dict[str, Any]:
    """
    Return the files of a data item from its configuration (the file of
    its data, and optionally those of its mask and its error, or one
    error), as the arguments data, mask and error of Data.from_files, with
    the prefix prepended to the filenames.
    """
    if not isinstance(info, Mapping):
        raise ConfigError(
            f"a data item has the file of its data ('data'), and optionally "
            f"those of its mask ('mask') and error ('error'); it is {info!r}")
    info = parseutils.parse_options(
        info, required={'data'}, optional={'mask', 'error'})
    files = {}
    for key, value in info.items():
        if value is None:
            continue
        if key == 'error' and isinstance(value, (int, float)):
            files[key] = value
        else:
            with parseutils.config_path(key):
                files[key] = fits_file(value, prefix)
    return files


def pop_item_files(info: dict[str, Any], prefix: str) -> dict[str, Any]:
    """
    Remove the files of the one data item of a dataset from its
    configuration, where they are given beside its own options, and
    return them (see item_files).
    """
    return item_files(
        {key: info.pop(key) for key in _ITEM_FILES if key in info}, prefix)


def pop_moment_files(info: dict[str, Any], prefix: str) -> dict[str, Any]:
    """
    Remove the data items moment0 to moment7 from the configuration of a
    dataset, and return their files as the arguments moments, masks and
    errors of its from_files (see item_files).
    """
    moments, masks, errors = {}, {}, {}
    for order in _ORDERS:
        name = f'moment{order}'
        if info.get(name) is None:
            info.pop(name, None)
            continue
        with parseutils.config_path(name):
            files = item_files(info.pop(name), prefix)
        moments[order] = files['data']
        if 'mask' in files:
            masks[order] = files['mask']
        if 'error' in files:
            errors[order] = files['error']
    return dict(moments=moments, masks=masks, errors=errors)


def load_with_files(
        cls: type['Dataset'],
        info: dict[str, Any],
        ndim: int | None = None,
        **files: Any
) -> 'Dataset':
    """
    Return the dataset of class cls that its from_files makes with the
    given files and the other options of its configuration, which are
    checked against the parameters of from_files. With ndim, the options
    of the world coordinates with one value per axis are made lists (see
    parseutils.sanitize_dimensional_options).
    """
    if ndim is not None:
        parseutils.sanitize_dimensional_options(
            info, dict(step=float, rpix=float, rval=float), ndim)
    options = parseutils.parse_options_for_callable(
        info, cls.from_files, ignore_params=list(files))
    return cls.from_files(**files, **options)


def moment_items(moments: Mapping[int, Data]) -> dict[str, Data]:
    """
    Return the data items of the moments of the given orders: moment0 to
    moment7. Raise ConfigError for other orders.
    """
    if invalid := sorted(set(moments) - set(_ORDERS), key=str):
        raise ConfigError(
            f"the orders of the moments are 0 to 7; they include {invalid}")
    return {f'moment{order}': moments[order] for order in sorted(moments)}


def moment_orders(dataset: 'Dataset') -> tuple[int, ...]:
    """Return the orders of the moments of a dataset (see moment_items)."""
    return tuple(int(key.removeprefix('moment')) for key in dataset)


def read_moments(
        moments: Mapping[int, FitsFile],
        masks: Mapping[int, FitsFile] | None = None,
        errors: Mapping[int, FitsFile | float] | None = None,
        rpix: float | Sequence[float] | None = None,
        rval: float | Sequence[float] | None = None
) -> tuple[dict[int, Data], gridutils.Coords]:
    """
    Read the moments of the given orders from their files (see
    Data.from_files), and return them with the world coordinates of their
    files, which must agree.
    """
    masks = masks or {}
    errors = errors or {}
    if extra := sorted((set(masks) | set(errors)) - set(moments)):
        raise ConfigError(
            f"the moments {extra} have masks or errors but no data")
    items, coords = {}, {}
    for order, data in moments.items():
        items[order], coords[order] = Data.from_files(
            data, masks.get(order), errors.get(order), rpix, rval)
    if not coords:
        raise ConfigError("a dataset needs at least one moment")
    first = next(iter(coords.values()))
    if any(value != first for value in coords.values()):
        raise ConfigError(
            f"the files of the moments have different world coordinates: "
            f"{coords}")
    return items, first


def grid_coords(
        coords: gridutils.Coords,
        step: float | Sequence[float] | None,
        rota: float | None,
        spectral: bool = False
) -> dict[str, Any]:
    """
    Return the world coordinates of the grid of a dataset from those of
    its files, with step and rota, if given, instead (rpix and rval are
    given to the reading of the files). A spectral grid has the rest of
    its files too.
    """
    result = dict(
        step=coords.step if step is None else step,
        rpix=coords.rpix,
        rval=coords.rval,
        rota=coords.rota if rota is None else rota)
    if spectral:
        result.update(rest=coords.rest)
    return result


def make_grid(
        dataset: 'Dataset',
        step: float | Sequence[float] | None,
        rpix: float | Sequence[float] | None,
        rval: float | Sequence[float] | None,
        rota: float | None,
        rest: str | astropy.units.Quantity | None = None
) -> gridutils.Grid:
    """
    Return the grid of the items of a dataset, with the given world
    coordinates or their defaults (see gridutils.make_grid). The dataset
    declares its spectral axis (spectral_axis).
    """
    return gridutils.make_grid(
        dataset.shape()[::-1], step, rpix, rval, rota,
        dataset.spectral_axis, rest)


def dump_grid_dataset(
        dataset: 'Dataset',
        prefix: str = '',
        dump_path: bool = True,
        overwrite: bool = False
) -> dict[str, Any]:
    """
    Return the configuration of a dataset whose data items are on a grid:
    the world coordinates of the grid, and the items, written to FITS
    files named after the prefix and their names (with their world
    coordinates).
    """
    grid = dataset.grid()
    info = dict(
        type=dataset.type(),
        step=grid.coords.step,
        rpix=grid.coords.rpix,
        rval=grid.coords.rval,
        rota=grid.coords.rota)
    if grid.coords.rest is not None:
        info.update(rest=str(grid.coords.rest))

    def write(filename: str, array: np.ndarray) -> None:
        fitsutils.write_data(
            filename, array, grid.coords, grid.spectral_axis, overwrite)
    for key, item in dataset.items():
        info[key] = dump_data(
            item, item_filenames(prefix, key), write, dump_path)
    return info


def item_filenames(prefix: str, key: str) -> dict[str, str]:
    """Return the files of the arrays of a data item (see dump_data)."""
    return dict(
        data=f'{prefix}{key}_d.fits',
        mask=f'{prefix}{key}_m.fits',
        error=f'{prefix}{key}_e.fits')


def flatten_single_item(
        info: dict[str, Any], name: str
) -> dict[str, Any]:
    """
    Return the configuration of a dataset of one data item with the
    options of the item beside its own, as it is given.
    """
    info = dict(info)
    return info | info.pop(name)
