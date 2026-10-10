"""
Helpers shared by the datasets.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import astropy.units
import numpy as np

from gbkfit.utils import fitsutils, gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from .data import dump_data, load_data

if TYPE_CHECKING:
    from .base import Dataset


# The options of the world coordinates of the grid of a dataset (and of
# a spectral axis, rest)
_GRID_OPTIONS = ('step', 'rpix', 'rval', 'rota')
_SPECTRAL_OPTIONS = ('rest',)


def load_grid_dataset(
        cls: type['Dataset'],
        info: dict[str, Any],
        names: Sequence[str],
        prefix: str = ''
) -> dict[str, Any]:
    """
    Return the arguments of a dataset of class cls whose data items (of
    the given names) are on a grid (see make_grid): the items, loaded, and
    the world coordinates of their grid. Those not given as options (step,
    rpix, rval, rota and, with a spectral axis, rest) come from the
    headers of the data files of the items, which must agree.
    """
    parseutils.sanitize_dimensional_options(info, dict(
        step=float, rpix=float, rval=float), cls.ndim)
    step, rpix, rval, rota = (info.pop(key, None) for key in _GRID_OPTIONS)
    rest = info.pop('rest', None) if cls.spectral_axis is not None else None
    coords = {}
    for name in names:
        # (an item that is null is absent)
        if info.get(name) is not None:
            with parseutils.config_path(name):
                info[name], coords[name] = load_data(
                    info[name], prefix, rpix, rval, rest, cls.spectral_axis)
    if coords:
        first = next(iter(coords.values()))
        if any(value != first for value in coords.values()):
            raise ConfigError(
                f"the data files of the items have different world "
                f"coordinates: {coords}")
        info.update(
            step=first.step if step is None else step,
            rpix=first.rpix,
            rval=first.rval,
            rota=first.rota if rota is None else rota)
        if cls.spectral_axis is not None:
            info.update(rest=first.rest)
    return parseutils.parse_options_for_callable(info, cls.__init__)


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


def nest_single_item(info: dict[str, Any], name: str) -> dict[str, Any]:
    """
    Return the configuration of a dataset of one data item, given flat
    (the options of its item beside those of the dataset), with the
    options of the item under its name.
    """
    options = _GRID_OPTIONS + _SPECTRAL_OPTIONS
    item = {k: v for k, v in info.items() if k not in options}
    return {k: info[k] for k in options if k in info} | {name: item}


def flatten_single_item(
        info: dict[str, Any], name: str
) -> dict[str, Any]:
    """Return the inverse of nest_single_item."""
    info = dict(info)
    return info | info.pop(name)
