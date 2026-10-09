from numbers import Real

import numpy as np

from gbkfit.dataset.data import dump_data, load_data
from gbkfit.utils import fitsutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'dump_grid_dataset',
    'flatten_single_item',
    'load_grid_dataset',
    'make_grid',
    'nest_single_item'
]


# The options of the world coordinates of the grid of a dataset
_GRID_OPTIONS = ('step', 'rpix', 'rval', 'rota')


def load_grid_dataset(cls, info, names, prefix=''):
    """
    The options of a dataset of class cls whose data items (names) are on
    a grid (see make_grid): the items, loaded, and the world coordinates
    of their grid. Those not given as options (step, rpix, rval, rota)
    come from the headers of the data files of the items, which must
    agree.
    """
    desc = parseutils.make_typed_desc(cls, 'dataset')
    parseutils.sanitize_dimensional_options(info, dict(
        step=int | float, rpix=int | float, rval=int | float), cls._ndim)
    step, rpix, rval, rota = (info.pop(key, None) for key in _GRID_OPTIONS)
    coords = {}
    for name in names:
        if name in info:
            with parseutils.config_path(name):
                info[name], coords[name] = load_data(
                    info[name], prefix, rpix, rval)
    if coords:
        first = next(iter(coords.values()))
        if any(value != first for value in coords.values()):
            raise ConfigError(
                f"the data files of the items of {desc} have different "
                f"world coordinates: {coords}")
        info.update(
            step=first.step if step is None else step,
            rpix=first.rpix,
            rval=first.rval,
            rota=first.rota if rota is None else rota)
    return parseutils.parse_options_for_callable(info, desc, cls.__init__)


def make_grid(dataset, step, rpix, rval, rota):
    """
    The grid of the items of a dataset, with the given world coordinates
    (see fitsutils.Coords), a value or one per axis, or their defaults:
    step 1, the reference pixel at the centre, reference value 0 and no
    rotation. The dataset declares its spectral axis (_spectral_axis).
    """
    size = dataset.shape()[::-1]
    ndim = len(size)
    if step is None:
        step = 1
    if rpix is None:
        rpix = tuple((np.asarray(size) / 2 - 0.5).tolist())
    if rval is None:
        rval = 0
    if rota is None:
        rota = 0
    step, rpix, rval = (
        (value,) * ndim if isinstance(value, Real) else tuple(value)
        for value in (step, rpix, rval))
    for name, value in dict(step=step, rpix=rpix, rval=rval).items():
        if len(value) != ndim:
            raise RuntimeError(
                f"the data have {ndim} axes, but {name} has {len(value)} "
                f"values")
    if not all(value > 0 for value in step):
        raise RuntimeError(f"step must be positive; it is {step}")
    return fitsutils.Grid(
        tuple(size), fitsutils.Coords(step, rpix, rval, rota),
        dataset._spectral_axis)


def dump_grid_dataset(dataset, prefix='', dump_path=True, overwrite=False):
    """
    The info of a dataset whose data items are on a grid: the world
    coordinates of the grid, and the items, written to FITS files named
    after the prefix and their names (with their world coordinates).
    """
    grid = dataset.grid()
    info = dict(
        type=dataset.type(),
        step=grid.coords.step,
        rpix=grid.coords.rpix,
        rval=grid.coords.rval,
        rota=grid.coords.rota)
    for key, item in dataset.items():
        filenames = dict(
            data=f'{prefix}{key}_d.fits',
            mask=f'{prefix}{key}_m.fits',
            error=f'{prefix}{key}_e.fits')
        info[key] = dump_data(
            item, filenames, grid.coords, grid.spectral_axis, dump_path,
            overwrite)
    return info


def nest_single_item(info, name):
    """
    The info of a dataset of one data item, given flat (the options of its
    item beside those of the dataset), with the options of the item under
    its name.
    """
    item = {k: v for k, v in info.items() if k not in _GRID_OPTIONS}
    return {k: info[k] for k in _GRID_OPTIONS if k in info} | {name: item}


def flatten_single_item(info, name):
    """The inverse of nest_single_item."""
    info = dict(info)
    return info | info.pop(name)
