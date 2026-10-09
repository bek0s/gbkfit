from gbkfit.dataset.data import dump_data, load_data
from gbkfit.utils import fitsutils, gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'dump_grid_dataset',
    'flatten_single_item',
    'item_filenames',
    'load_grid_dataset',
    'make_grid',
    'nest_single_item'
]


# The options of the world coordinates of the grid of a dataset (and of
# a spectral axis, rest)
_GRID_OPTIONS = ('step', 'rpix', 'rval', 'rota')
_SPECTRAL_OPTIONS = ('rest',)


def load_grid_dataset(cls, info, names, prefix=''):
    """
    The options of a dataset of class cls whose data items (names) are on
    a grid (see make_grid): the items, loaded, and the world coordinates
    of their grid. Those not given as options (step, rpix, rval, rota and,
    with a spectral axis, rest) come from the headers of the data files of
    the items, which must agree.
    """
    desc = parseutils.make_typed_desc(cls, 'dataset')
    parseutils.sanitize_dimensional_options(info, dict(
        step=int | float, rpix=int | float, rval=int | float), cls.ndim)
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
                f"the data files of the items of {desc} have different "
                f"world coordinates: {coords}")
        info.update(
            step=first.step if step is None else step,
            rpix=first.rpix,
            rval=first.rval,
            rota=first.rota if rota is None else rota)
        if cls.spectral_axis is not None:
            info.update(rest=first.rest)
    return parseutils.parse_options_for_callable(info, desc, cls.__init__)


def make_grid(dataset, step, rpix, rval, rota, rest=None):
    """
    The grid of the items of a dataset, with the given world coordinates
    or their defaults (see gridutils.make_grid). The dataset declares its
    spectral axis (spectral_axis).
    """
    return gridutils.make_grid(
        dataset.shape()[::-1], step, rpix, rval, rota,
        dataset.spectral_axis, rest)


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
    if grid.coords.rest is not None:
        info.update(rest=str(grid.coords.rest))
    def write(filename, array):
        fitsutils.write_data(
            filename, array, grid.coords, grid.spectral_axis, overwrite)
    for key, item in dataset.items():
        info[key] = dump_data(
            item, item_filenames(prefix, key), write, dump_path)
    return info


def item_filenames(prefix, key):
    """The files of the arrays of a data item (see dump_data)."""
    return dict(
        data=f'{prefix}{key}_d.fits',
        mask=f'{prefix}{key}_m.fits',
        error=f'{prefix}{key}_e.fits')


def nest_single_item(info, name):
    """
    The info of a dataset of one data item, given flat (the options of its
    item beside those of the dataset), with the options of the item under
    its name.
    """
    options = _GRID_OPTIONS + _SPECTRAL_OPTIONS
    item = {k: v for k, v in info.items() if k not in options}
    return {k: info[k] for k in options if k in info} | {name: item}


def flatten_single_item(info, name):
    """The inverse of nest_single_item."""
    info = dict(info)
    return info | info.pop(name)
