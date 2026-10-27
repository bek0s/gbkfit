"""
Helpers shared by the datasets.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import astropy.units
import numpy as np
from astropy.table import Table

from gbkfit.instrument import PSF, lsf_parser, psf_parser
from gbkfit.utils import (
    fitsutils, gridutils, parseutils, tableutils, unitutils)
from gbkfit.utils.parseutils import ConfigError
from .data import Data, FitsFile, dump_data, fits_file

if TYPE_CHECKING:
    from .base import Dataset, NoiseCorrelation


# The options of the files of a data item
_ITEM_OPTIONS = ('data', 'mask', 'error', 'unit', 'beam')

# The options of a data item of the columns of a table (see table_item)
_TABLE_ITEM_OPTIONS = ('column', 'error', 'mask', 'unit')

# The parsers of the correlations of the noise of a data item, by kind
# (see Data)
_CORRELATION_PARSERS = dict(spatial=psf_parser, spectral=lsf_parser)

# The orders of the moments that datasets can hold
_ORDERS = range(8)


def item_options(
        info: Any,
        prefix: str,
        correlations: Sequence[str] = ()
) -> dict[str, Any]:
    """
    Return the options of a data item from its configuration (the file of
    its data, and optionally those of its mask and its error, or one
    error, its unit, its beam, and the correlations of its noise of the
    given kinds, if its dataset takes them: see Dataset.noise_correlations),
    as the arguments data, mask, error, unit, beam, spatial_correlation and
    spectral_correlation of Data.from_files, with the prefix prepended to
    the filenames of the arrays.
    """
    names = [f'{kind}_correlation' for kind in correlations]
    if not isinstance(info, Mapping):
        raise ConfigError(
            f"a data item has the file of its data ('data'), and optionally "
            f"those of its mask ('mask') and error ('error'), its unit "
            f"('unit'), beam ('beam') and the correlations of its noise "
            f"{names}; it is {info!r}")
    info = parseutils.parse_options(
        info, required={'data'},
        optional={'mask', 'error', 'unit', 'beam', *names})
    options = {}
    for key, value in info.items():
        if value is None:
            continue
        if key in ('unit', 'beam') \
                or (key == 'error' and isinstance(value, (int, float))):
            options[key] = value
        elif key in names:
            kind = key.removesuffix('_correlation')
            with parseutils.config_path(key):
                options[key] = _CORRELATION_PARSERS[kind].load(value)
        else:
            with parseutils.config_path(key):
                options[key] = fits_file(value, prefix)
    return options


def pop_item_options(
        info: dict[str, Any],
        prefix: str,
        correlations: Sequence[str] = ()
) -> dict[str, Any]:
    """
    Remove the options of the one data item of a dataset from its
    configuration, where they are given beside its own options, and
    return them (see item_options; with the correlations of the noise of
    the given kinds).
    """
    keys = (*_ITEM_OPTIONS, *(f'{kind}_correlation' for kind in correlations))
    return item_options(
        {key: info.pop(key) for key in keys if key in info}, prefix,
        correlations)


def pop_moment_options(
        info: dict[str, Any],
        prefix: str,
        correlations: Sequence[str] = ()
) -> dict[str, Any]:
    """
    Remove the data items moment0 to moment7 from the configuration of a
    dataset, and return their options as the arguments moments, masks,
    errors, units and beams of its from_files (see item_options), and
    spatial_correlations if the noise of moments can be correlated on the
    sky (correlations has 'spatial').
    """
    moments, masks, errors, units, beams = {}, {}, {}, {}, {}
    spatial = {}
    for order in _ORDERS:
        name = f'moment{order}'
        if info.get(name) is None:
            info.pop(name, None)
            continue
        with parseutils.config_path(name):
            options = item_options(info.pop(name), prefix, correlations)
        moments[order] = options['data']
        for key, values in (
                ('mask', masks), ('error', errors), ('unit', units),
                ('beam', beams), ('spatial_correlation', spatial)):
            if key in options:
                values[order] = options[key]
    correlated = dict(spatial_correlations=spatial) \
        if 'spatial' in correlations else {}
    return dict(
        moments=moments, masks=masks, errors=errors, units=units,
        beams=beams) | correlated


def read_table_moments(info: dict[str, Any], prefix: str) -> dict[int, Data]:
    """
    Return the moments of a dataset of the columns of a table, from the
    options of its configuration: the file of the table ('file'; any that
    astropy reads, see tableutils.read_table; the prefix is prepended to
    its filename), its format ('format', if astropy cannot tell it), and
    the data items moment0 to moment7, each of its columns (see
    table_item). Raise ConfigError for other options, or without moments.
    """
    names = {f'moment{order}': order for order in _ORDERS}
    options = parseutils.parse_options(
        info, required={'file'}, optional={'format', *names})
    with parseutils.config_path('file'):
        table = tableutils.read_table(
            options['file'], options.get('format'), prefix)
    moments = {}
    for name, order in names.items():
        if options.get(name) is not None:
            with parseutils.config_path(name):
                moments[order] = table_item(table, options[name])
    if not moments:
        raise ConfigError("a dataset needs at least one moment")
    return moments


def load_with_files(
        cls: type['Dataset'],
        info: dict[str, Any],
        ndim: int | None = None,
        **files: Any
) -> 'Dataset':
    """
    Return the dataset of class cls that its from_files makes with the
    given files and the other options of its configuration, which are
    checked against the parameters of from_files, and its spectral axis,
    if it has one (see spectral_axis). With ndim, the options of the world
    coordinates with one value per axis (on the sky, of a dataset with a
    spectral axis) are made lists (see
    parseutils.sanitize_dimensional_options).
    """
    if ndim is not None:
        parseutils.sanitize_dimensional_options(
            info, dict(step=float, rpix=float, rval=float), ndim)
    if cls.spectral_axis is not None:
        parseutils.load_option_and_update_info(
            gridutils.spectral_axis_parser, info, 'spectral')
    options = parseutils.parse_options_for_callable(
        info, cls.from_files, ignore_params=list(files))
    return cls.from_files(**files, **options)


def moment_items(moments: Mapping[int, Data]) -> dict[str, Data]:
    """
    Return the data items of the moments of the given orders: moment0 to
    moment7, those of the orders of 1 and above in km/s, as the models
    give them, or above 2 the dimensionless coefficients of a
    Gauss-Hermite series (see _in_kms). Raise ConfigError for other
    orders.
    """
    if invalid := sorted(set(moments) - set(_ORDERS), key=str):
        raise ConfigError(
            f"the orders of the moments are 0 to 7; they include {invalid}")
    return {
        f'moment{order}':
            moments[order] if order == 0 else _in_kms(moments[order], order)
        for order in sorted(moments)}


def _in_kms(data: Data, order: int) -> Data:
    """
    Return the data of a moment of an order of 1 and above in km/s: the
    mean velocity (1) and the dispersion (2) in km/s, the central moments
    of an order k above 2 in (km/s)^k; converted from their unit (e.g.
    m/s), if they have one, or taken to be in km/s. A moment of an order k
    above 2 can be the coefficient h_k of a Gauss-Hermite series instead,
    dimensionless (e.g. of the method gauss_hermite of the observables of
    moments, which say which they take). Raise ConfigError if their unit
    is not one of velocity (to the power k), nor dimensionless for an
    order above 2.
    """
    unit = data.unit()
    if unit is None:
        return data
    power = 1 if order <= 2 else order
    kms = (astropy.units.km / astropy.units.s) ** power
    if power > 1 and unit.is_equivalent(astropy.units.dimensionless_unscaled):
        target = astropy.units.dimensionless_unscaled
    elif unit.is_equivalent(kms):
        target = kms
    else:
        kind = f"a unit of velocity (e.g. {unitutils.format_unit(kms)!r})"
        if power > 1:
            kind = (
                f"a unit of velocity^{power} (e.g. "
                f"{unitutils.format_unit(kms)!r}), nor dimensionless (h{order} "
                f"of a Gauss-Hermite series)")
        raise ConfigError(
            f"moment {order} is in {unitutils.format_unit(unit)!r}, which is "
            f"not {kind}")
    factor = unit.to(target)
    if factor == 1:
        return data
    error = data.error()
    return Data(
        data.data() * factor, data.mask(),
        None if error is None else error * factor, target, data.beam(),
        data.spatial_correlation(), data.spectral_correlation())


def table_item(table: Table, info: Any) -> Data:
    """
    Return a data item of the columns of a table from its configuration:
    its column ('column'), and optionally its error ('error': a column, or
    one error for all its values), mask ('mask': a column, 0 for the values
    to leave out) and unit ('unit'; by default that of its column); its
    values, errors and mask, in its unit.
    """
    if not isinstance(info, Mapping):
        raise ConfigError(
            f"a data item has its column ('column'), and optionally its "
            f"error ('error': a column, or one error), mask ('mask': a "
            f"column) and unit ('unit'); it is {info!r}")
    info = parseutils.parse_options(
        info, required={'column'},
        optional=set(_TABLE_ITEM_OPTIONS) - {'column'})
    values = tableutils.read_column(table, info['column'], info.get('unit'))
    unit = None if values.unit is astropy.units.dimensionless_unscaled \
        else values.unit
    error = info.get('error')
    if isinstance(error, (int, float)):
        error = np.full(values.shape, float(error))
    elif error is not None:
        with parseutils.config_path('error'):
            error = tableutils.read_column(table, error, None)
            if error.unit is not astropy.units.dimensionless_unscaled:
                try:
                    error = error.to(values.unit)
                except astropy.units.UnitConversionError:
                    raise ConfigError(
                        f"the errors are in {error.unit}, but the values in "
                        f"{values.unit}") from None
            error = error.value
    mask = None
    if info.get('mask') is not None:
        with parseutils.config_path('mask'):
            mask = np.nan_to_num(
                tableutils.read_column(table, info['mask'], None).value)
    return Data(values.value, mask, error, unit)


def moment_orders(dataset: 'Dataset') -> tuple[int, ...]:
    """Return the orders of the moments of a dataset (see moment_items)."""
    return tuple(int(key.removeprefix('moment')) for key in dataset)


def read_moments(
        moments: Mapping[int, FitsFile],
        masks: Mapping[int, FitsFile] | None = None,
        errors: Mapping[int, FitsFile | float] | None = None,
        units: Mapping[int, str] | None = None,
        beams: Mapping[int, Mapping[str, Any]] | None = None,
        rpix: float | Sequence[float] | None = None,
        rval: float | Sequence[float] | None = None,
        spatial_correlations: Mapping[int, PSF] | None = None
) -> tuple[dict[int, Data], gridutils.Coords]:
    """
    Read the moments of the given orders from their files (see
    Data.from_files), and return them with the world coordinates of their
    files, which must agree.
    """
    masks = masks or {}
    errors = errors or {}
    units = units or {}
    beams = beams or {}
    spatial_correlations = spatial_correlations or {}
    if extra := sorted(
            (set(masks) | set(errors) | set(units) | set(beams)
             | set(spatial_correlations)) - set(moments)):
        raise ConfigError(
            f"the moments {extra} have masks, errors, units, beams or "
            f"correlations of their noise but no data")
    items, coords = {}, {}
    for order, data in moments.items():
        items[order], coords[order] = Data.from_files(
            data, masks.get(order), errors.get(order), rpix, rval,
            unit=units.get(order), beam=beams.get(order),
            spatial_correlation=spatial_correlations.get(order))
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
        rota: float | None
) -> dict[str, Any]:
    """
    Return the world coordinates of the grid of a dataset from those of
    its files, with step and rota, if given, instead (rpix and rval are
    given to the reading of the files).
    """
    return dict(
        step=coords.step if step is None else step,
        rpix=coords.rpix,
        rval=coords.rval,
        rota=coords.rota if rota is None else rota)


def file_references(
        nspatial: int,
        rpix: float | Sequence[float] | None,
        rval: float | Sequence[float] | None,
        spectral: gridutils.SpectralAxis | None
) -> dict[str, Any]:
    """
    Return the options of the reading of the files of a dataset of
    nspatial axes on the sky and a spectral axis after them (see
    Data.from_files): the reference pixels and values given, for each
    axis (None where not given), and the rest and the medium of the
    spectral axis.

    Raises
    ------
    ConfigError
        If the spectral axis has a size: its data give it.
    """
    spectral = spectral or gridutils.SpectralAxis()
    if spectral.size() is not None:
        raise ConfigError(
            "the data give the number of channels of their spectral axis; "
            "remove its size")

    def per_axis(spatial, value):
        values = [None] * nspatial if spatial is None \
            else np.broadcast_to(np.asarray(spatial, float), nspatial)
        values = [*values, value]
        return None if all(v is None for v in values) else values
    return dict(
        rpix=per_axis(rpix, spectral.rpix()),
        rval=per_axis(rval, spectral.rval()),
        rest=spectral.rest(), medium=spectral.medium())


def cube_coords(
        coords: gridutils.Coords,
        step: float | Sequence[float] | None,
        rota: float | None,
        spectral: gridutils.SpectralAxis | None
) -> dict[str, Any]:
    """
    Return the world coordinates of the grid of a dataset of axes on the
    sky and a spectral axis after them (see cube_grid), from those of its
    files, with the steps, rota and the convention, if given, instead (see
    file_references for the others).
    """
    spectral = spectral or gridutils.SpectralAxis()
    nspatial = len(coords.step) - 1
    return dict(
        step=coords.step[:nspatial] if step is None else step,
        rpix=coords.rpix[:nspatial],
        rval=coords.rval[:nspatial],
        rota=coords.rota if rota is None else rota,
        spectral=gridutils.SpectralAxis(
            step=spectral.step() or coords.step[nspatial],
            rpix=coords.rpix[nspatial], rval=coords.rval[nspatial],
            convention=spectral.convention() or coords.convention,
            rest=coords.rest, medium=coords.medium))


def make_grid(
        dataset: 'Dataset',
        step: float | Sequence[float] | None,
        rpix: float | Sequence[float] | None,
        rval: float | Sequence[float] | None,
        rota: float | None
) -> gridutils.Grid:
    """
    Return the grid of the items of a dataset, with the given world
    coordinates or their defaults (see gridutils.make_grid).
    """
    return gridutils.make_grid(dataset.shape()[::-1], step, rpix, rval, rota)


def cube_grid(
        dataset: 'Dataset',
        step: float | Sequence[float] | None,
        rpix: float | Sequence[float] | None,
        rval: float | Sequence[float] | None,
        rota: float | None,
        spectral: gridutils.SpectralAxis | None
) -> gridutils.Grid:
    """
    Return the grid of the items of a dataset of axes on the sky and a
    spectral axis after them, its last, with the given world coordinates
    or their defaults (see gridutils.make_cube_grid). The data give the
    number of channels.

    Raises
    ------
    ConfigError
        If the spectral axis has another number of channels, or the grid is
        not valid.
    """
    shape = dataset.shape()[::-1]
    spectral = spectral or gridutils.SpectralAxis()
    if spectral.size() not in (None, shape[-1]):
        raise ConfigError(
            f"the data have {shape[-1]} channels; the spectral axis has "
            f"{spectral.size()}")
    return gridutils.make_cube_grid(
        shape[:-1], step, rpix, rval, rota,
        spectral.replaced(gridutils.SpectralAxis(size=shape[-1])))


def dump_grid_dataset(
        dataset: 'Dataset',
        prefix: str = '',
        dump_path: bool = True,
        overwrite: bool = False
) -> dict[str, Any]:
    """
    Return the configuration of a dataset whose data items are on a grid:
    the world coordinates of the grid (and of its spectral axis, if any,
    but its convention, which the headers of the files give), and the
    items, written to FITS files named after the prefix and their names
    (with their world coordinates).
    """
    grid = dataset.grid()
    coords = grid.coords
    axes = slice(None) if grid.spectral_axis is None \
        else slice(grid.spectral_axis)
    info = dict(
        type=dataset.type(),
        step=coords.step[axes],
        rpix=coords.rpix[axes],
        rval=coords.rval[axes],
        rota=coords.rota)
    if grid.spectral_axis is not None:
        spectral = gridutils.SpectralAxis.of(grid).dump()
        info.update(spectral={
            key: value for key, value in spectral.items()
            if key not in ('size', 'convention')})

    def write(filename: str, array: np.ndarray) -> None:
        fitsutils.write_data(
            filename, array, grid.coords, grid.spectral_axis, overwrite)
    for key, item in dataset.items():
        info[key] = dump_data(
            item, item_filenames(prefix, key), write, dump_path,
            f'{prefix}{key}_', overwrite)
    return info


def grid_noise_correlation(
        dataset: 'Dataset', key: str
) -> 'NoiseCorrelation | None':
    """
    Return the correlation of the noise of a data item of a dataset on a
    grid (see Dataset.noise_correlation): the shapes of its correlations
    on the pixels and the channels of the grid, out to the largest
    separations of the data, divided by their values at their centres.
    Raise ConfigError if it has a spectral correlation but the grid no
    spectral axis, or a shape is not largest at its centre.
    """
    from .base import NoiseCorrelation
    item = dataset[key]
    spatial, spectral = item.spatial_correlation(), item.spectral_correlation()
    if spatial is None and spectral is None:
        return None
    grid = dataset.grid()
    if spectral is not None and grid.spectral_axis is None:
        raise ConfigError(
            f"the noise of '{key}' has a spectral correlation, but the "
            f"data have no spectral axis")
    if spatial is not None:
        nx, ny = grid.size[:2]
        spatial = _centred(spatial.asarray(
            grid.coords.step[:2], (2 * nx - 1, 2 * ny - 1),
            rota=grid.coords.rota), key)
    if spectral is not None:
        axis = grid.spectral_axis
        spectral = _centred(spectral.asarray(
            abs(grid.coords.step[axis]), 2 * grid.size[axis] - 1), key)
    return NoiseCorrelation(spatial, spectral)


def _centred(shape: np.ndarray, key: str) -> np.ndarray:
    """
    Return the shape of a correlation of the noise of data item key (of
    odd sizes) divided by its value at its centre, or raise ConfigError
    unless that is its largest (the variance of the noise).
    """
    centre = shape[tuple(n // 2 for n in shape.shape)]
    if not centre > 0 or np.any(shape > centre):
        raise ConfigError(
            f"the correlation of the noise of '{key}' must be largest at "
            f"its centre (at a separation of 0)")
    return shape / centre


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
