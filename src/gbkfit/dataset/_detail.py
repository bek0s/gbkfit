"""
Helpers shared by the datasets.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import astropy.units
import numpy as np
from astropy.table import Table

from gbkfit.instrument import lsf_parser, psf_parser
from gbkfit.utils import (
    fitsutils, gridutils, parseutils, tableutils, unitutils)
from gbkfit.utils.parseutils import ConfigError
from .data import Data, DataColumns, DataFiles, dump_data

if TYPE_CHECKING:
    from .base import Dataset, NoiseCorrelation


_log = logging.getLogger(__name__)

# The parsers of the correlations of the noise of a data item, by their
# options (see Data)
_CORRELATION_PARSERS = dict(
    spatial_correlation=psf_parser, spectral_correlation=lsf_parser)

# The data items of moments (and the options of their files): the central
# moments of the orders 0 to 7 (moment0, the flux; moment1, the mean
# velocity; moment2, the dispersion; moment3 and up, in (km/s)^k), and the
# coefficients h3 and h4 of a Gauss-Hermite series (dimensionless, as pPXF
# names them: not central moments)
_ORDERS = range(8)
_HERMITE_ORDERS = (3, 4)
MOMENTS = tuple(f'moment{order}' for order in _ORDERS)
HERMITE = tuple(f'h{order}' for order in _HERMITE_ORDERS)
MOMENT_ITEMS = MOMENTS + HERMITE


def cleaned(
        dataset: 'Dataset',
        outliers: Mapping[str, Any] | None,
        mask_whole_spectra: bool = False
) -> 'Dataset':
    """
    Return a dataset with its outliers masked, if given the options size
    and nsigma (see Dataset.screen_outliers), and then every spectrum with
    a masked value masked whole, if asked (see
    Dataset.mask_whole_spectra); and log how many values each item loses
    to each.
    """
    if outliers is not None:
        with parseutils.config_path('outliers'):
            options = parseutils.parse_options(
                dict(outliers), required={'size', 'nsigma'})
            dataset = _logged(
                dataset,
                dataset.screen_outliers(options['size'], options['nsigma']),
                "outliers masked in the data")
    if mask_whole_spectra:
        dataset = _logged(
            dataset, dataset.mask_whole_spectra(),
            "values masked with whole spectra")
    return dataset


def _logged(before: 'Dataset', after: 'Dataset', what: str) -> 'Dataset':
    """Return a dataset after a change, and log how many values it masked."""
    counts = {
        key: int(np.count_nonzero(before[key].mask())
                 - np.count_nonzero(after[key].mask()))
        for key in before}
    _log.info(f"{what}: {counts}")
    return after


def load_correlations(info: dict[str, Any]) -> None:
    """
    Load the correlations of the noise of a data item in the options of
    its files (spatial_correlation and spectral_correlation; see
    DataFiles), in place.
    """
    for key, parser in _CORRELATION_PARSERS.items():
        parseutils.load_option_and_update_info(parser, info, key)


def load_moment_correlations(info: dict[str, Any]) -> None:
    """
    Load the correlations of the noise of the moments in the options of
    their files (see MOMENT_ITEMS and load_correlations), in place.
    """
    for name in MOMENT_ITEMS:
        if isinstance(info.get(name), Mapping):
            info[name] = dict(info[name])
            with parseutils.config_path(name):
                load_correlations(info[name])


def moment_options(values: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return the options of the moments given among the arguments of a
    constructor (see MOMENT_ITEMS; e.g. its locals()), by name.
    """
    return {
        name: values[name] for name in MOMENT_ITEMS
        if values[name] is not None}


def split_moments(
        items: Mapping[str, Any]
) -> tuple[dict[int, Any], dict[int, Any]]:
    """
    Return the central moments and the Gauss-Hermite coefficients among
    items by name (see MOMENT_ITEMS), by order, as datasets of moments take
    them.
    """
    moments = {
        int(name.removeprefix('moment')): item
        for name, item in items.items() if name in MOMENTS}
    hermite = {
        int(name.removeprefix('h')): item
        for name, item in items.items() if name in HERMITE}
    return moments, hermite


def load_with_files(
        cls: type['Dataset'],
        info: dict[str, Any],
        ndim: int | None = None
) -> 'Dataset':
    """
    Return the dataset of class cls that its from_files makes with the
    options of its configuration, with its spectral axis, if it has one
    (see spectral_axis). With ndim, the options of the world coordinates
    with one value per axis (on the sky, of a dataset with a spectral axis)
    are made lists (see parseutils.sanitize_dimensional_options).
    """
    if ndim is not None:
        parseutils.sanitize_dimensional_options(
            info, dict(step=float, rpix=float, rval=float), ndim)
    if cls.spectral_axis is not None:
        parseutils.load_option_and_update_info(
            gridutils.spectral_axis_parser, info, 'spectral')
    return cls.from_files(
        **parseutils.parse_options_for_callable(info, cls.from_files))


def moment_items(
        moments: Mapping[int, Data],
        hermite: Mapping[int, Data] | None = None
) -> dict[str, Data]:
    """
    Return the data items of the central moments of the given orders
    (moment0 to moment7) and of the given coefficients of a Gauss-Hermite
    series (h3 and h4; see MOMENT_ITEMS), as the models give them: those of
    the orders of 1 and above in km/s (see _in_kms), and the coefficients
    dimensionless. Raise ConfigError for other orders, or units.
    """
    hermite = hermite or {}
    if invalid := sorted(set(moments) - set(_ORDERS), key=str):
        raise ConfigError(
            f"the orders of the moments are 0 to 7; they include {invalid}")
    if invalid := sorted(set(hermite) - set(_HERMITE_ORDERS), key=str):
        raise ConfigError(
            f"the coefficients of a Gauss-Hermite series are h3 and h4; "
            f"they include the orders {invalid}")
    items = {
        f'moment{order}':
            moments[order] if order == 0 else _in_kms(moments[order], order)
        for order in sorted(moments)}
    return items | {
        f'h{order}': _dimensionless(hermite[order], order)
        for order in sorted(hermite)}


def _in_kms(data: Data, order: int) -> Data:
    """
    Return the data of a moment of an order of 1 and above in km/s: the
    mean velocity (1) and the dispersion (2) in km/s, the central moments
    of an order k above 2 in (km/s)^k; converted from their unit (e.g.
    m/s), if they have one, or taken to be in km/s. Raise ConfigError if
    their unit is not one of velocity (to the power k; a dimensionless one
    is of h_k of a Gauss-Hermite series, an item of its own).
    """
    unit = data.unit()
    if unit is None:
        return data
    power = 1 if order <= 2 else order
    kms = (astropy.units.km / astropy.units.s) ** power
    if not unit.is_equivalent(kms):
        hermite = f" (h{order} of a Gauss-Hermite series is the item " \
            f"h{order})" if order in _HERMITE_ORDERS and unit.is_equivalent(
                astropy.units.dimensionless_unscaled) else ""
        raise ConfigError(
            f"moment {order} is in {unitutils.format_unit(unit)!r}, which is "
            f"not a unit of velocity^{power} (e.g. "
            f"{unitutils.format_unit(kms)!r}){hermite}")
    factor = unit.to(kms)
    if factor == 1:
        return data
    return _scaled(data, factor, kms)


def _dimensionless(data: Data, order: int) -> Data:
    """
    Return the data of the coefficient h_k of a Gauss-Hermite series,
    dimensionless (converted from a dimensionless unit, e.g. percent), or
    taken to be. Raise ConfigError if its unit is not dimensionless.
    """
    unit = data.unit()
    if unit is None:
        return data
    if not unit.is_equivalent(astropy.units.dimensionless_unscaled):
        raise ConfigError(
            f"h{order} is in {unitutils.format_unit(unit)!r}; it is "
            f"dimensionless (a central moment is the item moment{order})")
    factor = unit.to(astropy.units.dimensionless_unscaled)
    if factor == 1:
        return data
    return _scaled(data, factor, astropy.units.dimensionless_unscaled)


def _scaled(data: Data, factor: float, unit: astropy.units.UnitBase) -> Data:
    """Return data times a factor, in the given unit."""
    error = data.error()
    return Data(
        data.data() * factor, data.mask(),
        None if error is None else error * factor, unit, data.beam(),
        data.spatial_correlation(), data.spectral_correlation())


def table_item(table: Table, columns: DataColumns) -> Data:
    """
    Return a data item of the columns of a table: its values, errors and
    mask, in its unit (by default that of its column).
    """
    with parseutils.config_path('column'):
        values = tableutils.read_column(table, columns.column, columns.unit)
    unit = None if values.unit is astropy.units.dimensionless_unscaled \
        else values.unit
    error = columns.error
    if isinstance(error, bool):
        raise ConfigError(
            f"the error must be a column, or one error for all the values; "
            f"it is {error}")
    if isinstance(error, int | float):
        if not error > 0:
            raise ConfigError(
                f"one error for all the values must be positive; it is "
                f"{error}")
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
    if columns.mask is not None:
        with parseutils.config_path('mask'):
            mask = np.nan_to_num(
                tableutils.read_column(table, columns.mask, None).value)
    return Data(values.value, mask, error, unit)


def moment_orders(dataset: 'Dataset') -> tuple[int, ...]:
    """
    Return the orders of the moments of a dataset, central moments and
    coefficients of a Gauss-Hermite series (see moment_items).
    """
    return tuple(sorted(
        int(key.removeprefix('moment').removeprefix('h')) for key in dataset))


def read_moments(
        files: Mapping[str, DataFiles],
        rpix: float | Sequence[float] | None = None,
        rval: float | Sequence[float] | None = None
) -> tuple[dict[str, Data], gridutils.Coords]:
    """
    Read the moments of the given names from their files (see
    MOMENT_ITEMS and Data.from_files), and return them with the world
    coordinates of their files, which must agree.
    """
    items, coords = {}, {}
    for name, item in files.items():
        items[name], coords[name] = Data.from_files(
            item.data, item.mask, item.error, rpix, rval, unit=item.unit,
            beam=item.beam, spatial_correlation=item.spatial_correlation,
            spectral_correlation=item.spectral_correlation,
            error_kind=item.error_kind)
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
    axis (None where not given), and the rest, the medium and the
    convention of the spectral axis (velocity: how an ambiguous header is
    read, see fitsutils.read_data).

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
        rest=spectral.rest(), medium=spectral.medium(),
        velocity=spectral.convention())


def cube_coords(
        coords: gridutils.Coords,
        step: float | Sequence[float] | None,
        rota: float | None,
        spectral: gridutils.SpectralAxis | None
) -> dict[str, Any]:
    """
    Return the world coordinates of the grid of a dataset of axes on the
    sky and a spectral axis after them (see cube_grid), from those of its
    files, with the steps, rota, and the convention and the frame of its
    velocities, if given, instead (see file_references for the others).
    A convention or a frame given for files that state another (e.g.
    optical for VRAD, radio velocities) relabels their velocities,
    without converting them: for a header that states the wrong one; a
    warning says so (see parseutils.warn), as a convention copied by
    mistake puts the lines off (by 3.4% at 5000 km/s), and a frame the
    velocities (by up to some 30 km/s).
    """
    spectral = spectral or gridutils.SpectralAxis()
    given, stated = spectral.convention(), coords.convention
    if given is not None and stated is not None and given != stated:
        parseutils.warn(
            f"the spectral axis of the data states velocities of the "
            f"convention {stated!r}; they are taken as given, of the "
            f"convention {given!r}, without converting them")
    frame = spectral.frame()
    if frame is not None and coords.frame is not None \
            and frame != coords.frame:
        parseutils.warn(
            f"the spectral axis of the data states velocities in the frame "
            f"{coords.frame!r}; they are taken as given, in the frame "
            f"{frame!r}, without converting them")
    nspatial = len(coords.step) - 1
    return dict(
        step=coords.step[:nspatial] if step is None else step,
        rpix=coords.rpix[:nspatial],
        rval=coords.rval[:nspatial],
        rota=coords.rota if rota is None else rota,
        spectral=gridutils.SpectralAxis(
            step=spectral.step() or coords.step[nspatial],
            rpix=coords.rpix[nspatial], rval=coords.rval[nspatial],
            convention=given or stated,
            rest=coords.rest, medium=coords.medium,
            frame=frame or coords.frame))


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
