"""
Helpers shared by the datasets.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import astropy.units
import numpy as np
from astropy.table import Table

import gbkfit.units
from gbkfit.config import config_path, parse_options, warn
from gbkfit.grid import (
    Coords, Grid, SpectralAxis, make_cube_grid, make_grid, per_axis,
    pixels_on, rest_of_convention)
from gbkfit.instrument import Channel
from gbkfit.io import read_column, write_data
from .data import Data, DataColumns, DataFiles, dump_data

if TYPE_CHECKING:
    from .base import Dataset, NoiseCorrelation


_log = logging.getLogger(__name__)

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

# How far (in pixels) the world coordinates of the maps of moments may put
# their pixels from those of the first (as those of the errors and masks of
# data, see read_arrays)
_GRID_TOLERANCE = 1e-3


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
        with config_path('outliers'):
            options = parse_options(
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


def moment_items(
        moments: Mapping[int, Data],
        hermite: Mapping[int, Data] | None = None
) -> dict[str, Data]:
    """
    Return the data items of the central moments of the given orders
    (moment0 to moment7) and of the given coefficients of a Gauss-Hermite
    series (h3 and h4; see MOMENT_ITEMS), as the models give them: those of
    the orders of 1 and above in km/s (see _in_kms), and the coefficients
    dimensionless. Raise ValueError for other orders, or units.
    """
    hermite = hermite or {}
    if invalid := sorted(set(moments) - set(_ORDERS), key=str):
        raise ValueError(
            f"the orders of the moments are 0 to 7; they include {invalid}")
    if invalid := sorted(set(hermite) - set(_HERMITE_ORDERS), key=str):
        raise ValueError(
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
    m/s), if they have one, or taken to be in km/s (their file states
    none, see read_moments). Raise ValueError if their unit is not one of
    velocity (to the power k; a dimensionless one is of h_k of a
    Gauss-Hermite series, an item of its own).
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
        raise ValueError(
            f"moment {order} is in {gbkfit.units.format_unit(unit)!r}, "
            f"which is not a unit of velocity^{power} (e.g. "
            f"{gbkfit.units.format_unit(kms)!r}){hermite}")
    return data if unit.to(kms) == 1 else data.to_unit(kms)


def _dimensionless(data: Data, order: int) -> Data:
    """
    Return the data of the coefficient h_k of a Gauss-Hermite series,
    dimensionless (converted from a dimensionless unit, e.g. percent), or
    taken to be. Raise ValueError if its unit is not dimensionless.
    """
    unit = data.unit()
    if unit is None:
        return data
    if not unit.is_equivalent(astropy.units.dimensionless_unscaled):
        raise ValueError(
            f"h{order} is in {gbkfit.units.format_unit(unit)!r}; it is "
            f"dimensionless (a central moment is the item moment{order})")
    dimensionless = astropy.units.dimensionless_unscaled
    return data if unit.to(dimensionless) == 1 \
        else data.to_unit(dimensionless)


def table_item(table: Table, columns: DataColumns) -> Data:
    """
    Return a data item of the columns of a table: its values, errors and
    mask, in its unit (by default that of its column). Errors without a
    unit are in that of the column of the values (or in the unit given,
    if the column has none).
    """
    with config_path('column'):
        values = read_column(table, columns.column, columns.unit)
    unit = None if values.unit is astropy.units.dimensionless_unscaled \
        else values.unit
    error = columns.error
    if isinstance(error, bool):
        raise ValueError(
            f"the error must be a column, or one error for all the values; "
            f"it is {error}")
    if isinstance(error, int | float):
        if not error > 0:
            raise ValueError(
                f"one error for all the values must be positive; it is "
                f"{error}")
        error = np.full(values.shape, float(error))
    elif error is not None:
        with config_path('error'):
            error = read_column(table, error, None)
            if error.unit is astropy.units.dimensionless_unscaled:
                error = error.value * (
                    table[columns.column].unit or values.unit)
            try:
                error = error.to_value(values.unit)
            except astropy.units.UnitConversionError:
                raise ValueError(
                    f"the errors are in {error.unit}, but the values in "
                    f"{values.unit}") from None
    mask = None
    if columns.mask is not None:
        with config_path('mask'):
            mask = np.nan_to_num(
                read_column(table, columns.mask, None).value)
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
) -> tuple[dict[str, Data], Coords]:
    """
    Read the moments of the given names from their files (see
    MOMENT_ITEMS and Data.from_files), and return them with the world
    coordinates of the file of the first. Maps of moments must be on the
    pixels of the first on the sky, to within _GRID_TOLERANCE (e.g. their
    headers may round the steps differently; what else their world
    coordinates state, e.g. the frame of velocities of a moment 1, is not
    of the grid); moments of regions have no pixels on the sky.
    """
    items, coords = {}, {}
    for name, item in files.items():
        items[name], coords[name] = Data.from_files(
            item.data, item.mask, item.error, rpix, rval, unit=item.unit,
            beam=item.beam, spatial_correlation=item.spatial_correlation,
            spectral_correlation=item.spectral_correlation,
            error_kind=item.error_kind, mask_kind=item.mask_kind,
            footprint=item.footprint)
    if not coords:
        raise ValueError("a dataset needs at least one moment")
    (first, first_coords), *others = coords.items()
    for name, value in others:
        if items[name].ndim() == 2 and not _on_the_same_pixels(
                items[name].shape(), value,
                items[first].shape(), first_coords):
            raise ValueError(
                f"the file of {name} is not on the pixels of that of "
                f"{first}: their world coordinates are {value} and "
                f"{first_coords}")
    return items, first_coords


def _on_the_same_pixels(
        shape: tuple[int, ...],
        coords: Coords,
        other_shape: tuple[int, ...],
        other_coords: Coords
) -> bool:
    """
    Return whether a map of a shape (numpy order) and world coordinates is
    on the pixels of another on the sky (see _GRID_TOLERANCE).
    """
    if shape != other_shape:
        return False
    def sky_grid(c: Coords) -> Grid:
        return make_grid(shape[::-1], c.step, c.rpix, c.rval, c.rota)
    x, y = pixels_on(sky_grid(coords), sky_grid(other_coords))
    j, i = np.indices(shape)
    return bool(
        np.allclose(x, i, rtol=0, atol=_GRID_TOLERANCE)
        and np.allclose(y, j, rtol=0, atol=_GRID_TOLERANCE))


def grid_coords(
        coords: Coords,
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
        spectral: SpectralAxis | None
) -> dict[str, Any]:
    """
    Return the options of the reading of the files of a dataset of
    nspatial axes on the sky and a spectral axis after them (see
    Data.from_files): the reference pixels and values given, for each
    axis (None where not given), and the rest, the medium and the
    convention of the spectral axis (velocity: how an ambiguous header is
    read, see model_header), and its type (spectral_type: how an axis whose
    header gives a type FITS does not define is read, see model_header).

    Raises
    ------
    ValueError
        If the spectral axis has a size: its data give it; or rpix or
        rval has neither one value nor one for each axis on the sky.
    """
    spectral = spectral or SpectralAxis()
    if spectral.size() is not None:
        raise ValueError(
            "the data give the number of channels of their spectral axis; "
            "remove its size")

    def with_spectral(
            option: str,
            spatial: float | Sequence[float] | None,
            value: float | None
    ) -> list[float | None] | None:
        values = [None] * nspatial if spatial is None \
            else per_axis(option, spatial, nspatial)
        values = [*values, value]
        return None if all(v is None for v in values) else values
    return dict(
        rpix=with_spectral('rpix', rpix, spectral.rpix()),
        rval=with_spectral('rval', rval, spectral.rval()),
        rest=spectral.rest(), medium=spectral.medium(),
        velocity=spectral.convention(), spectral_type=spectral.type())


def cube_coords(
        coords: Coords,
        step: float | Sequence[float] | None,
        rota: float | None,
        spectral: SpectralAxis | None
) -> dict[str, Any]:
    """
    Return the world coordinates of the grid of a dataset of axes on the
    sky and a spectral axis after them (see cube_grid), from those of its
    files, with the steps, rota, and the convention and the frame of its
    velocities, if given, instead (see file_references for the others).
    A convention or a frame given for files that state another (e.g.
    optical for VRAD, radio velocities) relabels their velocities,
    without converting them: for a header that states the wrong one; a
    warning says so (see warn), as a convention copied by
    mistake puts the lines off (by 3.4% at 5000 km/s), and a frame the
    velocities (by up to some 30 km/s). The rest of the files is then
    that of the same line for the convention given (see
    rest_of_convention; e.g. the wavelength of a rest frequency).
    """
    spectral = spectral or SpectralAxis()
    given, stated = spectral.convention(), coords.convention
    if given is not None and stated is not None and given != stated:
        warn(
            f"the spectral axis of the data states velocities of the "
            f"convention {stated!r}; they are taken as given, of the "
            f"convention {given!r}, without converting them")
    frame = spectral.frame()
    if frame is not None and coords.frame is not None \
            and frame != coords.frame:
        warn(
            f"the spectral axis of the data states velocities in the frame "
            f"{coords.frame!r}; they are taken as given, in the frame "
            f"{frame!r}, without converting them")
    rest, medium = coords.rest, coords.medium
    if given is not None:
        rest, medium = rest_of_convention(rest, medium, given)
    nspatial = len(coords.step) - 1
    return dict(
        step=coords.step[:nspatial] if step is None else step,
        rpix=coords.rpix[:nspatial],
        rval=coords.rval[:nspatial],
        rota=coords.rota if rota is None else rota,
        spectral=SpectralAxis(
            step=spectral.step() or coords.step[nspatial],
            rpix=coords.rpix[nspatial], rval=coords.rval[nspatial],
            convention=given or stated, rest=rest, medium=medium,
            frame=frame or coords.frame))


def dataset_grid(
        dataset: 'Dataset',
        step: float | Sequence[float] | None,
        rpix: float | Sequence[float] | None,
        rval: float | Sequence[float] | None,
        rota: float | None
) -> Grid:
    """
    Return the grid of the items of a dataset, with the given world
    coordinates or their defaults (see make_grid).
    """
    return make_grid(dataset.shape()[::-1], step, rpix, rval, rota)


def cube_grid(
        dataset: 'Dataset',
        step: float | Sequence[float] | None,
        rpix: float | Sequence[float] | None,
        rval: float | Sequence[float] | None,
        rota: float | None,
        spectral: SpectralAxis | None
) -> Grid:
    """
    Return the grid of the items of a dataset of axes on the sky and a
    spectral axis after them, its last, with the given world coordinates
    or their defaults (see make_cube_grid). The data give the
    number of channels.

    Raises
    ------
    ValueError
        If the spectral axis has another number of channels, or the grid is
        not valid.
    """
    shape = dataset.shape()[::-1]
    spectral = spectral or SpectralAxis()
    if spectral.size() not in (None, shape[-1]):
        raise ValueError(
            f"the data have {shape[-1]} channels; the spectral axis has "
            f"{spectral.size()}")
    return make_cube_grid(
        shape[:-1], step, rpix, rval, rota,
        spectral.replaced(SpectralAxis(size=shape[-1])))


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
        spectral = SpectralAxis.of(grid).dump()
        info.update(spectral={
            key: value for key, value in spectral.items()
            if key not in ('size', 'convention')})

    def write(filename: str, array: np.ndarray) -> None:
        write_data(
            filename, array, grid.coords, grid.spectral_axis,
            overwrite=overwrite)
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
    Raise ValueError if it has a spectral correlation but the grid no
    spectral axis, or a shape is not largest at its centre.
    """
    from .base import NoiseCorrelation
    item = dataset[key]
    spatial, spectral = item.spatial_correlation(), item.spectral_correlation()
    if spatial is None and spectral is None:
        return None
    grid = dataset.grid()
    if spectral is not None and grid.spectral_axis is None:
        raise ValueError(
            f"the noise of '{key}' has a spectral correlation, but the "
            f"data have no spectral axis")
    if spatial is not None:
        nx, ny = grid.size[:2]
        spatial = _centred(spatial.asarray(
            grid.coords.step[:2], (2 * nx - 1, 2 * ny - 1),
            rota=grid.coords.rota), key)
    if spectral is not None:
        # (drawn at the middle channel, of the widths of the light there)
        channel = Channel.middle(grid)
        spectral = _centred(spectral.on(channel).asarray(
            channel.width, 2 * grid.size[grid.spectral_axis] - 1), key)
    return NoiseCorrelation(spatial, spectral)


def _centred(shape: np.ndarray, key: str) -> np.ndarray:
    """
    Return the shape of a correlation of the noise of data item key (of
    odd sizes) divided by its value at its centre, or raise ValueError
    unless that is its largest (the variance of the noise).
    """
    centre = shape[tuple(n // 2 for n in shape.shape)]
    if not centre > 0 or np.any(shape > centre):
        raise ValueError(
            f"the correlation of the noise of '{key}' must be largest at "
            f"its centre (at a separation of 0)")
    return shape / centre


def item_filenames(prefix: str, key: str) -> dict[str, str]:
    """Return the files of the arrays of a data item (see dump_data)."""
    return dict(
        data=f'{prefix}{key}_d.fits',
        mask=f'{prefix}{key}_m.fits',
        error=f'{prefix}{key}_e.fits',
        footprint=f'{prefix}{key}_f.fits')


def flatten_single_item(
        info: dict[str, Any], name: str
) -> dict[str, Any]:
    """
    Return the configuration of a dataset of one data item with the
    options of the item beside its own, as it is given.
    """
    info = dict(info)
    return info | info.pop(name)
