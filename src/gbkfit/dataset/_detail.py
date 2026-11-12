"""
Helpers of the datasets of moments (see DatasetPixelMoments and
DatasetRegionMoments): the options and units of their data items, and
the reading of their files (their names and orders are those of
gbkfit.moments).
"""

from collections.abc import Mapping, Sequence
from typing import Any

import astropy.units
import numpy as np

import gbkfit.units
from gbkfit.grid import Coords, Grid, make_grid, pixels_on
from gbkfit.moments import HERMITE_ORDERS, MOMENT_ITEMS, MOMENT_ORDERS
from .data import Data, DataFiles


# How far (in pixels) the world coordinates of the maps of moments may put
# their pixels from those of the first (as those of the errors and masks of
# data, see read_arrays)
_GRID_TOLERANCE = 1e-3


def moment_options(values: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return the options of the moments given among the arguments of a
    constructor (see MOMENT_ITEMS; e.g. its locals()), by name.
    """
    return {
        name: values[name] for name in MOMENT_ITEMS
        if values[name] is not None}


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
    if invalid := sorted(set(moments) - set(MOMENT_ORDERS), key=str):
        raise ValueError(
            f"the orders of the moments are 0 to 7; they include {invalid}")
    if invalid := sorted(set(hermite) - set(HERMITE_ORDERS), key=str):
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
            f"h{order})" if order in HERMITE_ORDERS and unit.is_equivalent(
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
