"""
The names of the moments of spectra: the data items of moments (moment0
to moment7, and h3 and h4 of a Gauss-Hermite series), their orders, and
the methods that measure them, each up to its own order.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal, TypeAlias

from gbkfit.config import Choices


__all__ = [
    'HERMITE',
    'HERMITE_ORDERS',
    'MOMENTS',
    'MOMENT_ITEMS',
    'MOMENT_METHODS',
    'MOMENT_ORDERS',
    'Method',
    'check_moment_items',
    'check_moment_orders',
    'moment_item',
    'moment_orders',
    'split_moments'
]


# The data items of moments (and the options of the files of datasets):
# the central moments of the orders 0 to 7 (moment0, the flux; moment1,
# the mean velocity; moment2, the dispersion; moment3 and up, in
# (km/s)^k), and the coefficients h3 and h4 of a Gauss-Hermite series
# (dimensionless, as pPXF names them: not central moments)
MOMENT_ORDERS = tuple(range(8))
HERMITE_ORDERS = (3, 4)
MOMENTS = tuple(f'moment{order}' for order in MOMENT_ORDERS)
HERMITE = tuple(f'h{order}' for order in HERMITE_ORDERS)
MOMENT_ITEMS = MOMENTS + HERMITE

# How the moment maps are measured from the spectra of a cube: their
# moments, the moments of a Gaussian fitted to them (its flux, centre and
# dispersion), or a Gauss-Hermite series fitted to them (its flux, centre,
# dispersion, h3 and h4)
MOMENT_METHODS = ('moments', 'gauss_fit', 'gauss_hermite')
Method: TypeAlias = Annotated[Literal[MOMENT_METHODS], Choices(
    moments='Moments', gauss_fit='Gaussian fit',
    gauss_hermite='Gauss-Hermite fit')]

# The highest order of the maps of each method, as the native kernels
# measure them: the moments to 7, the three parameters of a Gaussian, and
# a Gauss-Hermite series to h4
_MAX_ORDERS = dict(moments=7, gauss_fit=2, gauss_hermite=4)


def moment_item(order: int, method: str) -> str:
    """
    Return the name of the data item of the moment of an order measured
    by a method (see MOMENT_METHODS): moment0 to moment7, the central
    moments (the flux, the mean velocity, the dispersion, and above in
    (km/s)^k), but the coefficients h3 and h4 of a Gauss-Hermite series
    (gauss_hermite; dimensionless, as pPXF names them).
    """
    if method == 'gauss_hermite' and order > 2:
        return f'h{order}'
    return f'moment{order}'


def moment_orders(keys: Iterable[str]) -> tuple[int, ...]:
    """
    Return the orders of the data items of moments of the given names
    (e.g. the keys of a dataset), central moments and coefficients of a
    Gauss-Hermite series (see MOMENT_ITEMS), sorted.
    """
    return tuple(sorted(
        int(key.removeprefix('moment').removeprefix('h')) for key in keys))


def split_moments[T](
        items: Mapping[str, T]
) -> tuple[dict[int, T], dict[int, T]]:
    """
    Return the central moments and the Gauss-Hermite coefficients among
    items by name (see MOMENT_ITEMS), by order, as datasets of moments
    take them.
    """
    moments = {
        int(name.removeprefix('moment')): item
        for name, item in items.items() if name in MOMENTS}
    hermite = {
        int(name.removeprefix('h')): item
        for name, item in items.items() if name in HERMITE}
    return moments, hermite


def check_moment_orders(
        orders: Sequence[int],
        method: str
) -> tuple[int, ...]:
    """
    Return the moment orders, sorted and unique.

    Parameters
    ----------
    orders : Sequence of int
        The orders.
    method : str
        The method that measures them (see MOMENT_METHODS).

    Returns
    -------
    tuple of int
        The orders, sorted and unique.

    Raises
    ------
    ValueError
        If there are none, or they are not valid for the method (a
        Gaussian has moments 0 to 2, a Gauss-Hermite series 0 to 4).
    """
    max_order = _MAX_ORDERS[method]
    orders = tuple(sorted(set(orders)))
    if not orders:
        raise ValueError("at least one moment order is required")
    if any(order < 0 or order > max_order for order in orders):
        raise ValueError(
            f"the moment orders of the method {method} must be between 0 "
            f"and {max_order}")
    return orders


def check_moment_items(keys: Iterable[str], method: str) -> None:
    """
    Raise ValueError if data items of moments of the given names (e.g.
    the keys of a dataset) are not all measured by a method: the
    coefficients h3 and h4 of a Gauss-Hermite series without the method
    gauss_hermite, or central moments of the orders 3 and 4 with it (see
    moment_item: of the same order, they are other numbers).
    """
    hermite = method == 'gauss_hermite'
    for key in keys:
        if key.startswith('h') and not hermite:
            raise ValueError(
                f"{key}, a coefficient of a Gauss-Hermite series, is "
                f"measured by the method gauss_hermite; the method is "
                f"{method}")
        if hermite and key in ('moment3', 'moment4'):
            raise ValueError(
                f"{key} is a central moment, in (km/s)^{key[-1]}; the method "
                f"gauss_hermite measures h{key[-1]}, a coefficient of a "
                f"Gauss-Hermite series (name the data h{key[-1]}, if they "
                f"are)")
