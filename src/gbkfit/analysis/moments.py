"""
Moment maps of spectral cubes (e.g. of data, or of models evaluated
alone), measured as the observables of moments measure them.
"""

import dataclasses
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from gbkfit.driver import Driver
from gbkfit.driver.drivers.host import DriverHost
from gbkfit.observation.observables import _moments
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'MomentMaps',
    'moment_maps'
]


@dataclasses.dataclass(frozen=True)
class MomentMaps:
    """
    The moment maps of a spectral cube (see moment_maps).

    Attributes
    ----------
    data : dict of str to np.ndarray
        The map (ny, nx) of each order, by name (moment0, moment1, ...).
    mask : np.ndarray
        Where the moments are measured (ny, nx): False where the spectrum
        is masked, whose moments from order 1 are NaN (moment 0 keeps its
        value, but is NaN where the spectrum has no value at all).
    """
    data: dict[str, np.ndarray]
    mask: np.ndarray


def moment_maps(
        cube: npt.ArrayLike,
        velocities: npt.ArrayLike,
        orders: Sequence[int] = (0, 1, 2),
        method: str = 'moments',
        flux_floor: float = 0.0,
        driver: Driver | None = None
) -> MomentMaps:
    """
    Return the moment maps of a spectral cube, measured as the observable
    of moments (PixelMoments) measures those of its models: the moments
    of each spectrum (its flux, mean velocity, dispersion, and central
    moments above 2), or those of a Gaussian or a Gauss-Hermite series
    fitted to it. A channel without a value (NaN) counts as one without
    flux.

    Parameters
    ----------
    cube : ArrayLike
        The cube (channels, ny, nx).
    velocities : ArrayLike
        The velocities of its channels, evenly spaced, in either order.
    orders : Sequence of int, optional
        The orders of the maps.
    method : str, optional
        How the maps are measured: 'moments'; 'gaussian_fit' (the flux,
        centre and dispersion of a Gaussian fitted to each spectrum, for
        the orders 0 to 2); or 'gauss_hermite' (of a Gauss-Hermite series,
        for the orders 0 to 4: its flux, centre, dispersion, h3 and h4).
    flux_floor : float, optional
        The spectra whose moment 0 is not above this fraction of the
        largest (in absolute value) are masked: from 0 to below 1.
    driver : Driver, optional
        The driver that measures them (a DriverHost by default).

    Returns
    -------
    MomentMaps
        The maps.

    Raises
    ------
    ValueError
        If the cube is not of the velocities, they are not evenly spaced
        (or fewer than two), or the options are not valid.
    """
    cube = np.asarray(cube)
    velocities = np.asarray(velocities, dtype=float)
    if cube.ndim != 3 or velocities.shape != (len(cube),):
        raise ValueError(
            "the cube must be (channels, ny, nx), of one velocity per "
            "channel")
    steps = np.diff(velocities)
    if len(velocities) < 2 or not np.allclose(
            steps, steps[0], rtol=1e-6, atol=0) or steps[0] == 0:
        raise ValueError(
            "the velocities must be at least two, and evenly spaced")
    if method not in _moments.METHODS:
        raise ValueError(
            f"the method must be one of {_moments.METHODS}; it is {method!r}")
    if not 0 <= flux_floor < 1:
        raise ValueError(
            f"the flux floor must be from 0 to below 1; it is {flux_floor}")
    try:
        orders = _moments.check_moment_orders(orders, method)
    except ConfigError as e:
        raise ValueError(str(e)) from None
    # (the velocities from the lowest, so that moment 0 is positive)
    if steps[0] < 0:
        cube, velocities = cube[::-1], velocities[::-1]
    dtype = np.float32 if cube.dtype == np.float32 else np.float64
    blank = np.isnan(cube).all(axis=0)
    cube = np.nan_to_num(cube.astype(dtype), nan=0.0)
    if driver is None:
        driver = DriverHost()
    _, ny, nx = cube.shape
    plan = _moments.MomentsPlan(
        driver, (nx, ny), orders, flux_floor, method, dtype)
    step = (velocities[-1] - velocities[0]) / (len(velocities) - 1)
    maps = plan.evaluate(
        (1, 1, step), (0, 0, velocities[0]), driver.mem_copy_h2d(cube),
        None)
    data = {key: driver.mem_copy_d2h(value['d'])
            for key, value in maps.items()}
    mask = driver.mem_copy_d2h(maps[f'moment{orders[0]}']['m']) == 1
    mask &= ~blank
    for value in data.values():
        value[blank] = np.nan
    return MomentMaps(data=data, mask=mask)
