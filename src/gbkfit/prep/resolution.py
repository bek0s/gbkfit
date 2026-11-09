"""
The LSF of an instrument of a resolving power that changes along the
spectral axis (e.g. of a curve of R against wavelength, or of a
polynomial): a Gaussian of a FWHM of c / R km/s of the light.
"""

import astropy.constants
import astropy.table
import astropy.units as u
import numpy as np
import numpy.typing as npt


__all__ = [
    'resolving_power_table'
]


# The speed of light (km/s)
_C = astropy.constants.c.to_value(u.km / u.s)


def resolving_power_table(
        points: u.Quantity,
        resolving_power: npt.ArrayLike,
        name: str = 'wavelength'
) -> astropy.table.QTable:
    """
    Return the LSF of resolving powers at points of the spectral axis: a
    table of the FWHM of its Gaussian (km/s of the light: c / R) against
    the points.

    Parameters
    ----------
    points : Quantity
        The points: wavelengths, frequencies or velocities.
    resolving_power : array_like
        The resolving power at each point.
    name : str, optional
        The name of the column of the points.

    Returns
    -------
    QTable
        The table: the points, and the FWHM (column fwhm).

    Raises
    ------
    ValueError
        If the resolving powers are not positive numbers, one at each
        point.
    """
    power = np.asarray(resolving_power, dtype=float)
    if power.shape != np.shape(points) or not np.all(power > 0):
        raise ValueError(
            f"a resolving power is a positive number at each point; there "
            f"are {np.size(points)} points, and the powers are "
            f"{power.tolist()}")
    return astropy.table.QTable({
        name: points, 'fwhm': _C / power * (u.km / u.s)})
