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
    'resolving_power_lsf_table'
]


def resolving_power_lsf_table(
        points: u.Quantity,
        resolving_power: npt.ArrayLike,
        name: str | None = None
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
        The name of the column of the points; by default, of their kind:
        wavelength, frequency or velocity.

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
    if power.shape != np.shape(points):
        raise ValueError(
            f"there must be a resolving power at each point; there are "
            f"{np.size(points)} points and {power.size} resolving powers")
    if not np.all(power > 0):
        raise ValueError(
            f"resolving powers must be positive; the lowest is "
            f"{np.min(power):g}")
    if name is None:
        name = ('wavelength' if points.unit.is_equivalent(u.m)
                else 'frequency' if points.unit.is_equivalent(u.Hz)
                else 'velocity')
    return astropy.table.QTable({
        name: points, 'fwhm': (astropy.constants.c / power).to(u.km / u.s)})
