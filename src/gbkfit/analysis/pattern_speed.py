"""
The pattern speed of a galaxy by the method of Tremaine & Weinberg
(1984), from its flux and velocity maps.
"""

import dataclasses

import numpy as np
import numpy.typing as npt
import scipy.spatial

from ._fields import galaxy_axes


__all__ = [
    'PatternSpeed',
    'pattern_speed'
]


@dataclasses.dataclass(frozen=True)
class PatternSpeed:
    """
    The pattern speed of a galaxy (see pattern_speed).

    Attributes
    ----------
    omega_sini : float
        Omega_p sin(i): the slope of <V> against <X> over the slits (in
        the units of the velocities per those of the coordinates, e.g.
        km/s/arcsec); positive for a pattern that turns as a disk whose
        receding side is at the position angle.
    omega_sini_error : float
        Its error (1 sigma), from the scatter of the slits about the line.
    omega, omega_error : float or None
        Omega_p and its error, if the inclination is given.
    intercept : float
        The intercept of the line (0 for the right systemic velocity and
        centre).
    offsets : np.ndarray
        The offset of each slit from the major axis.
    x, v : np.ndarray
        <X> and <V> of each slit: the flux-weighted means of the position
        along the major axis and of the velocity (less the systemic).
    """
    omega_sini: float
    omega_sini_error: float
    omega: float | None
    omega_error: float | None
    intercept: float
    offsets: np.ndarray
    x: np.ndarray
    v: np.ndarray


def pattern_speed(
        flux: npt.ArrayLike,
        velocity: npt.ArrayLike,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        centre: tuple[float, float],
        pa: float,
        incl: float | None = None,
        systemic: float = 0.0,
        width: float | None = None,
        offsets: npt.ArrayLike | None = None
) -> PatternSpeed:
    """
    Return the pattern speed of a galaxy by the method of Tremaine &
    Weinberg (1984): along slits parallel to its major axis, the
    flux-weighted means of the position along the axis, <X>, and of the
    velocity less the systemic, <V>, which a pattern that turns at
    Omega_p makes <V> = Omega_p sin(i) <X>; Omega_p sin(i) is the slope
    of the straight line fitted to the slits (with an intercept, which
    takes errors of the systemic velocity and of the centre). The tracer
    must obey the continuity equation (e.g. old stars), and the points of
    the maps be of equal areas (e.g. pixels).

    Parameters
    ----------
    flux, velocity : ArrayLike
        The flux and the velocity at the points (NaN where none).
    x, y : ArrayLike
        The coordinates of the points (of the shape of flux):
        their positions on the sky (arcsec, e.g.
        gridutils.sky_positions of their grid), in which the angles
        are those of the models (posa, north through east); the
        pixels of a rotated grid give angles in its pixel axes.
    centre : tuple of float
        The centre (x, y) of the galaxy.
    pa : float
        The position angle (degrees) of its major axis, as that of the
        models (posa): from +y towards -x.
    incl : float, optional
        Its inclination (degrees), for Omega_p.
    systemic : float, optional
        Its systemic velocity.
    width : float, optional
        The width of the slits: by default, the spacing of the points.
    offsets : ArrayLike, optional
        The offsets of the slits from the major axis: by default, every
        width across the points.

    Returns
    -------
    PatternSpeed
        The pattern speed.

    Raises
    ------
    ValueError
        If the shapes differ, or fewer than three slits have flux.
    """
    flux, velocity, x, y = (
        np.asarray(a, dtype=float) for a in (flux, velocity, x, y))
    if not flux.shape == velocity.shape == x.shape == y.shape:
        raise ValueError(
            "the flux, the velocity and the coordinates must have one shape")
    good = np.isfinite(flux) & np.isfinite(velocity) \
        & np.isfinite(x) & np.isfinite(y)
    flux, velocity = flux[good], velocity[good] - systemic
    major, minor = galaxy_axes(pa)
    dx, dy = x[good] - centre[0], y[good] - centre[1]
    along = dx * major[0] + dy * major[1]
    across = dx * minor[0] + dy * minor[1]
    if width is None:
        points = np.column_stack((x[good], y[good]))
        distances, _ = scipy.spatial.cKDTree(points).query(points, k=2)
        width = float(np.median(distances[:, 1]))
    if offsets is None:
        reach = np.max(np.abs(across))
        offsets = np.arange(-reach, reach + width / 2, width)
    rows = []
    for offset in np.asarray(offsets, dtype=float):
        inside = np.abs(across - offset) <= width / 2
        total = flux[inside].sum()
        if inside.any() and total != 0:
            rows.append((
                offset, np.sum(flux[inside] * along[inside]) / total,
                np.sum(flux[inside] * velocity[inside]) / total))
    if len(rows) < 3:
        raise ValueError(
            f"fewer than three slits have flux ({len(rows)}): too few to "
            f"fit the line")
    offsets, mean_x, mean_v = map(np.array, zip(*rows))
    (slope, intercept), covariance = np.polyfit(mean_x, mean_v, 1, cov=True)
    error = float(np.sqrt(covariance[0, 0]))
    omega = omega_error = None
    if incl is not None:
        sin_i = np.sin(np.radians(incl))
        omega, omega_error = float(slope / sin_i), error / sin_i
    return PatternSpeed(
        omega_sini=float(slope), omega_sini_error=error, omega=omega,
        omega_error=omega_error, intercept=float(intercept),
        offsets=offsets, x=mean_x, v=mean_v)
