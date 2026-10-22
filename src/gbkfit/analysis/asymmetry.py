"""
The asymmetry of maps about the axes of a galaxy: the differences
between a map and its reflections, as NIRVANA's (Westfall et al.), and
the global asymmetry A_RC (Andersen & Bershady 2013).
"""

import dataclasses

import numpy as np
import numpy.typing as npt

from ._fields import Field, reflections


__all__ = [
    'AsymmetryMaps',
    'asymmetry_maps'
]


@dataclasses.dataclass(frozen=True)
class AsymmetryMaps:
    """
    The asymmetry of a map about the axes of a galaxy (see
    asymmetry_maps).

    Attributes
    ----------
    major, minor, rotation : np.ndarray
        The differences between the map and its reflection about the
        major axis, about the minor axis, and through the centre (a
        rotation by 180 degrees), of the shape of the map: f(reflected)
        - f, with f of the other sign at the reflections about the minor
        axis and through the centre for an odd map (e.g. a velocity
        field); NaN where the map or the reflection has no value. 0 in a
        map of a disk that rotates without other motions.
    a_rc : float
        The asymmetry A_RC of the map (Andersen & Bershady 2013, Eq. 7),
        the mean of those of its reflections about the major and the
        minor axes: sum |(|f| - |f_r|)| / s over sum (|f| + |f_r|) / (2
        s), of the values of the map and of its reflection f_r where both
        are, with s = sqrt(e^2 + e_r^2) of their errors (1 without); 0 for
        a symmetric map, 1 at most.
    """
    major: np.ndarray
    minor: np.ndarray
    rotation: np.ndarray
    a_rc: float


def asymmetry_maps(
        values: npt.ArrayLike,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        centre: tuple[float, float],
        pa: float,
        odd: bool = True,
        systemic: float = 0.0,
        errors: npt.ArrayLike | None = None,
        maxd: float | None = None
) -> AsymmetryMaps:
    """
    Return the asymmetry of a map about the axes of a galaxy: the
    differences between the map and its reflections, and its asymmetry
    A_RC (see AsymmetryMaps). The values at the reflected points are
    interpolated linearly between the points of the map (see Field).

    Parameters
    ----------
    values : ArrayLike
        The map (NaN where it has no value): of a regular grid, or of bins.
    x, y : ArrayLike
        The coordinates of its points (of the shape of values).
    centre : tuple of float
        The centre (x, y) of the galaxy.
    pa : float
        The position angle (degrees) of the major axis of the galaxy, as
        that of the models (posa): from +y towards -x.
    odd : bool, optional
        Whether the map is odd about the minor axis (e.g. a velocity
        field) or even (e.g. a dispersion or flux map).
    systemic : float, optional
        The value about which an odd map is odd (e.g. the systemic
        velocity), taken from it.
    errors : ArrayLike, optional
        The errors of the values, for A_RC.
    maxd : float, optional
        The farthest a reflected point may be from the nearest point of
        the map (see Field).

    Returns
    -------
    AsymmetryMaps
        The asymmetry.

    Raises
    ------
    ValueError
        If the shapes differ, or the map has fewer than three values.
    """
    values = np.asarray(values, dtype=float)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    f = values - systemic if odd else values
    field = Field(f, x, y, maxd)
    sign = -1 if odd else 1
    reflected = {
        name: field(*point)
        for name, point in reflections(x, y, centre, pa).items()}
    differences = dict(
        major=reflected['major'] - f,
        minor=sign * reflected['minor'] - f,
        rotation=sign * reflected['rotation'] - f)
    if errors is None:
        e = np.ones_like(f)
        e_reflected = dict.fromkeys(('major', 'minor'), e)
    else:
        e = np.asarray(errors, dtype=float)
        error_field = Field(e, x, y, field.maxd())
        e_reflected = {
            name: error_field(*point)
            for name, point in reflections(x, y, centre, pa).items()
            if name in ('major', 'minor')}
    a_rc = []
    for name in ('major', 'minor'):
        s = np.hypot(e, e_reflected[name])
        good = np.isfinite(f) & np.isfinite(reflected[name]) & np.isfinite(s)
        a, b, s = np.abs(f[good]), np.abs(reflected[name][good]), s[good]
        a_rc.append(np.sum(np.abs(a - b) / s) / np.sum((a + b) / (2 * s)))
    return AsymmetryMaps(a_rc=float(np.mean(a_rc)), **differences)
