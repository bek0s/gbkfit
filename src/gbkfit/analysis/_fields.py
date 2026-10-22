"""
The maps that the analysis tools take: values at points (x, y), of a
regular map or of bins, interpolated between them, and the frame of the
axes of a galaxy on them.
"""

import numpy as np
import numpy.typing as npt
import scipy.interpolate
import scipy.spatial


__all__ = [
    'Field',
    'galaxy_axes',
    'reflections'
]


class Field:
    """
    The values of a map at points (x, y) (arrays of one shape; NaN where it
    has no value), interpolated linearly between the points that have
    one, where such a point lies within maxd (else NaN: the gaps of the
    map are not bridged).

    Parameters
    ----------
    values, x, y : ArrayLike
        The values and the coordinates of the points (any units).
    maxd : float, optional
        The farthest a point may be from the nearest point with a value:
        by default, the median distance between neighbouring points (the
        spacing of the map).

    Raises
    ------
    ValueError
        If the shapes differ, or fewer than three points have a value.
    """

    def __init__(
            self,
            values: npt.ArrayLike,
            x: npt.ArrayLike,
            y: npt.ArrayLike,
            maxd: float | None = None
    ):
        values, x, y = (np.asarray(a, dtype=float) for a in (values, x, y))
        if not values.shape == x.shape == y.shape:
            raise ValueError(
                f"the values and the coordinates must have one shape; they "
                f"are {values.shape}, {x.shape} and {y.shape}")
        good = np.isfinite(values) & np.isfinite(x) & np.isfinite(y)
        if good.sum() < 3:
            raise ValueError("the map has fewer than three values")
        points = np.column_stack((x[good], y[good]))
        self._tree = scipy.spatial.cKDTree(points)
        if maxd is None:
            maxd = float(np.median(self._tree.query(points, k=2)[0][:, 1]))
        self._maxd = maxd
        self._interpolator = scipy.interpolate.LinearNDInterpolator(
            points, values[good])

    def maxd(self) -> float:
        """Return the farthest a point may be from the nearest value."""
        return self._maxd

    def __call__(self, x: npt.ArrayLike, y: npt.ArrayLike) -> np.ndarray:
        """Return the values at the points (x, y) (NaN where none)."""
        x, y = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        values = self._interpolator(x, y)
        distance, _ = self._tree.query(np.stack((x, y), axis=-1))
        return np.where(distance <= self._maxd, values, np.nan)


def galaxy_axes(pa: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the unit vectors (x, y) of the major axis of a galaxy of the
    position angle pa (degrees, as the position angle posa of the models
    on the axes of their maps: from +y towards -x), towards pa, and of its
    minor axis, 90 degrees further.
    """
    angle = np.radians(pa)
    major = np.array([-np.sin(angle), np.cos(angle)])
    minor = np.array([-np.cos(angle), -np.sin(angle)])
    return major, minor


def reflections(
        x: np.ndarray,
        y: np.ndarray,
        centre: tuple[float, float],
        pa: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Return the points (x, y) reflected about the major axis of a galaxy of
    the given centre and position angle (see galaxy_axes), about its minor
    axis, and through its centre (a rotation by 180 degrees), by name
    ('major', 'minor', 'rotation').
    """
    major, minor = galaxy_axes(pa)
    dx, dy = x - centre[0], y - centre[1]
    along = dx * major[0] + dy * major[1]
    across = dx * minor[0] + dy * minor[1]

    def point(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (centre[0] + a * major[0] + b * minor[0],
                centre[1] + a * major[1] + b * minor[1])
    return dict(
        major=point(along, -across),
        minor=point(-along, across),
        rotation=point(-along, -across))
