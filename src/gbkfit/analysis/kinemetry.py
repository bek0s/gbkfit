"""
Kinemetry (Krajnovic et al. 2006): the harmonic expansion of a map along
the ellipses that fit it best, ring by ring.
"""

import dataclasses

import numpy as np
import numpy.typing as npt
import scipy.optimize

from ._fields import Field, galaxy_axes


__all__ = [
    'Kinemetry',
    'kinemetry'
]


# The points along each ellipse, at even steps of the eccentric anomaly
_SAMPLES = 100

# The grid of position angles (degrees, over 180) and axis ratios that
# start the fit of the ellipse of each ring
_GRID_PA = np.arange(0, 180, 10.0)
_GRID_Q = np.arange(0.1, 1.01, 0.1)

# The least axis ratio of an ellipse
_MIN_Q = 0.05


@dataclasses.dataclass(frozen=True)
class Kinemetry:
    """
    The kinemetry of a map (see kinemetry): of each ring, the ellipse that
    fits the map best and the harmonic expansion of the map along it, K(
    psi) = A0 + sum_n A_n sin(n psi) + B_n cos(n psi), psi the eccentric
    anomaly from the major axis (towards the position angle). NaN in the
    rings that the map does not cover.

    Attributes
    ----------
    radii : np.ndarray
        The semi-major axes of the rings.
    pa : np.ndarray
        The position angle of each ring (degrees, as that of the models:
        from +y towards -x), of the receding side for an odd map.
    q : np.ndarray
        The axis ratio of each ring.
    a0 : np.ndarray
        A0 of each ring (e.g. the systemic velocity of a velocity field).
    a, b : np.ndarray
        A_n and B_n of each ring (rings, n = 1 to nterms).
    coverage : np.ndarray
        The fraction of each ellipse with values of the map.
    """
    radii: np.ndarray
    pa: np.ndarray
    q: np.ndarray
    a0: np.ndarray
    a: np.ndarray
    b: np.ndarray
    coverage: np.ndarray

    def k(self) -> np.ndarray:
        """
        Return the amplitudes k_n = sqrt(A_n^2 + B_n^2) of each ring
        (rings, n = 1 to nterms): k1, the rotation of a velocity field,
        and k5 / k1, its departure from simple rotation.
        """
        return np.hypot(self.a, self.b)


def kinemetry(
        values: npt.ArrayLike,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        centre: tuple[float, float],
        radii: npt.ArrayLike | None = None,
        odd: bool = True,
        nterms: int = 5,
        cover: float = 0.75,
        maxd: float | None = None
) -> Kinemetry:
    """
    Return the kinemetry of a map (Krajnovic et al. 2006): for each ring,
    the ellipse (position angle and axis ratio, about the given centre)
    that fits the map best, and the harmonic expansion of the map along
    it (see Kinemetry). The best ellipse of an odd map (e.g. a velocity
    field) makes A1, A3 and B3 least, that of an even one (e.g. a
    dispersion or flux map) A1, B1, A2 and B2: by least squares, from the
    best of a grid of angles and axis ratios. The map is sampled at 100
    points along each ellipse, interpolated linearly between its points
    (see Field); a ring of which less than cover of the points have values
    is not fitted.

    Parameters
    ----------
    values : ArrayLike
        The map (NaN where it has no value): of a regular grid, or of bins.
    x, y : ArrayLike
        The coordinates of its points (of the shape of values).
    centre : tuple of float
        The centre (x, y) of the galaxy.
    radii : ArrayLike, optional
        The semi-major axes of the rings: by default, from the spacing of
        the map, by steps of 10% (at least the spacing), to the farthest
        point with a value.
    odd : bool, optional
        Whether the map is odd (e.g. a velocity field) or even.
    nterms : int, optional
        The number of harmonics, at least 3.
    cover : float, optional
        The least fraction of an ellipse with values, in (0, 1].
    maxd : float, optional
        The farthest a point of an ellipse may be from the nearest point
        of the map (see Field).

    Returns
    -------
    Kinemetry
        The kinemetry.

    Raises
    ------
    ValueError
        If the shapes differ, the map has fewer than three values, nterms
        is less than 3, or cover is not in (0, 1].
    """
    if nterms < 3:
        raise ValueError(f"nterms must be at least 3; it is {nterms}")
    if not 0 < cover <= 1:
        raise ValueError(f"cover must be in (0, 1]; it is {cover}")
    values = np.asarray(values, dtype=float)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    field = Field(values, x, y, maxd)
    if radii is None:
        radii = _default_radii(values, x, y, centre, field.maxd())
    radii = np.asarray(radii, dtype=float)
    psi = np.linspace(0, 2 * np.pi, _SAMPLES, endpoint=False)
    n = np.arange(1, nterms + 1)
    design = np.column_stack(
        [np.ones_like(psi), np.sin(np.outer(psi, n)), np.cos(np.outer(psi, n))])
    # (the terms that the best ellipse makes least: A1, A3, B3 of an odd
    # map, A1, B1, A2, B2 of an even one; columns of the design)
    least = [1, 3, nterms + 3] if odd else [1, nterms + 1, 2, nterms + 2]

    def expansion(
            radius: float, pa: float, q: float
    ) -> tuple[np.ndarray | None, float]:
        """The coefficients along an ellipse, and its coverage."""
        major, minor = galaxy_axes(pa)
        along, across = radius * np.cos(psi), radius * q * np.sin(psi)
        samples = field(
            centre[0] + along * major[0] + across * minor[0],
            centre[1] + along * major[1] + across * minor[1])
        good = np.isfinite(samples)
        coverage = good.mean()
        if coverage < cover:
            return None, coverage
        coefficients, *_ = np.linalg.lstsq(
            design[good], samples[good], rcond=None)
        return coefficients, coverage

    def misfit(params: np.ndarray, radius: float) -> np.ndarray:
        coefficients, _ = expansion(radius, *params)
        if coefficients is None:
            return np.full(len(least), 1e10)
        return coefficients[least]

    rows = []
    for radius in radii:
        starts = [(pa, q) for pa in _GRID_PA for q in _GRID_Q]
        costs = [np.sum(misfit(np.array(s), radius) ** 2) for s in starts]
        best = np.array(starts[int(np.argmin(costs))])
        if not np.min(costs) < 1e20:
            rows.append(None)
            continue
        fit = scipy.optimize.least_squares(
            misfit, best, args=(radius,),
            bounds=([best[0] - 90, _MIN_Q], [best[0] + 90, 1.0]))
        pa, q = fit.x
        coefficients, coverage = expansion(radius, pa, q)
        if coefficients is None:
            rows.append(None)
            continue
        # (an odd map: the receding side, where B1 > 0)
        if odd and coefficients[nterms + 1] < 0:
            pa += 180
            coefficients, coverage = expansion(radius, pa, q)
        rows.append((pa % 360 if odd else pa % 180, q, coefficients, coverage))
    return _collect(radii, rows, nterms)


def _default_radii(
        values: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        centre: tuple[float, float],
        spacing: float
) -> np.ndarray:
    """
    Return the semi-major axes of the rings of a map: from its spacing, by
    steps of 10% (at least the spacing), to its farthest point with a
    value.
    """
    good = np.isfinite(values)
    farthest = float(np.max(np.hypot(
        x[good] - centre[0], y[good] - centre[1])))
    radii = [spacing]
    while (following := max(radii[-1] * 1.1, radii[-1] + spacing)) \
            <= farthest:
        radii.append(following)
    return np.array(radii)


def _collect(
        radii: np.ndarray,
        rows: list[tuple[float, float, np.ndarray, float] | None],
        nterms: int
) -> Kinemetry:
    """Return the kinemetry of the rings (None: not fitted)."""
    pa, q, a0, coverage = (np.full(len(radii), np.nan) for _ in range(4))
    a, b = np.full((len(radii), nterms), np.nan), np.full(
        (len(radii), nterms), np.nan)
    for i, row in enumerate(rows):
        if row is None:
            continue
        pa[i], q[i], coefficients, coverage[i] = row
        a0[i] = coefficients[0]
        a[i] = coefficients[1:nterms + 1]
        b[i] = coefficients[nterms + 1:]
    return Kinemetry(
        radii=radii, pa=pa, q=q, a0=a0, a=a, b=b, coverage=coverage)
