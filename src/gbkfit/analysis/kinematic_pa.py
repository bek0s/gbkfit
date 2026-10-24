"""
The global kinematic position angle of a velocity field, by its
symmetrisation (Krajnovic et al. 2006, Appendix C, as PaFit).
"""

import dataclasses
import math

import numpy as np
import numpy.typing as npt

from ._fields import Field, galaxy_axes, reflections


__all__ = [
    'KinematicPA',
    'kinematic_pa'
]


@dataclasses.dataclass(frozen=True)
class KinematicPA:
    """
    The global kinematic position angle of a velocity field (see
    kinematic_pa).

    Attributes
    ----------
    pa : float
        The position angle (degrees, in [0, 360)) of the receding half of
        the major axis, as that of the models (posa): from +y towards -x.
    pa_error : float
        Its error (1 sigma; degrees).
    systemic : float
        The systemic velocity: the median of the field, corrected by the
        median difference between the field and its symmetrisation.
    angles, chi2 : np.ndarray
        The angles tried (degrees, in [0, 180)) and the chi-square of each.
    """
    pa: float
    pa_error: float
    systemic: float
    angles: np.ndarray
    chi2: np.ndarray


def kinematic_pa(
        values: npt.ArrayLike,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        centre: tuple[float, float],
        errors: npt.ArrayLike | None = None,
        step: float = 0.5,
        maxd: float | None = None
) -> KinematicPA:
    """
    Return the global kinematic position angle of a velocity field, as
    Krajnovic et al. (2006, Appendix C) and PaFit: for each angle, the
    chi-square between the field (less its median) and its symmetrisation
    about the axes of that angle (the mean of the field and of its
    reflections about the major axis, and of the other sign about the
    minor axis and through the centre: see asymmetry_maps); the angle of
    the least chi-square (refined by a parabola), turned to the receding
    side. Its error: with the chi-square scaled to a reduced one of 1, a
    third of the half-width of the angles within 9 + 3 sqrt(2 N) of its
    least (N the number of values), and at least half a step.

    Parameters
    ----------
    values : ArrayLike
        The velocity field (NaN where it has no value): of a regular grid,
        or of bins.
    x, y : ArrayLike
        The coordinates of its points (of the shape of values):
        their positions on the sky (arcsec, e.g.
        gridutils.sky_positions of their grid), in which the angles
        are those of the models (posa, north through east); the
        pixels of a rotated grid give angles in its pixel axes.
    centre : tuple of float
        The centre (x, y) of the galaxy.
    errors : ArrayLike, optional
        The errors of the values (1 without).
    step : float, optional
        The step of the angles tried (degrees).
    maxd : float, optional
        The farthest a reflected point may be from the nearest point of
        the field (see Field).

    Returns
    -------
    KinematicPA
        The position angle.

    Raises
    ------
    ValueError
        If the shapes differ, the field has fewer than three values, or
        the step is not in (0, 90].
    """
    if not 0 < step <= 90:
        raise ValueError(f"step must be in (0, 90]; it is {step}")
    values = np.asarray(values, dtype=float)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    good = np.isfinite(values)
    median = float(np.median(values[good]))
    f = values - median
    e = np.ones_like(f) if errors is None else np.asarray(errors, float)
    field = Field(f, x, y, maxd)
    points = x[good], y[good]

    def symmetrised(angle: float) -> np.ndarray:
        mirrors = reflections(*points, centre, angle)
        return np.nanmean([
            f[good], field(*mirrors['major']), -field(*mirrors['minor']),
            -field(*mirrors['rotation'])], axis=0)

    angles = np.arange(0, 180, step)
    chi2 = np.array([
        np.sum(((f[good] - symmetrised(angle)) / e[good]) ** 2)
        for angle in angles])
    best = int(np.argmin(chi2))
    # (a parabola through the least and its neighbours, round 180 degrees)
    lo, mid, hi = chi2[best - 1], chi2[best], chi2[(best + 1) % len(chi2)]
    curvature = lo - 2 * mid + hi
    shift = 0.5 * (lo - hi) / curvature if curvature > 0 else 0.0
    angle = angles[best] + step * float(np.clip(shift, -1, 1))
    n = int(good.sum())
    scaled = (chi2 - chi2.min()) * n / max(chi2.min(), np.finfo(float).tiny)
    inside = angles[scaled <= 9 + 3 * math.sqrt(2 * n)]
    # (the half-width of the angles inside, about the best, round 180)
    offsets = (inside - angles[best] + 90) % 180 - 90
    error = max((offsets.max() - offsets.min()) / 2 / 3, step / 2)
    major, _ = galaxy_axes(angle)
    along = (x[good] - centre[0]) * major[0] + (y[good] - centre[1]) * major[1]
    if np.mean(f[good] * np.sign(along)) < 0:
        angle += 180
    systemic = median + float(np.median(f[good] - symmetrised(angle)))
    return KinematicPA(
        pa=angle % 360, pa_error=float(error), systemic=systemic,
        angles=angles, chi2=chi2)
