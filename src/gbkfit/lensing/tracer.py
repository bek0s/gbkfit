"""
The paths of the light of the planes of a lensed sky to the observer
(see Tracer): where each ray of an image crosses each plane, bent by the
lenses of the planes in front of it.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from gbkfit.cosmology import Cosmology
from .lens import Lens
from .pieces import SkyPoints
from .planes import Plane


__all__ = [
    'Tracer',
    'stretch'
]


class Tracer:
    """
    The paths of the light of planes (see plane_layout) to the observer:
    the ray of each point of the image crosses each plane at a point,
    where the lenses of the plane bend it towards the observer (Schneider,
    Ehlers & Falco 1992, eq. 9.7: multi-plane lensing, recursive):

        theta_q = theta_1 - sum_{p < q} f_pq alpha_p(theta_p),
        f_pq = D_pq / D_q,

    with theta_1 the point of the image, theta_q the point of the ray on
    plane q, and alpha_p the deflection of the lenses of plane p at
    D_ls / D_s = 1 (alpha-hat; see Lens.deflection), at the point of the
    ray there. The lenses of one plane add their deflections, and do not
    bend the light of its galaxies. With one lens plane and a source at
    the source_redshift of its lenses, theta_q = theta_1 - alpha, of the
    strengths they quote.

    The distances are those of the redshifts of the planes, in the
    cosmology, fixed at its construction.

    Parameters
    ----------
    planes : Sequence of Plane
        The planes, near to far (see plane_layout).
    lenses : Sequence of Lens
        The lenses, which the planes hold by index.
    cosmology : Cosmology
        The cosmology of the distances.

    Raises
    ------
    ValueError
        If the planes are not near to far, do not hold each lens once,
        or a lens quotes its pieces for a source in front of its plane
        (see Lens.quoted_factor).
    """

    def __init__(
            self,
            planes: Sequence[Plane],
            lenses: Sequence[Lens],
            cosmology: Cosmology
    ):
        planes = tuple(planes)
        redshifts = [plane.redshift for plane in planes]
        if any(a >= b for a, b in zip(redshifts, redshifts[1:])):
            raise ValueError(
                f"the planes must be near to far; their redshifts are "
                f"{redshifts}")
        held = sorted(i for plane in planes for i in plane.lenses)
        if held != list(range(len(lenses))):
            raise ValueError(
                f"the planes must hold each of the {len(lenses)} lenses "
                f"once; they hold {held}")
        self._planes = planes
        self._lenses = tuple(lenses)
        self._quoted = [0.0] * len(lenses)
        for plane in planes:
            for i in plane.lenses:
                self._quoted[i] = lenses[i].quoted_factor(
                    plane.redshift, cosmology)
        # The factors of each lens plane p, for each plane q behind it
        self._factors = {
            (p, q): cosmology.angular_diameter_distance_between(
                planes[p].redshift, planes[q].redshift)
            / cosmology.angular_diameter_distance(planes[q].redshift)
            for p in range(len(planes)) if planes[p].lenses
            for q in range(p + 1, len(planes))}

    def planes(self) -> tuple[Plane, ...]:
        """Return the planes, near to far."""
        return self._planes

    def lenses(self) -> tuple[Lens, ...]:
        """Return the lenses."""
        return self._lenses

    def lensed(self, plane: int) -> bool:
        """
        Check whether the light of a plane is bent: whether a plane in
        front of it has lenses.

        Parameters
        ----------
        plane : int
            The plane, by index.

        Returns
        -------
        bool
            True if its light is bent.
        """
        return any(self._planes[p].lenses for p in range(plane))

    def factor(self, near: int, far: int) -> float:
        """
        Return the factor D_pq / D_q of the deflection of the lenses of a
        plane p (near) on the rays of a plane q behind it (far).

        Parameters
        ----------
        near, far : int
            The planes, by index.

        Returns
        -------
        float
            The factor.

        Raises
        ------
        KeyError
            If the near plane has no lenses or is not in front of the
            far one.
        """
        return self._factors[near, far]

    def deflection(
            self,
            plane: int,
            points: SkyPoints,
            values: Sequence[Mapping[str, float]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the deflection of the lenses of a plane (by index) at
        points of it, at D_ls / D_s = 1 (alpha-hat): the sum of theirs
        (see Lens.deflection).

        Parameters
        ----------
        plane : int
            The plane.
        points : SkyPoints
            The points, on the plane.
        values : Sequence of Mapping
            The values of the parameters of each lens, by its own names
            (see Lens.pdescs).

        Returns
        -------
        tuple of np.ndarray
            The deflection (arcsec) along x and y, of the shape of the
            points.

        Raises
        ------
        ModelEvaluationError
            If a piece has no deflection at a point (e.g. off a map).
        """
        alpha_x = np.zeros(np.shape(points.x))
        alpha_y = np.zeros(np.shape(points.y))
        for i in self._planes[plane].lenses:
            lens_x, lens_y = self._lenses[i].deflection(
                points, values[i], self._quoted[i])
            alpha_x += lens_x
            alpha_y += lens_y
        return alpha_x, alpha_y

    def ray(
            self,
            plane: int,
            points: SkyPoints,
            deflections: Mapping[int, tuple[np.ndarray, np.ndarray]]
    ) -> SkyPoints:
        """
        Return the points where the rays of points of the image cross a
        plane (by index), of the deflections of the lens planes in front
        of it at the points of the rays there (see deflection).

        Parameters
        ----------
        plane : int
            The plane.
        points : SkyPoints
            The points of the image.
        deflections : Mapping of int to tuple of np.ndarray
            The deflection of each plane in front with lenses, by index.

        Returns
        -------
        SkyPoints
            The points of the rays on the plane.

        Raises
        ------
        KeyError
            If the deflection of a lens plane in front is missing.
        """
        x, y = points.x, points.y
        for p in range(plane):
            if self._planes[p].lenses:
                alpha_x, alpha_y = deflections[p]
                f = self._factors[p, plane]
                x = x - f * alpha_x
                y = y - f * alpha_y
        return SkyPoints(x, y, points.origin)

    def trace(
            self,
            points: SkyPoints,
            values: Sequence[Mapping[str, float]],
            upto: int | None = None
    ) -> tuple[SkyPoints, ...]:
        """
        Return the points where the rays of points of the image cross each
        plane, near to far, up to a plane (see the class).

        Parameters
        ----------
        points : SkyPoints
            The points of the image.
        values : Sequence of Mapping
            The values of the parameters of each lens, by its own names
            (see Lens.pdescs).
        upto : int, optional
            The farthest plane (by index); the last by default.

        Returns
        -------
        tuple of SkyPoints
            The points of the rays on each plane, from the first to upto.

        Raises
        ------
        ModelEvaluationError
            If a piece has no deflection at a point of a ray (e.g. off a
            map).
        """
        upto = len(self._planes) - 1 if upto is None else upto
        rays = []
        deflections = {}
        for q in range(upto + 1):
            rays.append(self.ray(q, points, deflections))
            # (the deflection of the farthest plane bends no ray traced)
            if self._planes[q].lenses and q < upto:
                deflections[q] = self.deflection(q, rays[q], values)
        return tuple(rays)

    def jacobian(
            self,
            plane: int,
            points: SkyPoints,
            values: Sequence[Mapping[str, float]],
            step: float
    ) -> np.ndarray:
        """
        Return the Jacobian of the points of the rays on a plane against
        the points of the image, d theta_q / d theta_1, by central
        differences of traced rays.

        Parameters
        ----------
        plane : int
            The plane, by index.
        points : SkyPoints
            The points of the image.
        values : Sequence of Mapping
            The values of the parameters of each lens, by its own names
            (see Lens.pdescs).
        step : float
            The step of the differences (arcsec), on either side of each
            point: e.g. half a pixel of the image.

        Returns
        -------
        np.ndarray
            The Jacobian at each point, of the shape of the points and
            (2, 2): d x_q / d x and d x_q / d y in its first row, those of
            y_q in its second.

        Raises
        ------
        ModelEvaluationError
            If a piece has no deflection at a point of a ray (e.g. off a
            map).
        """
        def ray(dx: float, dy: float) -> SkyPoints:
            moved = SkyPoints(points.x + dx, points.y + dy, points.origin)
            return self.trace(moved, values, plane)[plane]
        along_x = [ray(step, 0), ray(-step, 0)]
        along_y = [ray(0, step), ray(0, -step)]
        jacobian = np.empty(np.shape(points.x) + (2, 2))
        for col, (ahead, behind) in enumerate((along_x, along_y)):
            jacobian[..., 0, col] = (ahead.x - behind.x) / (2 * step)
            jacobian[..., 1, col] = (ahead.y - behind.y) / (2 * step)
        return jacobian


def stretch(jacobian: np.ndarray) -> np.ndarray:
    """
    Return how much the image stretches the plane of a Jacobian (see
    Tracer.jacobian) at each point, along the direction it stretches most:
    1 over the smallest singular value of the Jacobian (the Jacobian of
    several planes is not symmetric); infinite on a critical curve.

    Parameters
    ----------
    jacobian : np.ndarray
        The Jacobian at each point, of shape (..., 2, 2).

    Returns
    -------
    np.ndarray
        The stretch at each point, of the shape of the points.
    """
    smallest = np.linalg.svd(jacobian, compute_uv=False)[..., -1]
    with np.errstate(divide='ignore'):
        return 1 / smallest
