"""
The shapes of spheroids (see spheroids.MassSpheroid): how the density
rho_sph(r) of their profile (see profiles and halos) is laid out in
space, on spheres or on flattened spheroids.
"""

import abc
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.integrate

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import parseutils
from .base import MassContext, forces_in_plane
from .profiles import SpheroidProfile
from .units import G


__all__ = [
    'ShapeFlattened',
    'ShapeSphere',
    'SpheroidShape',
    'spheroid_shape_parser'
]


def _flattened_nodes() -> tuple[np.ndarray, np.ndarray]:
    """
    Return the nodes phi of the integral of ShapeFlattened.vcirc2 over [0,
    pi / 2] and their weights: Gauss-Legendre points on panels of
    geometric widths towards both ends (where the profile can have a cusp,
    and where a flat spheroid changes on the scale q).
    """
    x, w = np.polynomial.legendre.leggauss(8)
    half = np.pi / 4
    widths = np.concatenate(([0], np.geomspace(1e-4, half, 10)))
    edges = np.concatenate((widths, np.pi / 2 - widths[::-1][1:]))
    lo, hi = edges[:-1, None], edges[1:, None]
    return (((hi - lo) * x + hi + lo) / 2).ravel(), ((hi - lo) * w / 2).ravel()


_FLATTENED_PHI, _FLATTENED_WEIGHTS = _flattened_nodes()


def _flattened_enclosed_mass(
        r: float,
        profile: SpheroidProfile,
        params: ParamValues,
        context: MassContext
) -> float:
    """
    Return the mass of a flattened spheroid (see ShapeFlattened) within
    the sphere of the radius r: that of the shells of m wholly inside it
    (each keeps the mass of its sphere, M(<m) of the profile), and the
    share that it cuts of the others, whose mass is uniform in the cosine
    mu of the angle of their points (R = m sqrt(1 - mu^2), z = q m mu): of
    an oblate one (q < 1), within m < r and the share 1 - sqrt((1 - r^2 /
    m^2) / (1 - q^2)) for r < m < r / q; of a prolate one (q > 1), within
    m < r / q and the share sqrt((r^2 / m^2 - 1) / (q^2 - 1)) for r / q <
    m < r. The share is integrated by adaptive quadrature (to 1e-10).
    """
    q = params['q']
    if r <= 0:
        return 0.0
    if q == 1:
        return float(profile.enclosed_mass(np.array([r]), params, context)[0])
    if q < 1:
        whole, edge = r, np.inf if q == 0 else r / q

        def share(m: float) -> float:
            return 1 - np.sqrt((1 - r ** 2 / m ** 2) / (1 - q ** 2))
    else:
        whole, edge = r / q, r

        def share(m: float) -> float:
            return np.sqrt((r ** 2 / m ** 2 - 1) / (q ** 2 - 1))

    def shells(m: float) -> float:
        density = profile.density(np.array([m]), params, context)[0]
        return 4 * np.pi * m ** 2 * density * share(m)
    inside = profile.enclosed_mass(np.array([whole]), params, context)[0]
    cut, _ = scipy.integrate.quad(
        shells, whole, edge, epsabs=0, epsrel=1e-10, limit=200)
    return float(inside) + cut


class SpheroidShape(parseutils.TypedSerializable, abc.ABC):
    """
    The shape of a spheroid: how it lays out the density of its profile,
    and its gravity from it.
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    def pdescs(self) -> dict[str, ParamDesc]:
        """Return its parameters, by name; none here."""
        return {}

    @abc.abstractmethod
    def spherical(self) -> bool:
        """Check whether it is a sphere."""
        pass

    @abc.abstractmethod
    def vcirc2(
            self,
            R: npt.ArrayLike,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return the square of the circular velocity in the plane of the
        spheroid of this shape and the profile (see MassComponent.vcirc2;
        params has the values of the parameters of both).
        """
        pass

    @abc.abstractmethod
    def density(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return the density of the spheroid at the points (R, z) (see
        MassComponent.density).
        """
        pass

    @abc.abstractmethod
    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return the mass of the spheroid within the spheres of the radii r
        (see MassComponent.enclosed_mass).
        """
        pass

    def forces(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the forces of the spheroid at the points (R, z) (see
        MassComponent.forces): in its plane only, here.
        """
        return forces_in_plane(
            R, z, lambda radii: self.vcirc2(radii, profile, params, context),
            f"a spheroid of the shape '{self.type()}'")

    def potential(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return the potential of the spheroid at the points (R, z) (see
        MassComponent.potential).

        Raises
        ------
        NotImplementedError
            If the shape does not give it (here).
        """
        raise NotImplementedError(
            f"a spheroid of the shape '{self.type()}' does not give its "
            f"potential")


class ShapeSphere(SpheroidShape):
    """
    A sphere: the density of the profile at the spherical radius r, its
    circular velocity G M(<r) / r and its forces exactly, and its
    potential as the profile gives it.
    """

    @staticmethod
    def type() -> str:
        return 'sphere'

    def spherical(self) -> bool:
        return True

    def vcirc2(
            self,
            R: npt.ArrayLike,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        # G M(<r) / r, 0 at the centre
        R = np.asarray(R, dtype=float)
        return G * np.divide(
            profile.enclosed_mass(R, params, context), R,
            out=np.zeros_like(R), where=R > 0)

    def density(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return profile.density(np.hypot(R, z), params, context)

    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return profile.enclosed_mass(r, params, context)

    def forces(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> tuple[np.ndarray, np.ndarray]:
        # -G M(<r) / r^3 (R, z), 0 at the centre
        R, z = np.broadcast_arrays(
            np.asarray(R, dtype=float), np.asarray(z, dtype=float))
        r = np.hypot(R, z)
        scale = -G * np.divide(
            profile.enclosed_mass(r, params, context), r ** 3,
            out=np.zeros_like(r), where=r > 0)
        return scale * R, scale * z

    def potential(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return profile.potential(np.hypot(R, z), params, context)


class ShapeFlattened(SpheroidShape):
    """
    A flattened (oblate, q < 1) or elongated (prolate, q > 1) spheroid of
    the axis ratio q (its height over its width), a parameter: the
    density rho_sph(m) / q at m^2 = R^2 + z^2 / q^2, the sphere of the
    profile squashed along z with the mass of each shell kept. The
    parameters of the profile keep their meaning: e.g. the mass, the mass
    M_D within m < r_D of a halo (the parameters of the sphere before it
    is squashed), and seen face-on, the projected profile of the sphere
    (e.g. a Sersic profile of the effective radius re). Its circular
    velocity in its plane (Binney & Tremaine 2008, 2.132) is v^2(R) = 4
    pi G int_0^R rho_sph(m) m^2 dm / sqrt(R^2 - e^2 m^2), e^2 = 1 - q^2,
    computed by quadrature (160 nodes: to about 1e-8 for Sersic and halo
    profiles, cusps too, and 1e-6 for Gaussians, down to q = 0.05). Its
    forces anywhere (Binney & Tremaine 2008, 2.140, with x =
    (1 + u)^-1/2): F_R = -4 pi G R int_0^1 rho_sph(m) x^2 dx / (1 - e^2
    x^2)^(1/2) and F_z = -4 pi G z int_0^1 rho_sph(m) x^2 dx / (1 - e^2
    x^2)^(3/2), m^2 = x^2 (R^2 + z^2 / (1 - e^2 x^2)), by the same
    quadrature (x = sin(phi)). Its potential is not given.
    """

    @staticmethod
    def type() -> str:
        return 'flattened'

    def spherical(self) -> bool:
        return False

    def pdescs(self) -> dict[str, ParamDesc]:
        return dict(q=ParamScalarDesc(
            desc="the axis ratio (height over width)", minimum=0))

    def vcirc2(
            self,
            R: npt.ArrayLike,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        # With m = R sin(phi): 4 pi G R^2 int_0^(pi/2) rho_sph(R sin(phi))
        # sin^2(phi) cos(phi) / sqrt(cos^2(phi) + q^2 sin^2(phi)) dphi
        R = np.asarray(R, dtype=float)
        q = params['q']
        sin, cos = np.sin(_FLATTENED_PHI), np.cos(_FLATTENED_PHI)
        weights = _FLATTENED_WEIGHTS * sin ** 2 * cos / np.sqrt(
            cos ** 2 + q ** 2 * sin ** 2)
        radii = R.ravel()
        out = np.zeros_like(radii)
        positive = radii > 0
        density = profile.density(
            radii[positive, None] * sin, params, context)
        out[positive] = 4 * np.pi * G * radii[positive] ** 2 \
            * (density @ weights)
        return out.reshape(R.shape)

    def density(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        q = params['q']
        return profile.density(
            np.hypot(R, np.asarray(z, dtype=float) / q), params, context) / q

    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        masses = [
            _flattened_enclosed_mass(radius, profile, params, context)
            for radius in r.ravel()]
        return np.reshape(masses, r.shape)

    def forces(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            profile: SpheroidProfile,
            params: ParamValues,
            context: MassContext
    ) -> tuple[np.ndarray, np.ndarray]:
        # With x = sin(phi), 1 - e^2 x^2 = cos^2(phi) + q^2 sin^2(phi)
        R, z = np.broadcast_arrays(
            np.asarray(R, dtype=float), np.asarray(z, dtype=float))
        q = params['q']
        sin, cos = np.sin(_FLATTENED_PHI), np.cos(_FLATTENED_PHI)
        squashed = cos ** 2 + q ** 2 * sin ** 2
        weights = _FLATTENED_WEIGHTS * sin ** 2 * cos
        radii, heights = R.ravel(), z.ravel()
        force_R, force_z = np.zeros_like(radii), np.zeros_like(radii)
        # (no force at the centre)
        off = (radii > 0) | (heights != 0)
        m = sin * np.sqrt(
            radii[off, None] ** 2 + heights[off, None] ** 2 / squashed)
        density = profile.density(m, params, context) * weights
        force_R[off] = -4 * np.pi * G * radii[off] * (
            density @ squashed ** -0.5)
        force_z[off] = -4 * np.pi * G * heights[off] * (
            density @ squashed ** -1.5)
        return force_R.reshape(R.shape), force_z.reshape(R.shape)


spheroid_shape_parser = parseutils.TypedParser(SpheroidShape, [
    ShapeSphere,
    ShapeFlattened])
