"""
Axisymmetric mass disks: a surface density (see surfaces) and a
vertical structure (see vertical), each chosen on its own.
"""

from collections.abc import Collection
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.integrate

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from .base import MassComponent, MassContext, forces_in_plane
from .surfaces import DiskSurface, disk_surface_parser
from .vertical import DiskVertical, disk_vertical_parser


__all__ = [
    'MassDisk'
]


class MassDisk(MassComponent):
    """
    An axisymmetric disk, of a surface density (see surfaces: e.g. an
    exponential, or that of a component of the galaxy) and a vertical
    structure (see vertical: a law, e.g. sech2, or that of a component),
    each chosen on its own. A disk follows at most one component (by its
    surface density, its vertical structure, or both); following it by
    both, it is the density of the component itself (each of its terms of
    the pdf of its height). Its forces are given anywhere: in its plane,
    the radial force of its circular velocity; off it, the pull of its
    rings (see DiskSurface.force).

    Parameters
    ----------
    name : str
        Its name (see MassComponent).
    surface : DiskSurface
        Its surface density.
    vertical : DiskVertical
        Its vertical structure.

    Raises
    ------
    ConfigError
        If the name is not valid, or the surface density and the vertical
        structure follow different components.
    """

    @staticmethod
    def type() -> str:
        return 'disk'

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'MassDisk':
        parseutils.load_option_and_update_info(
            disk_surface_parser, info, 'surface', required=True)
        parseutils.load_option_and_update_info(
            disk_vertical_parser, info, 'vertical', required=True)
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(
            surface=disk_surface_parser.dump(self._surface),
            vertical=disk_vertical_parser.dump(self._vertical))

    def __init__(
            self, name: str, surface: DiskSurface, vertical: DiskVertical):
        super().__init__(name)
        followed = {
            surface.followed_component(), vertical.followed_component()}
        followed.discard(None)
        if len(followed) > 1:
            raise ConfigError(
                f"a mass disk follows at most one component; its surface "
                f"density follows {surface.followed_component()!r} and its "
                f"vertical structure {vertical.followed_component()!r}")
        self._surface = surface
        self._vertical = vertical
        self._followed = tuple(followed)

    def surface(self) -> DiskSurface:
        """Return its surface density."""
        return self._surface

    def vertical(self) -> DiskVertical:
        """Return its vertical structure."""
        return self._vertical

    def followed_components(self) -> tuple[str, ...]:
        return self._followed

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._surface.pdescs() | self._vertical.pdescs()

    def vcirc2(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self._surface.vcirc2(R, self._vertical, params, context)

    def density(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self._surface.surface_density(R, params, context) \
            * self._vertical.height_pdf(R, z, params, context)

    def gives_density(self, thick: Collection[str]) -> bool:
        # (a thin disk has a surface density only)
        return not self._vertical.thin_given(thick)

    def surface_density(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self._surface.surface_density(R, params, context)

    def cell_masses(
            self,
            r_edges: np.ndarray,
            b_edges: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        if not self._vertical.thin(context):
            return super().cell_masses(r_edges, b_edges, params, context)
        # (thin: all the mass in the plane, in the cells next to it: that
        # of the rings between the radii, by 4-point Gauss-Legendre in R)
        r_edges = np.asarray(r_edges, dtype=float)
        nodes, weights = np.polynomial.legendre.leggauss(4)
        middle, half = (r_edges[1:] + r_edges[:-1]) / 2, np.diff(r_edges) / 2
        R = middle[:, None] + half[:, None] * nodes
        sigma = self._surface.surface_density(R, params, context)
        masses = np.zeros((r_edges.size - 1, np.size(b_edges) - 1))
        if b_edges[0] == 0:
            masses[:, 0] = 2 * np.pi * half * np.sum(
                weights * R * sigma, axis=1)
        return masses

    def gives_cell_masses(self, thick: Collection[str]) -> bool:
        return True

    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        # 2 pi int_0^r Sigma(R) s(R, sqrt(r^2 - R^2)) R dR, s the share of
        # the mass at R within that height (see DiskVertical.share_within),
        # by adaptive quadrature (to 1e-10)
        def mass(radius: float) -> float:
            def ring(R: float) -> float:
                height = np.sqrt(max(radius ** 2 - R ** 2, 0))
                sigma = self._surface.surface_density(
                    np.array([R]), params, context)[0]
                share = self._vertical.share_within(
                    R, height, params, context)
                return 2 * np.pi * R * sigma * float(share)
            value, _ = scipy.integrate.quad(
                ring, 0, radius, epsabs=0, epsrel=1e-10, limit=200)
            return value
        r = np.asarray(r, dtype=float)
        return np.reshape(
            [mass(radius) if radius > 0 else 0.0 for radius in r.ravel()],
            r.shape)

    def forces(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> tuple[np.ndarray, np.ndarray]:
        # (the radial force in its plane by its circular velocity, exactly
        # as it rotates there; off it, by the pull of its rings)
        R, z = np.broadcast_arrays(
            np.asarray(R, dtype=float), np.asarray(z, dtype=float))
        force_R, force_z = np.empty_like(R), np.zeros_like(R)
        plane, off = z == 0, z != 0
        if plane.any():
            force_R[plane] = forces_in_plane(
                R[plane], 0,
                lambda radii: self.vcirc2(radii, params, context),
                f"the mass disk {self.name()!r}")[0]
        if off.any():
            force_R[off] = self._surface.force(
                'R', R[off], z[off], self._vertical, params, context)
            force_z[off] = self.vertical_force(
                R[off], z[off], params, context)
        return force_R, force_z

    def vertical_force(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self._surface.force('z', R, z, self._vertical, params, context)
