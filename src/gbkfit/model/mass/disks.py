"""
Axisymmetric mass disks: a surface density (see surfaces) and a
vertical structure (see vertical), each chosen on its own.
"""

from collections.abc import Collection
from typing import Any

import numpy as np
import numpy.typing as npt

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError, check_options
from .base import DiskDensity, MassComponent, MassContext, forces_in_plane
from .surfaces import DiskSurface, disk_surface_parser
from .vertical import DiskVertical, disk_vertical_parser


__all__ = [
    'MassDisk'
]


# The composite Gauss-Legendre quadratures of the enclosed masses of
# disks: the nodes of each panel, the largest widths of the panels in ln R
# (and in ln theta, see MassDisk.enclosed_mass), and the smallest of their
# edges but 0, relative to the largest one
_ENCLOSED_NODES = 8
_ENCLOSED_PANEL = 0.2
_ENCLOSED_SMALLEST = 1e-8


def _panel_edges(top: float, cuts: np.ndarray) -> np.ndarray:
    """
    Return the edges of the panels of a composite quadrature from 0 to
    top: 0, then at most _ENCLOSED_PANEL apart in ln from _ENCLOSED_SMALLEST
    times top, and the given cuts.
    """
    start = np.log(_ENCLOSED_SMALLEST * top)
    count = int(np.ceil((np.log(top) - start) / _ENCLOSED_PANEL))
    return np.unique(np.concatenate(
        [[0.0], np.exp(np.linspace(start, np.log(top), count + 1)), cuts]))


def _panel_nodes(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the nodes and the weights of the Gauss-Legendre quadratures of
    the panels between the edges (along the last axis; e.g. one row for
    each of several quadratures), flat over the panels of each: of shape
    (..., panels * _ENCLOSED_NODES).
    """
    nodes, weights = np.polynomial.legendre.leggauss(_ENCLOSED_NODES)
    middle = (edges[..., 1:] + edges[..., :-1]) / 2
    half = np.diff(edges, axis=-1) / 2
    shape = edges.shape[:-1] + (-1,)
    return ((middle[..., None] + half[..., None] * nodes).reshape(shape),
            (half[..., None] * weights).reshape(shape))


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
        check_options(MassDisk.__init__, locals())
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

    @classmethod
    def followed_kind(cls) -> 'type[DiskDensity]':
        return DiskDensity

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._surface.pdescs() | self._vertical.pdescs()

    def mass_ratios(self) -> dict[str, str]:
        return self._surface.mass_ratios()

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
        # By composite Gauss-Legendre quadrature, of all the radii at once:
        # of a thin disk, the mass within the cylinder of each radius, 2 pi
        # int_0^r Sigma(R) R dR, summed over panels between the radii, at
        # most _ENCLOSED_PANEL wide in ln R and cut at the knots of the
        # surface density (exact to rounding); of a thick one, 2 pi int_0^r
        # Sigma(R) s(R, sqrt(r^2 - R^2)) R dR (s the share of the mass at R
        # within the height of the sphere there, see
        # DiskVertical.share_within) in theta, R = r sin(theta), where it
        # is smooth, of panels at most _ENCLOSED_PANEL wide in ln theta and
        # ln (pi/2 - theta), cut at the knots of the surface density and
        # of the vertical structure (to about 1e-12)
        r = np.asarray(r, dtype=float)
        radii = r.ravel()
        inside = radii > 0
        out = np.zeros(radii.shape)
        if not np.any(inside):
            return out.reshape(r.shape)
        if self._vertical.thin(context):
            edges = _panel_edges(radii.max(), np.concatenate(
                [radii[inside], self._surface.knots(context)]))
            # (each panel alone: the masses between the edges)
            R, weights = _panel_nodes(
                np.stack([edges[:-1], edges[1:]], axis=1))
            masses = np.concatenate([[0], np.cumsum(2 * np.pi * np.sum(
                weights * R * self._surface.surface_density(
                    R, params, context), axis=1))])
            return masses[np.searchsorted(edges, radii)].reshape(r.shape)
        # (the knots beyond a radius cut it at pi/2: panels of no width)
        radius = radii[inside, None]
        quarter = _panel_edges(np.pi / 4, np.empty(0))
        common = np.union1d(quarter, np.pi / 2 - quarter)
        knots = np.union1d(
            self._surface.knots(context), self._vertical.knots(context))
        edges = np.sort(np.concatenate([
            np.broadcast_to(common, (len(radius), len(common))),
            np.arcsin(np.minimum(knots / radius, 1))], axis=1), axis=1)
        theta, weights = _panel_nodes(edges)
        R, h = radius * np.sin(theta), radius * np.cos(theta)
        out[inside] = 2 * np.pi * np.sum(
            weights * self._surface.surface_density(R, params, context)
            * self._vertical.share_within(R, h, params, context) * R * h,
            axis=1)
        return out.reshape(r.shape)

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
