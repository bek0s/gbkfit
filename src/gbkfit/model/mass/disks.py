"""
Axisymmetric mass disks: a surface density (see surfaces) and a
vertical structure (see vertical), each chosen on its own (MassDisk), or
the density that a disk component traces (TracedDisk).
"""

from collections.abc import Collection
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import scipy.integrate
from annotated_types import Gt

from gbkfit.config import check_name, check_options
from gbkfit.params import ParamDesc, ParamValues
from .base import DiskDensity, MassComponent, MassContext, forces_in_plane
from .surfaces import DiskSurface, SurfaceComponent, disk_surface_parser
from .vertical import DiskVertical, VerticalComponent, disk_vertical_parser


__all__ = [
    'MassDisk',
    'TracedDisk'
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
    An axisymmetric disk of a surface density and a vertical structure,
    each chosen on its own. Its surface density is a law, e.g. an
    exponential (see surfaces), and its vertical structure a law, e.g.
    sech2, or the layer of a disk component of the galaxy (see vertical).
    A disk that traces the density of a component is a TracedDisk. Its
    forces are given anywhere: in its plane, the radial force of its
    circular velocity; off it, the pull of its rings (see
    DiskSurface.force).

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
    ValueError
        If the name is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'disk'

    @staticmethod
    def type_title() -> str:
        return 'Disk'

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(
            surface=disk_surface_parser.dump(self._surface),
            vertical=disk_vertical_parser.dump(self._vertical))

    def __init__(
            self, name: str, surface: DiskSurface, vertical: DiskVertical):
        check_options(MassDisk.__init__, locals())
        self._init(name, surface, vertical)

    def _init(
            self, name: str, surface: DiskSurface, vertical: DiskVertical
    ) -> None:
        """Make the disk (for the constructors, which check the options)."""
        super().__init__(name)
        self._surface = surface
        self._vertical = vertical
        # (the components it follows, by its surface density and by its
        # vertical structure: one, two, or none)
        self._followed = tuple(dict.fromkeys(
            c for c in (surface.followed_component(),
                        vertical.followed_component()) if c is not None))

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

    def gives_surface_density(self) -> bool:
        return True

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
            weights * self._column_within(R, h, params, context) * R * h,
            axis=1)
        return out.reshape(r.shape)

    def _column_within(
            self,
            R: np.ndarray,
            h: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its mass per unit area (Msun/kpc^2) within the heights h
        (kpc) of its plane, at the radii R (kpc): its surface density
        times the share of its vertical structure there.
        """
        return self._surface.surface_density(R, params, context) \
            * self._vertical.share_within(R, h, params, context)

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


class TracedDisk(MassDisk):
    """
    A disk whose mass traces a disk component of the galaxy: the density
    of the component (e.g. of its light, or the optical depth of its
    dust) times a mass per unit of it, upsilon (see
    surfaces.SurfaceComponent). Its vertical structure is the layer of the
    component (each term of the pdf of its height; razor-thin for a thin
    disk), unless another is given: a law (e.g. sech2, of the scale height
    hz) or the layer of another disk component (e.g. molecular gas from a
    CO surface density, in the layer of the HI). In its own layer, its
    density is the sum over the terms of the component of their surface
    density times the pdf of their height, at the subnodes, and linear in
    R between them, as its gravity takes it (so that its density, its
    mass and its forces are those of one disk).

    Parameters
    ----------
    name : str
        Its name (see MassComponent).
    component : str
        The name of the disk component whose density it traces.
    rstep : float, optional
        The width (in the units of the rings of the component, e.g.
        arcsec) of the rings at whose subnodes it takes the density; by
        default, the subnodes of the component.
    vertical : DiskVertical, optional
        Its vertical structure; by default, the layer of the component.

    Raises
    ------
    ValueError
        If the name, the component or rstep is not valid, or vertical is
        the layer of the component itself (the default: leave it out).
    """

    @staticmethod
    def type() -> str:
        return 'traced_disk'

    @staticmethod
    def type_title() -> str:
        return 'Traced disk'

    def dump(self) -> dict[str, Any]:
        info = MassComponent.dump(self) | dict(component=self._component)
        if self._rstep is not None:
            info.update(rstep=self._rstep)
        if not self._own_layer:
            info.update(vertical=disk_vertical_parser.dump(self.vertical()))
        return info

    def __init__(
            self,
            name: str,
            component: str,
            rstep: Annotated[float, Gt(0)] | None = None,
            vertical: DiskVertical | None = None
    ):
        check_options(TracedDisk.__init__, locals())
        check_name(component)
        if isinstance(vertical, VerticalComponent) \
                and vertical.component() == component:
            raise ValueError(
                f"a traced disk is in the layer of its component "
                f"{component!r} unless another vertical structure is "
                f"given: leave vertical out")
        self._component = component
        self._rstep = rstep
        self._own_layer = vertical is None
        surface = SurfaceComponent(component, rstep)
        self._init(
            name, surface,
            VerticalComponent(component, rstep) if vertical is None
            else vertical)

    def component(self) -> str:
        """Return the name of the component whose density it traces."""
        return self._component

    def density(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        if not self._own_layer:
            return super().density(R, z, params, context)
        # (each term of the component, its surface density times the pdf
        # of its height, summed at the subnodes)
        density, knots, sigmas, upsilon = self._knots(params, context)

        def at(k: int, heights: np.ndarray) -> np.ndarray:
            pdfs = density.height_pdfs(knots[k:k + 1], heights)
            return sigmas[:, k] @ pdfs[:, 0]
        R, z = np.broadcast_arrays(
            np.asarray(R, dtype=float), np.asarray(z, dtype=float))
        return upsilon * _linear_between(R, z, knots, at)

    def _column_within(
            self,
            R: np.ndarray,
            h: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        if not self._own_layer:
            return super()._column_within(R, h, params, context)
        # (that of its density: of each subnode by adaptive quadrature, to
        # 1e-10, and linear between them; the pdfs are even in z)
        density, knots, sigmas, upsilon = self._knots(params, context)

        def at(k: int, heights: np.ndarray) -> np.ndarray:
            if not np.any(sigmas[:, k]):
                return np.zeros(heights.shape)

            def column(z: float) -> float:
                pdfs = density.height_pdfs(knots[k:k + 1], np.array([z]))
                return float(sigmas[:, k] @ pdfs[:, 0, 0])
            return np.array([
                2 * scipy.integrate.quad(
                    column, 0, height, epsabs=0, epsrel=1e-10,
                    limit=200)[0]
                for height in heights])
        R, h = np.broadcast_arrays(
            np.asarray(R, dtype=float), np.asarray(h, dtype=float))
        return upsilon * _linear_between(R, h, knots, at)

    def _knots(
            self, params: ParamValues, context: MassContext
    ) -> tuple[DiskDensity, np.ndarray, np.ndarray, float]:
        """
        Return the density of its component, the subnodes where it takes
        it, the surface density of each term there (terms, subnodes) and
        upsilon, its mass per unit of it (with that of a unit of the
        density).
        """
        density = context.density(self._component)
        knots = density.knots(self._rstep)
        return (density, knots, density.surface_densities(knots),
                self.surface().upsilon(params, context))


def _linear_between(
        R: np.ndarray,
        points: np.ndarray,
        knots: np.ndarray,
        at
) -> np.ndarray:
    """
    Return a quantity at the radii R, linear in R between the knots and 0
    beyond the first and the last: at(k, points) gives it at the knot k
    for the given points (e.g. heights) of the radii near it.
    """
    out = np.zeros(R.shape)
    inside = (R >= knots[0]) & (R <= knots[-1])
    left = np.clip(np.searchsorted(knots, R, side='right') - 1,
                   0, len(knots) - 2)
    t = (R - knots[left]) / (knots[left + 1] - knots[left])
    for k in np.unique(left[inside]):
        near = inside & (left == k)
        out[near] = (1 - t[near]) * at(k, points[near]) \
            + t[near] * at(k + 1, points[near])
    return out
