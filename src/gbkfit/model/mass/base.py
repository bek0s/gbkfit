"""
The contract of the components of mass models (MassComponent), and the
spherical components (MassSpherical).
"""

import abc
import dataclasses
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import check_options
from .units import G

if TYPE_CHECKING:
    from .profiles import SpheroidProfile
    from .shapes import SpheroidShape


__all__ = [
    'ComponentDensity',
    'DiskDensity',
    'MassComponent',
    'MassContext',
    'MassSpherical',
    'SpheroidDensity',
    'forces_in_plane'
]


def forces_in_plane(
        R: npt.ArrayLike,
        z: npt.ArrayLike,
        vcirc2: Callable[[np.ndarray], np.ndarray],
        label: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the forces per unit mass ((km/s)^2/kpc) in the plane of a mass
    (labelled label in errors), -vcirc2(R) / R and 0, at the points (R,
    z), broadcast together.

    Raises
    ------
    NotImplementedError
        For points off its plane.
    """
    R, z = np.broadcast_arrays(
        np.asarray(R, dtype=float), np.asarray(z, dtype=float))
    if np.any(z != 0):
        raise NotImplementedError(
            f"{label} gives its forces in its plane (z = 0) only")
    force_R = -np.divide(
        vcirc2(R), R, out=np.zeros_like(R), where=R > 0)
    return force_R, np.zeros_like(R)


class ComponentDensity(Protocol):
    """
    The density of a component of a galaxy, which mass components follow
    (see MassComponent.followed_kind), in the units of the component: of
    one kind, that of a disk (DiskDensity) or of a spheroid
    (SpheroidDensity).
    """

    def mass_per_unit(self) -> float | None:
        """
        Return the mass surface density (Msun/kpc^2) of one unit of the
        density, if the component has a physical unit (e.g. of the HI in
        it); None without one.
        """
        ...


class DiskDensity(ComponentDensity, Protocol):
    """
    The density of a disk of a galaxy, which mass disks follow (see
    surfaces.SurfaceComponent and vertical.VerticalComponent): its
    azimuthal mean, a sum of terms (e.g. one for each density polar
    trait), each a surface density times the pdf of the height, between
    the inner and the outer edge of the component; in kpc.
    """

    def knots(self, rstep: float | None) -> np.ndarray:
        """
        Return the radii (kpc) of the rings of the component, from its
        inner to its outer edge: those it is evaluated on, or rings of
        width at most rstep (in the units of its rings, e.g. arcsec).
        """
        ...

    def thick(self) -> bool:
        """Check whether its terms have pdfs of the height (thick)."""
        ...

    def surface_densities(self, R: np.ndarray) -> np.ndarray:
        """
        Return the surface density of each term at the radii R (kpc,
        between its edges): an array of shape (terms, len(R)).
        """
        ...

    def height_pdfs(self, R: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Return the pdf of the height (1/kpc) of each term at the radii R
        (kpc, between its edges) and the heights z (kpc): an array of shape
        (terms, len(R), len(z)).
        """
        ...


class SpheroidDensity:
    """
    The density of a spheroid of a galaxy, which mass spheroids follow
    (see spheroids.MassLightSpheroid): a sum of terms, each a spherical
    profile laid out in a shape, in kpc, of the total mass of its light
    (in the units of the component times kpc^2: its density times kpc^2
    gives the mass per unit of the light, see mass_per_unit).

    Parameters
    ----------
    terms : Sequence of tuple
        Its terms: the shape (SpheroidShape), the profile (SpheroidProfile)
        and the values of the parameters of both of each.
    mass_per_unit : float, optional
        The mass surface density (Msun/kpc^2) of one unit of the density;
        none without a physical unit.
    """

    def __init__(
            self,
            terms: Sequence[tuple['SpheroidShape', 'SpheroidProfile',
                                  ParamValues]],
            mass_per_unit: float | None
    ):
        self._terms = tuple(terms)
        self._mass_per_unit = mass_per_unit

    def terms(self) -> tuple[tuple[
            'SpheroidShape', 'SpheroidProfile', ParamValues], ...]:
        """Return its terms."""
        return self._terms

    def mass_per_unit(self) -> float | None:
        return self._mass_per_unit


@dataclasses.dataclass(frozen=True)
class MassContext:
    """
    What mass components need besides their parameters: the critical
    density of the universe (Msun/kpc^3) and its matter density
    parameter, at the epoch of the galaxy (for the overdensities that
    define the masses of halos), the densities of the components of the
    galaxy that they follow, by the names of the components, and the
    squared circular velocities of the mass components that they respond
    to (see MassComponent.responds_to), as functions of the radius (kpc),
    by their names, and their masses within spheres (Msun) of the radii
    (kpc), of those that give them (see MassComponent.enclosed_mass); the
    cosmic baryon fraction Omega_b / Omega_m, if
    known (for the contractions of halos that take it); the Ricci
    scalar of the background universe (kpc^-2) at the epoch of the galaxy
    and today, if known (for f(R) gravity); and the length of an arcsec in
    the galaxy (kpc), if known (for tables measured at a distance, see
    MassTable).
    """
    critical_density: float
    omega_m: float
    densities: Mapping[str, ComponentDensity] = dataclasses.field(
        default_factory=dict)
    vcirc2s: Mapping[str, Callable[[np.ndarray], np.ndarray]] = \
        dataclasses.field(default_factory=dict)
    enclosed_masses: Mapping[str, Callable[[np.ndarray], np.ndarray]] = \
        dataclasses.field(default_factory=dict)
    baryon_fraction: float | None = None
    ricci_scalar: float | None = None
    ricci_scalar_today: float | None = None
    kpc_per_arcsec: float | None = None

    def density(self, component: str) -> ComponentDensity:
        """
        Return the density of a component that mass components follow, by
        the name of the component.

        Raises
        ------
        ValueError
            If the context does not give the density of the component.
        """
        if component not in self.densities:
            raise ValueError(
                f"a mass component follows the component {component!r}, "
                f"whose density its context does not give")
        return self.densities[component]


class MassComponent(parseutils.TypedSerializable, abc.ABC):
    """
    A component of a mass model (e.g. the stars, the gas or the dark
    matter halo of a galaxy), axisymmetric about the z axis: R is the
    cylindrical radius and z the height above its plane (see units for
    the units).

    A component computes what it can exactly, or numerically to a known
    precision. What it cannot compute (yet) raises NotImplementedError,
    never an approximation (e.g. a disk treated as a sphere).

    Parameters
    ----------
    name : str
        Its name, which prefixes its parameters in the mass model.

    Raises
    ------
    ConfigError
        If the name is not valid (see parseutils.check_name).
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type(), name=self._name)

    def __init__(self, name: str):
        check_options(MassComponent.__init__, locals())
        parseutils.check_name(name)
        self._name = name

    def name(self) -> str:
        """Return its name."""
        return self._name

    def followed_components(self) -> tuple[str, ...]:
        """
        Return the names of the components of its galaxy whose densities
        it follows (see ComponentDensity); none here.
        """
        return ()

    @classmethod
    def followed_kind(cls) -> type[ComponentDensity] | None:
        """
        Return the kind of the densities that it follows (e.g.
        DiskDensity), if it can follow any; none here.
        """
        return None

    def spherical(self) -> bool:
        """Check whether it is spherically symmetric; not here."""
        return False

    def responds_to(self) -> tuple[str, ...]:
        """
        Return the names of the mass components of its model whose
        gravity it responds to (e.g. a halo contracted by the baryons),
        whose squared circular velocities and masses within spheres its
        context gives (see MassContext); none here.
        """
        return ()

    def takes_enclosed_masses(self) -> bool:
        """
        Check whether it takes the masses within spheres of the components
        it responds to (e.g. a halo contracted by them), which they must
        give (see gives_enclosed_mass); here, it does not.
        """
        return False

    @abc.abstractmethod
    def pdescs(self) -> dict[str, ParamDesc]:
        """Return its parameters, by name."""
        pass

    def mass_ratios(self) -> dict[str, str]:
        """
        Return its parameters that are its mass per unit of the density
        of a component of its galaxy that it follows (e.g. upsilon), each
        with the name of the component: their unit depends on that of the
        density, which its galaxy knows and sets (their descriptions have
        none); none here.
        """
        return {}

    @abc.abstractmethod
    def vcirc2(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return the square of its circular velocity in its plane, R dPhi/dR
        at z = 0: signed (negative where it pushes outwards, e.g. inside
        a ring).

        Parameters
        ----------
        R : array_like
            The radii (kpc).
        params : dict
            The values of its parameters, by their names in pdescs.
        context : MassContext
            The epoch of the galaxy, and the densities it follows (see
            MassContext).

        Returns
        -------
        np.ndarray
            The squared circular velocity ((km/s)^2) at each radius.
        """
        pass

    @abc.abstractmethod
    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its mass (Msun) within the spheres of the radii r (kpc)
        about its centre (see vcirc2 for the other arguments): e.g. of a
        disk, its mass within the heights |z| < sqrt(r^2 - R^2) at each
        radius R < r (all of it, if thin).

        Raises
        ------
        NotImplementedError
            If the component does not give it (see gives_enclosed_mass).
        """
        pass

    def gives_enclosed_mass(self) -> bool:
        """Check whether it gives its enclosed mass; here, it does."""
        return True

    def density(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its density (Msun/kpc^3) at the points (R, z), broadcast
        together (see vcirc2 for the other arguments).

        Raises
        ------
        NotImplementedError
            If the component does not give its density (see
            gives_density).
        """
        raise NotImplementedError(
            f"the mass component '{self.type()}' does not give its density")

    def gives_density(self, thick: Collection[str]) -> bool:
        """
        Check whether it gives its density, in a galaxy whose thick
        components (those whose densities have pdfs of the height, see
        DiskDensity.thick) have the given names; here, it does not.
        """
        return False

    def cell_masses(
            self,
            r_edges: np.ndarray,
            b_edges: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its mass (Msun) in each cell between the spheres of the
        radii r_edges (kpc) and the latitudes b_edges (radians, from its
        plane), at both sides of the plane, of shape (len(r_edges) - 1,
        len(b_edges) - 1) (see vcirc2 for the other arguments). By default,
        of its density at 2 x 2 points of each cell at either side, by
        Gauss-Legendre quadrature in r^3 and sin(b), in which the volume is
        uniform.

        Raises
        ------
        NotImplementedError
            If the component does not give them (see gives_cell_masses).
        """
        r_edges = np.asarray(r_edges, dtype=float)
        b_edges = np.asarray(b_edges, dtype=float)
        nodes = np.array([-1, 1]) / np.sqrt(3)

        def points(edges: np.ndarray) -> np.ndarray:
            middle, half = (edges[1:] + edges[:-1]) / 2, np.diff(edges) / 2
            return middle[:, None] + half[:, None] * nodes

        r = np.cbrt(points(r_edges ** 3))[:, None, :, None]
        b = np.arcsin(points(np.sin(b_edges)))[None, :, None, :]
        R, z = r * np.cos(b), r * np.sin(b)
        density = (self.density(R, z, params, context)
                   + self.density(R, -z, params, context)) / 2
        volumes = 4 * np.pi / 3 * np.outer(
            np.diff(r_edges ** 3), np.diff(np.sin(b_edges)))
        return density.mean(axis=(2, 3)) * volumes

    def gives_cell_masses(self, thick: Collection[str]) -> bool:
        """
        Check whether it gives its mass in cells (see cell_masses), in a
        galaxy whose thick components have the given names (see
        gives_density); by default, if it gives its density.
        """
        return self.gives_density(thick)

    def surface_density(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its surface density (Msun/kpc^2), seen face-on, at the radii
        R (see vcirc2 for the other arguments).

        Raises
        ------
        NotImplementedError
            If the component does not give its surface density.
        """
        raise NotImplementedError(
            f"the mass component '{self.type()}' does not give its surface "
            f"density")

    def forces(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return its forces per unit mass ((km/s)^2/kpc), -dPhi/dR and
        -dPhi/dz, at the points (R, z), broadcast together (see vcirc2
        for the other arguments). By default, in its plane only, where
        they are -vcirc2 / R and 0.

        Raises
        ------
        NotImplementedError
            For points off its plane, if the component does not give its
            forces there (see gives_forces).
        """
        return forces_in_plane(
            R, z, lambda radii: self.vcirc2(radii, params, context),
            f"the mass component '{self.type()}'")

    def gives_forces(self) -> bool:
        """
        Check whether it gives its forces off its plane (see forces and
        vertical_force); here, it does.
        """
        return True

    def takes_shares(self) -> bool:
        """
        Check whether it takes the shares of the components in the squared
        circular velocity of the mass (e.g. a spheroid given by its dark
        fraction, see DarkFraction), which the gravity of its galaxy must
        give (see Gravity.gives); here, it does not.
        """
        return False

    def vertical_force(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its vertical force per unit mass ((km/s)^2/kpc), -dPhi/dz,
        at the points (R, z), broadcast together (see vcirc2 for the other
        arguments): by default, that of its forces (see forces).

        Raises
        ------
        NotImplementedError
            For points off its plane, if the component does not give its
            forces there (see gives_forces).
        """
        return self.forces(R, z, params, context)[1]

    def potential(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its gravitational potential ((km/s)^2), 0 at infinity, at
        the points (R, z), broadcast together (see vcirc2 for the other
        arguments).

        Raises
        ------
        NotImplementedError
            If the component does not give its potential.
        """
        raise NotImplementedError(
            f"the mass component '{self.type()}' does not give its "
            f"potential")


class MassSpherical(MassComponent, abc.ABC):
    """
    A spherical mass component, given by its density and its enclosed
    mass, as functions of the spherical radius r: its circular velocity
    and its forces follow exactly.
    """

    def spherical(self) -> bool:
        return True

    def gives_density(self, thick: Collection[str]) -> bool:
        return True

    @abc.abstractmethod
    def density_spherical(
            self,
            r: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """Return its density (Msun/kpc^3) at the spherical radii r."""
        pass

    def cell_masses(
            self,
            r_edges: np.ndarray,
            b_edges: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        # (of its enclosed mass: the share of a shell between the latitudes
        # b1 < |b| < b2 is sin(b2) - sin(b1))
        return np.outer(
            np.diff(self.enclosed_mass(r_edges, params, context)),
            np.diff(np.sin(b_edges)))

    def vcirc2(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        # G M(<r) / r, 0 at the centre
        R = np.asarray(R, dtype=float)
        return G * np.divide(
            self.enclosed_mass(R, params, context), R,
            out=np.zeros_like(R), where=R > 0)

    def density(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        return self.density_spherical(np.hypot(R, z), params, context)

    def forces(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> tuple[np.ndarray, np.ndarray]:
        # -G M(<r) / r^3 (R, z), 0 at the centre
        R, z = np.broadcast_arrays(
            np.asarray(R, dtype=float), np.asarray(z, dtype=float))
        r = np.hypot(R, z)
        scale = -G * np.divide(
            self.enclosed_mass(r, params, context), r ** 3,
            out=np.zeros_like(r), where=r > 0)
        return scale * R, scale * z
