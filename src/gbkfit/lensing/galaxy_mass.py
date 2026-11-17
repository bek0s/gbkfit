"""
The piece of a lens that is the mass of a galaxy (LensGalaxyMass): the
light that passes the plane of its lens is bent by the mass model of a
galaxy there, so that one set of mass parameters sets both the rotation
of the galaxy and its lensing.
"""

from typing import Any

from gbkfit.config import check_name, check_options
from gbkfit.model import Capability, Need
from gbkfit.params import ParamScalarDesc
from .pieces import LensPiece, Normalisation


__all__ = [
    'LensGalaxyMass'
]


class LensGalaxyMass(LensPiece):
    """
    The mass of a galaxy at the plane of its lens (see OfGalaxy): it bends
    the light that passes the plane by the mass model of the galaxy, laid
    out on the sky by the galaxy in each evaluation (see GalaxyState.sky
    and SkyMass), which those who evaluate its lens give it (see
    GalaxyBending and MassBending): its
    mass components projected, each where the components of the galaxy
    place it (a part that follows a component takes the geometry of the
    component, ring by ring for a disk, warps included; the others lie in
    the plane of the mass at its one centre; see Galaxy.placed).

    Its deflection is the deflection angle itself (alpha-hat, D_ls / D_s
    = 1; see Normalisation), that of the convergence kappa = 4 pi G D_l
    Sigma / c^2 of the surface density Sigma of the mass on the sky, D_l
    the distance of the galaxy (its length of an arcsec, the distance its
    mass is built at): the distance of the plane of its lens enters only
    the factors D_ls / D_s of the planes behind (see Tracer), as with the
    other pieces. Each layer of the mass deflects by its integral (a
    point, a round layer by its projected mass within each radius, an
    elliptical layer by the integrals over the ellipses within each point,
    a band of rings in closed form; see SkyMass), never by a map of the
    mass cut at an edge.

    It has no parameters: those of the mass are those of the galaxy (its
    mass, distance and components). The galaxy must have mass and a
    distance, a gravity that bends light as its mass does (see
    Capability.LENSING: not MOND), mass components that lay out their
    mass on the sky (not tables or user components) and components that
    place it (see Galaxy.gives). It must be at the plane of its lens, and
    no other piece may be its mass (see ObservationGroup); it bends no
    light of its own plane (see Tracer).

    A shape beside it (e.g. an NFW halo in its lens) is mass that no model
    of a galaxy holds (e.g. the halo of a group): do not give the galaxy a
    shape of its own too, which would count its mass twice.

    Parameters
    ----------
    galaxy : str
        The name of the galaxy.
    name : str, optional
        Its name (see LensPiece; 'mass' by default).

    Raises
    ------
    ValueError
        If a name is not valid.
    """

    config_titles = dict(galaxy='Galaxy')
    config_references = dict(galaxy='galaxies')

    @staticmethod
    def type() -> str:
        return 'mass'

    @staticmethod
    def type_title() -> str:
        return 'Mass of a galaxy'

    def dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return super().dump() | dict(galaxy=self._galaxy)

    def __init__(self, galaxy: str, name: str | None = None):
        check_options(LensGalaxyMass.__init__, locals())
        super().__init__(name)
        check_name(galaxy)
        self._galaxy = galaxy

    def galaxy(self) -> str:
        """Return the name of its galaxy."""
        return self._galaxy

    def needs(self, label: str) -> tuple[Need, ...]:
        """
        Return what it needs of its galaxy (see OfGalaxy): its mass, its
        distance, and the bending of light by its mass (see
        Capability.LENSING), as the piece of the given label in messages.
        """
        galaxy = repr(self._galaxy)
        return (
            Need(Capability.MASS,
                 f"{label} is the mass of the galaxy {galaxy}, which has "
                 f"none"),
            Need(Capability.DISTANCE,
                 f"{label} bends light by the mass of the galaxy {galaxy} "
                 f"at its distance"),
            Need(Capability.LENSING,
                 f"{label} bends light by the mass of the galaxy {galaxy}"))

    def normalisation(self) -> Normalisation:
        return Normalisation.UNIT

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return {}
