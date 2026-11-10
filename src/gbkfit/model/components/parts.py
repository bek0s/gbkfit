"""
The traits of the components made of nodes (rings or shells along a
radius, e.g. disks), which they take as options of their own (see
traits): the checks of their density traits (density_traits), their
kinematic traits (KinematicTraits, see kinematic_traits), and the
vertical equilibrium of thick disks (Equilibrium).
"""

import abc
from collections.abc import Sequence
from typing import Annotated, Any, Literal, TypeAlias

from gbkfit.config import (
    Choices, TypedParser, TypedSerializable, check_name, check_options)
from gbkfit.utils import iterutils
from . import traits
from ._nodal import MAX_TRAITS


__all__ = [
    'CentralEquilibrium',
    'Equilibrium',
    'JeansEquilibrium',
    'KinematicTraits',
    'SelfGravitatingEquilibrium',
    'density_traits',
    'dump_traits',
    'equilibrium_parser',
    'kinematic_traits'
]


def _polar_traits(
        option: str,
        value: traits.PTrait | Sequence[traits.PTrait] | None
) -> tuple[traits.PTrait, ...]:
    """
    Return the polar traits of an option as a tuple. Raise ValueError if
    there are none or too many.
    """
    value = iterutils.tuplify(value)
    if not value:
        raise ValueError(f"at least one of {option} is required")
    if len(value) > MAX_TRAITS:
        raise ValueError(
            f"at most {MAX_TRAITS} {option} are supported; {len(value)} "
            f"were given")
    return value


def _height_traits(
        option: str,
        value: traits.HTrait | Sequence[traits.HTrait | None] | None,
        npolar: int,
        polar_option: str
) -> tuple[traits.HTrait | None, ...] | None:
    """
    Return the height traits of an option as a tuple (None if not given),
    in which an item can be None (a default trait): one that every polar
    trait shares (one trait, or a list of one), or one for each polar
    trait, in their order. Raise ValueError for another number.
    """
    if value is None:
        return None
    value = iterutils.tuplify(value)
    if len(value) not in (1, npolar):
        raise ValueError(
            f"{option} takes one trait, which every one of the {polar_option} "
            f"shares, or one for each of them ({npolar}); {len(value)} "
            f"were given")
    return value


def _check_rotation_at_heights(vptrait: traits.VPTrait) -> None:
    """
    Raise ValueError unless the velocity polar trait can pair with the
    velocity height trait 'mass': the trait 'mass', whose pressure support
    at the heights has the density it takes (see traits.VHTraitMass).
    """
    if not isinstance(vptrait, traits.VPTraitMass):
        raise ValueError(
            f"the velocity height trait 'mass' is the rotation in the "
            f"gravity at each height: it pairs with the velocity trait "
            f"'mass' only, not {vptrait.type()!r}")
    support = vptrait.from_galaxy().support
    if support is not None and support.type() == 'self_gravitating' \
            and support.mass() is None:
        raise ValueError(
            "the velocity height trait 'mass' takes the self_gravitating "
            "pressure support at each height in the sheet of the surface "
            "density of a mass component, of the scale height sigma^2 / "
            "(pi G Sigma): give the mass of the pressure support")


def _check_dispersion_directions(
        dptraits: tuple[traits.DPTrait, ...]
) -> None:
    """
    Raise ValueError unless every direction of the dispersion has a
    dispersion polar trait, its own or an isotropic one (see traits),
    once a trait is along a direction.
    """
    given = {trait.direction() for trait in dptraits}
    if traits.DPT_DIRECTION_DEFAULT in given:
        return
    for direction in traits.DPT_DIRECTIONS:
        if direction != traits.DPT_DIRECTION_DEFAULT \
                and direction not in given:
            raise ValueError(
                f"no dispersion trait for the direction {direction!r}; give "
                f"one, or an {traits.DPT_DIRECTION_DEFAULT!r} trait")


def dump_traits(
        parser: TypedParser,
        value: tuple[traits.Trait | None, ...] | None
) -> list[dict[str, Any] | None] | None:
    """Return the configuration of traits (None items as None)."""
    if value is None:
        return None
    return [parser.dump(trait) for trait in value]


def density_traits(
        rptraits: traits.RPTrait | Sequence[traits.RPTrait],
        rhtraits: traits.RHTrait | Sequence[traits.RHTrait] | None,
        thick: bool
) -> tuple[tuple[traits.RPTrait, ...], tuple[traits.RHTrait, ...]]:
    """
    Return the density traits of a component as tuples: its polar traits,
    the density per unit area of its surface (the plane of a disk, seen
    face-on), and of a thick one, its height traits, the probability
    density of the distance from the surface of each (none for a thin
    one).

    Raises
    ------
    ValueError
        If there are no polar traits or too many, or a thick component has
        neither one height trait that they share nor one for each.
    """
    rptraits = _polar_traits('rptraits', rptraits)
    if not thick:
        return rptraits, ()
    if rhtraits is None:
        raise ValueError(
            "a thick component needs the height traits of its density "
            "(rhtraits)")
    rhtraits = _height_traits(
        'rhtraits', rhtraits, len(rptraits), 'rptraits')
    if None in rhtraits:
        raise ValueError("rhtraits cannot have empty items")
    return rptraits, rhtraits


def kinematic_traits(
        vptraits: traits.VPTrait | Sequence[traits.VPTrait] | None,
        dptraits: traits.DPTrait | Sequence[traits.DPTrait] | None,
        vhtraits:
        traits.VHTrait | Sequence[traits.VHTrait | None] | None = None,
        dhtraits:
        traits.DHTrait | Sequence[traits.DHTrait | None] | None = None
) -> 'KinematicTraits | None':
    """
    Return the kinematic traits of a component (see KinematicTraits), or
    None if it has none: a component without velocity and dispersion
    traits does not move (e.g. one seen only by images).

    Raises
    ------
    ValueError
        If only one of vptraits and dptraits is given, height traits are
        given without them, or the traits are not valid (see
        KinematicTraits).
    """
    if vptraits is None and dptraits is None:
        if vhtraits is not None or dhtraits is not None:
            raise ValueError(
                "vhtraits and dhtraits are the heights of the velocity and "
                "dispersion traits: give vptraits and dptraits")
        return None
    if vptraits is None or dptraits is None:
        raise ValueError(
            "a component that moves takes both its velocity traits "
            "(vptraits) and its dispersion traits (dptraits)")
    return KinematicTraits(vptraits, dptraits, vhtraits, dhtraits)


# The directions of the dispersion of a vertical equilibrium (see
# Equilibrium)
EQUILIBRIUM_DIRECTIONS = ('iso', 'ver')
EQUILIBRIUM_DIRECTION_DEFAULT = 'iso'
EquilibriumDirection: TypeAlias = Annotated[
    Literal[EQUILIBRIUM_DIRECTIONS],
    Choices(iso='Isotropic', ver='Vertical (sigma_z)')]


class KinematicTraits:
    """
    How a component moves as its traits give: the velocity and the
    velocity dispersion of each of its points (see traits). The velocity
    traits add up, so motions combine; the rotation of a disk in the
    gravity of the galaxy is the velocity trait 'mass' (see
    traits.VPTraitMass). Disks, cones and spheroids of traits take them
    as options of their own (see kinematic_traits).

    Parameters
    ----------
    vptraits : VPTrait or Sequence of VPTrait
        The velocity polar traits (parameters 'vpt_*').
    dptraits : DPTrait or Sequence of DPTrait
        The velocity dispersion polar traits (parameters 'dpt_*'),
        isotropic or each along a direction (see traits): once one is
        along a direction, every direction needs its own or an isotropic
        one.
    vhtraits : VHTrait or Sequence of VHTrait, optional
        The velocity height traits of a thick component: one that every
        velocity polar trait shares (parameters 'vht_*'), or one for each
        (None items or none at all: the velocity does not change with the
        distance from the surface).
    dhtraits : DHTrait or Sequence of DHTrait, optional
        The dispersion height traits of a thick component, one shared or
        one for each dispersion polar trait (as vhtraits).

    Raises
    ------
    ValueError
        If there are no polar traits or too many, neither one shared
        height trait nor one for each polar trait, a direction of the dispersion without a
        dispersion trait, several velocity traits 'mass' (the rotation in
        the gravity would count more than once), or a velocity height
        trait 'mass' that does not pair with the velocity trait 'mass' or
        takes its self_gravitating pressure support without its mass (see
        traits.VHTraitMass).
    """

    def dump(self) -> dict[str, Any]:
        """Return the options of the traits."""
        info = dict(
            vptraits=dump_traits(traits.vpt_parser, self._vptraits),
            dptraits=dump_traits(traits.dpt_parser, self._dptraits))
        if self._vhtraits is not None:
            info['vhtraits'] = dump_traits(traits.vht_parser, self._vhtraits)
        if self._dhtraits is not None:
            info['dhtraits'] = dump_traits(traits.dht_parser, self._dhtraits)
        return info

    def __init__(
            self,
            vptraits: traits.VPTrait | Sequence[traits.VPTrait],
            dptraits: traits.DPTrait | Sequence[traits.DPTrait],
            vhtraits:
            traits.VHTrait | Sequence[traits.VHTrait | None] | None = None,
            dhtraits:
            traits.DHTrait | Sequence[traits.DHTrait | None] | None = None
    ):
        self._vptraits = _polar_traits('vptraits', vptraits)
        if sum(isinstance(trait, traits.VPTraitMass)
               for trait in self._vptraits) > 1:
            raise ValueError(
                "a disk takes at most one velocity trait 'mass': each "
                "adds the whole rotation in the gravity")
        self._dptraits = _polar_traits('dptraits', dptraits)
        _check_dispersion_directions(self._dptraits)
        self._vhtraits = _height_traits(
            'vhtraits', vhtraits, len(self._vptraits), 'vptraits')
        # (a shared height trait pairs with every velocity trait)
        vhtraits = self._vhtraits or ()
        if len(vhtraits) == 1:
            vhtraits *= len(self._vptraits)
        for vptrait, vhtrait in zip(self._vptraits, vhtraits):
            if isinstance(vhtrait, traits.VHTraitMass):
                _check_rotation_at_heights(vptrait)
        self._dhtraits = _height_traits(
            'dhtraits', dhtraits, len(self._dptraits), 'dptraits')

    def vptraits(self) -> tuple[traits.VPTrait, ...]:
        """Return the velocity polar traits."""
        return self._vptraits

    def dptraits(self) -> tuple[traits.DPTrait, ...]:
        """Return the velocity dispersion polar traits."""
        return self._dptraits

    def vhtraits(self) -> tuple[traits.VHTrait | None, ...] | None:
        """Return the velocity height traits, if given."""
        return self._vhtraits

    def dhtraits(self) -> tuple[traits.DHTrait | None, ...] | None:
        """Return the velocity dispersion height traits, if given."""
        return self._dhtraits


class Equilibrium(TypedSerializable, abc.ABC):
    """
    The vertical equilibrium of a thick disk (see
    ThickDiskTraitsKinematics): the disk is held up against the gravity of
    its galaxy by its velocity dispersion. Its thickness is that of its
    density (its density height traits); the equilibrium gives the
    dispersion that holds it up, sigma_eq(R), which adds to that of its
    dispersion traits in quadrature, sigma = hypot(sigma_traits,
    sigma_eq), at every height
    (the dispersion of the midplane: isothermal in z). It is evaluated at
    the subnodes of the disk (its rings), from the azimuthal means of its
    traits, and interpolated linearly between them. The dispersion of the
    equilibrium is isotropic (direction 'iso', the default: it adds to
    the dispersion along every direction) or vertical ('ver': it adds to
    sigma_z only, the dispersion that holds the disk up; see traits for
    the directions of the dispersion).

    Parameters
    ----------
    direction : str, optional
        The direction of its dispersion: 'iso' (the default), every
        direction, or 'ver', z only.

    Raises
    ------
    ValueError
        If the direction is not one of those.
    """

    def dump(self) -> dict[str, Any]:
        info = dict(type=self.type())
        if self._direction != EQUILIBRIUM_DIRECTION_DEFAULT:
            info['direction'] = self._direction
        return info

    def __init__(
            self,
            direction: EquilibriumDirection = EQUILIBRIUM_DIRECTION_DEFAULT
    ):
        check_options(Equilibrium.__init__, locals())
        self._direction = direction

    def direction(self) -> str:
        """Return the direction of its dispersion: 'iso' or 'ver'."""
        return self._direction


class CentralEquilibrium(Equilibrium):
    """
    The vertical equilibrium in the pull of a spherical mass: sigma_eq =
    h V_c / R, of the rms height h and the circular velocity V_c. At the
    radius R, h is the rms height of the disk (that of its density, from
    its density height traits), and V_c the circular velocity of the mass
    of its galaxy, or without mass, the rotation of the disk (the
    azimuthal mean of its tangential velocity traits at the midplane). It
    is exact for a disk thin against its radius (h << R) in a spherical
    gravity, whatever the shape of its height profile (the vertical virial
    theorem: sigma^2 = Omega^2 <z^2>, Omega = V_c / R). At the centre,
    where V_c / R is 0 / 0, it takes Omega at the next subnode; NaN where
    V_c^2 is negative. It is the thick disk of GalPaK3D, there with h a
    fixed share of the radius of the disk. See Equilibrium.
    """

    @staticmethod
    def type() -> str:
        return 'central'

    @staticmethod
    def type_title() -> str:
        return 'Spherical mass'


class SelfGravitatingEquilibrium(Equilibrium):
    """
    The vertical equilibrium in the pull of the disk itself: sigma_eq^2 =
    2 sqrt(3) G Sigma h, of its surface density Sigma and rms height h.
    Sigma is the surface density of a mass component of its galaxy (e.g. a
    mass disk that follows the disk); it is exact for an isothermal slab
    of its own gravity, of the height profile sech^2(z / z0), z0 = sigma^2
    / (pi G Sigma) (<z^2> = pi^2 z0^2 / 12); NaN where Sigma is negative.
    See Equilibrium.

    Parameters
    ----------
    mass : str
        The name of the mass component whose surface density holds the
        disk up.
    direction : str, optional
        As that of Equilibrium.

    Raises
    ------
    ValueError
        If the name or the direction is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'self_gravitating'

    @staticmethod
    def type_title() -> str:
        return 'Self-gravitating disk'

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(mass=self._mass)

    def __init__(
            self,
            mass: str,
            direction: EquilibriumDirection = EQUILIBRIUM_DIRECTION_DEFAULT
    ):
        check_options(SelfGravitatingEquilibrium.__init__, locals())
        super().__init__(direction)
        check_name(mass)
        self._mass = mass

    def mass(self) -> str:
        """
        Return the name of the mass component whose surface density holds
        the disk up.
        """
        return self._mass


class JeansEquilibrium(Equilibrium):
    """
    The vertical equilibrium in the pull of all the mass of its galaxy, by
    the vertical Jeans equation. Its dispersion is sigma_eq^2 = int_0^inf
    z rho(z) (-F_z(R, z)) dz / int_0^inf rho(z) dz, the dispersion of the
    column weighted by its density (the vertical virial theorem, Binney &
    Tremaine 2008, eq. 4.222; rho of its density traits and density
    height traits), with F_z the vertical force of every mass component
    (see mass.MassComponent.vertical_force: user components give it by
    their function of the forces); it is exact for any mass and profile of
    the height, as the dispersion a face-on column shows. As 'central' and
    'self_gravitating' in their limits (a harmonic potential, the sheet
    alone): the dispersion of the midplane (int rho (-F_z) dz / rho(0)) is
    that of a profile isothermal in its own potential only, 1 / sqrt(2) of
    this for an exponential one (audit 2026-10-31 #13). At the centre, the
    value at the next subnode, as for 'central' (a point mass there makes
    it infinite). See Equilibrium.
    """

    @staticmethod
    def type() -> str:
        return 'jeans'

    @staticmethod
    def type_title() -> str:
        return 'Jeans equation'


equilibrium_parser = TypedParser(Equilibrium, [
    CentralEquilibrium,
    SelfGravitatingEquilibrium,
    JeansEquilibrium], family='equilibrium')
