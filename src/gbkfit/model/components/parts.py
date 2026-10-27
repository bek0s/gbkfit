"""
The density and the kinematics of the components made of nodes (rings or
shells along a radius, e.g. disks), each a set of traits (see traits),
and the vertical equilibrium of disks.
"""

import abc
from collections.abc import Sequence
from typing import Any, Self

from gbkfit.utils import iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from . import traits
from ._nodal import MAX_TRAITS


__all__ = [
    'CentralEquilibrium',
    'ConeTraitsKinematics',
    'Density',
    'DiskTraitsKinematics',
    'Equilibrium',
    'JeansEquilibrium',
    'KinematicTraits',
    'SelfGravitatingEquilibrium',
    'cone_kinematics_parser',
    'density_parser',
    'disk_kinematics_parser',
    'equilibrium_parser'
]


def _polar_traits(
        option: str,
        value: traits.PTrait | Sequence[traits.PTrait] | None
) -> tuple[traits.PTrait, ...]:
    """
    Return the polar traits of an option as a tuple. Raise ConfigError if
    there are none or too many.
    """
    value = iterutils.tuplify(value)
    if not value:
        raise ConfigError(f"at least one of {option} is required")
    if len(value) > MAX_TRAITS:
        raise ConfigError(
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
    in which an item can be None (a default trait). Raise ConfigError if
    there is not one for each polar trait.
    """
    if value is None:
        return None
    value = iterutils.tuplify(value)
    if len(value) != npolar:
        raise ConfigError(
            f"the number of {option} must be equal to the number of "
            f"{polar_option} ({len(value)} != {npolar})")
    return value


def _check_rotation_at_heights(vptrait: traits.VPTrait) -> None:
    """
    Raise ConfigError unless the velocity polar trait can pair with the
    velocity height trait 'mass': the trait 'mass', whose pressure support
    at the heights has the density it takes (see traits.VHTraitMass).
    """
    if not isinstance(vptrait, traits.VPTraitMass):
        raise ConfigError(
            f"the velocity height trait 'mass' is the rotation in the "
            f"gravity at each height: it pairs with the velocity trait "
            f"'mass' only, not {vptrait.type()!r}")
    support = vptrait.from_galaxy().support
    if support is not None and support.mode == 'self_gravitating' \
            and support.density is None:
        raise ConfigError(
            "the velocity height trait 'mass' takes the self_gravitating "
            "pressure support at each height in the sheet of the surface "
            "density of a mass component, of the scale height sigma^2 / "
            "(pi G Sigma): give pressure_density")


def _check_dispersion_directions(
        dptraits: tuple[traits.DPTrait, ...]
) -> None:
    """
    Raise ConfigError unless every direction of the dispersion has a
    dispersion polar trait, its own or an isotropic one (see traits),
    once a trait is along a direction.
    """
    given = {trait.direction() for trait in dptraits}
    if traits.DPT_DIRECTION_DEFAULT in given:
        return
    for direction in traits.DPT_DIRECTIONS:
        if direction != traits.DPT_DIRECTION_DEFAULT \
                and direction not in given:
            raise ConfigError(
                f"no dispersion trait for the direction {direction!r}; give "
                f"one, or an {traits.DPT_DIRECTION_DEFAULT!r} trait")


def _dump_traits(
        parser: parseutils.TypedParser,
        value: tuple[traits.Trait | None, ...] | None
) -> list[dict[str, Any] | None] | None:
    """Return the configuration of traits (None items as None)."""
    if value is None:
        return None
    return [parser.dump(trait) for trait in value]


class Density(parseutils.Serializable):
    """
    The density of a component: its polar traits, the density per unit
    area of its surface (the plane of a disk, seen face-on), and for a
    thick component its height traits, the probability density of the
    distance from the surface of each (see traits). What the density is
    depends on the roles of the component: light for one that emits, the
    optical depth for dust.

    Parameters
    ----------
    ptraits : RPTrait or Sequence of RPTrait
        The polar traits (parameters 'pt_*').
    htraits : RHTrait or Sequence of RHTrait, optional
        The height traits, one for each polar trait (parameters 'ht_*');
        a component without them is thin.

    Raises
    ------
    ConfigError
        If there are no polar traits or too many, or not one height trait
        for each polar trait.
    """

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        parseutils.load_option_and_update_info(
            traits.rpt_parser, info, 'ptraits', required=True)
        parseutils.load_option_and_update_info(
            traits.rht_parser, info, 'htraits')
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        info = dict(ptraits=_dump_traits(traits.rpt_parser, self._ptraits))
        if self._htraits is not None:
            info['htraits'] = _dump_traits(traits.rht_parser, self._htraits)
        return info

    def __init__(
            self,
            ptraits: traits.RPTrait | Sequence[traits.RPTrait],
            htraits: traits.RHTrait | Sequence[traits.RHTrait] | None = None
    ):
        self._ptraits = _polar_traits('ptraits', ptraits)
        self._htraits = _height_traits(
            'htraits', htraits, len(self._ptraits), 'ptraits')
        if self._htraits is not None and None in self._htraits:
            raise ConfigError("htraits cannot have empty items")

    def ptraits(self) -> tuple[traits.RPTrait, ...]:
        """Return the polar traits."""
        return self._ptraits

    def htraits(self) -> tuple[traits.RHTrait, ...] | None:
        """Return the height traits, if any."""
        return self._htraits

    def thick(self) -> bool:
        """Check whether it has height traits."""
        return self._htraits is not None


# The directions of the dispersion of a vertical equilibrium (see
# Equilibrium)
EQUILIBRIUM_DIRECTIONS = ('iso', 'ver')
EQUILIBRIUM_DIRECTION_DEFAULT = 'iso'


class KinematicTraits:
    """
    How a component moves as its traits give: the velocity and the
    velocity dispersion of each of its points (see traits). The velocity
    traits add up, so motions combine; the rotation of a disk in the
    gravity of the galaxy is the velocity trait 'mass' (see
    traits.VPTraitMass). The traits of the kinematics of traits of disks,
    cones and spheroids (see DiskTraitsKinematics, ConeTraitsKinematics
    and spheroids.SpheroidTraitsKinematics).

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
        The velocity height traits of a thick component, one for each
        velocity polar trait (None items or none at all: the velocity does
        not change with the distance from the surface).
    dhtraits : DHTrait or Sequence of DHTrait, optional
        The dispersion height traits of a thick component, one for each
        dispersion polar trait (as vhtraits).

    Raises
    ------
    ConfigError
        If there are no polar traits or too many, not one height trait
        for each polar trait, a direction of the dispersion without a
        dispersion trait, several velocity traits 'mass' (the rotation in
        the gravity would count more than once), or a velocity height
        trait 'mass' that does not pair with the velocity trait 'mass' or
        takes its self_gravitating pressure support without
        pressure_density (see traits.VHTraitMass).
    """

    @staticmethod
    def load_options(info: dict[str, Any]) -> None:
        """Load the traits of the options of kinematics, in place."""
        for parser, key, required in (
                (traits.vpt_parser, 'vptraits', True),
                (traits.dpt_parser, 'dptraits', True),
                (traits.vht_parser, 'vhtraits', False),
                (traits.dht_parser, 'dhtraits', False)):
            parseutils.load_option_and_update_info(
                parser, info, key, required=required)

    def dump(self) -> dict[str, Any]:
        """Return the options of the traits."""
        info = dict(
            vptraits=_dump_traits(traits.vpt_parser, self._vptraits),
            dptraits=_dump_traits(traits.dpt_parser, self._dptraits))
        if self._vhtraits is not None:
            info['vhtraits'] = _dump_traits(traits.vht_parser, self._vhtraits)
        if self._dhtraits is not None:
            info['dhtraits'] = _dump_traits(traits.dht_parser, self._dhtraits)
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
            raise ConfigError(
                "a disk takes at most one velocity trait 'mass': each "
                "adds the whole rotation in the gravity")
        self._dptraits = _polar_traits('dptraits', dptraits)
        _check_dispersion_directions(self._dptraits)
        self._vhtraits = _height_traits(
            'vhtraits', vhtraits, len(self._vptraits), 'vptraits')
        for vptrait, vhtrait in zip(self._vptraits, self._vhtraits or ()):
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


class Equilibrium(parseutils.TypedSerializable, abc.ABC):
    """
    The vertical equilibrium of a thick disk (see DiskTraitsKinematics):
    the disk is held up against the gravity of its galaxy by its velocity
    dispersion. Its thickness is that of its density (its density height
    traits); the equilibrium gives the dispersion that holds it up,
    sigma_eq(R), which adds to that of its dispersion traits in
    quadrature, sigma = hypot(sigma_traits, sigma_eq), at every height
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
    ConfigError
        If the direction is not one of those.
    """

    def dump(self) -> dict[str, Any]:
        info = dict(type=self.type())
        if self._direction != EQUILIBRIUM_DIRECTION_DEFAULT:
            info['direction'] = self._direction
        return info

    def __init__(self, direction: str = EQUILIBRIUM_DIRECTION_DEFAULT):
        if direction not in EQUILIBRIUM_DIRECTIONS:
            raise ConfigError(
                f"the direction of the dispersion of an equilibrium must be "
                f"one of {list(EQUILIBRIUM_DIRECTIONS)}; it is "
                f"{direction!r}")
        self._direction = direction

    def direction(self) -> str:
        """Return the direction of its dispersion: 'iso' or 'ver'."""
        return self._direction


class CentralEquilibrium(Equilibrium):
    """
    The vertical equilibrium in the pull of a spherical mass: sigma_eq =
    h V_c / R, with h the rms height of the disk at R (that of its
    density, from its density height traits) and V_c the circular
    velocity of the mass of its galaxy, or without mass, the rotation of
    the disk (the azimuthal mean of its tangential velocity traits at the
    midplane); exact for a disk thin against its radius (h << R) in a
    spherical gravity, whatever the shape of its height profile (the
    vertical virial theorem: sigma^2 = Omega^2 <z^2>, Omega = V_c / R). At
    the centre, where V_c / R is 0 / 0, it takes Omega at the next
    subnode; NaN where V_c^2 is negative. It is the thick disk of
    GalPaK3D, there with h a fixed share of the radius of the disk. See
    Equilibrium.
    """

    @staticmethod
    def type() -> str:
        return 'central'


class SelfGravitatingEquilibrium(Equilibrium):
    """
    The vertical equilibrium in the pull of the disk itself: sigma_eq^2 =
    2 sqrt(3) G Sigma h, with Sigma the surface density of a mass
    component of its galaxy (e.g. a mass disk that follows the disk) and
    h its rms height; exact for an isothermal slab of its own gravity, of
    the height profile sech^2(z / z0), z0 = sigma^2 / (pi G Sigma) (<z^2>
    = pi^2 z0^2 / 12); NaN where Sigma is negative. See Equilibrium.

    Parameters
    ----------
    density : str
        The name of the mass component whose surface density holds the
        disk up.
    direction : str, optional
        As that of Equilibrium.

    Raises
    ------
    ConfigError
        If the name or the direction is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'self_gravitating'

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(density=self._density)

    def __init__(
            self,
            density: str,
            direction: str = EQUILIBRIUM_DIRECTION_DEFAULT
    ):
        super().__init__(direction)
        parseutils.check_name(density)
        self._density = density

    def density(self) -> str:
        """
        Return the name of the mass component whose surface density holds
        the disk up.
        """
        return self._density


class JeansEquilibrium(Equilibrium):
    """
    The vertical equilibrium in the pull of all the mass of its galaxy
    (the vertical Jeans equation, isothermal in z): sigma_eq^2 = int_0^inf
    rho(z) (-F_z(R, z)) dz / rho(0), the weight of the column of the disk
    above its midplane over its density there (rho of its density traits
    and density height traits), with F_z the vertical force of every mass
    component (see mass.MassComponent.vertical_force: user components give
    it by their function of the forces); exact for any mass and profile of
    the height, given the isothermal dispersion. At the centre, the value
    at the next subnode, as for 'central' (a point mass there makes it
    infinite). See Equilibrium.
    """

    @staticmethod
    def type() -> str:
        return 'jeans'


equilibrium_parser = parseutils.TypedParser(Equilibrium, [
    CentralEquilibrium,
    SelfGravitatingEquilibrium,
    JeansEquilibrium])


class DiskTraitsKinematics(parseutils.TypedSerializable):
    """
    How a disk moves, as its traits give (see KinematicTraits), and with
    a vertical equilibrium, the dispersion that holds a thick disk up
    against the gravity of its galaxy (see Equilibrium). A disk that
    emits lines into a spectral cube needs kinematics.

    Parameters
    ----------
    vptraits, dptraits, vhtraits, dhtraits : optional
        The traits (see KinematicTraits).
    equilibrium : Equilibrium, optional
        The vertical equilibrium of a thick disk; none by default.

    Raises
    ------
    ConfigError
        If the traits are not valid (see KinematicTraits).
    """

    @staticmethod
    def type() -> str:
        return 'traits'

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        KinematicTraits.load_options(info)
        parseutils.load_option_and_update_info(
            equilibrium_parser, info, 'equilibrium')
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        info = dict(type=self.type()) | self._traits.dump()
        if self._equilibrium is not None:
            info['equilibrium'] = equilibrium_parser.dump(self._equilibrium)
        return info

    def __init__(
            self,
            vptraits: traits.VPTrait | Sequence[traits.VPTrait],
            dptraits: traits.DPTrait | Sequence[traits.DPTrait],
            vhtraits:
            traits.VHTrait | Sequence[traits.VHTrait | None] | None = None,
            dhtraits:
            traits.DHTrait | Sequence[traits.DHTrait | None] | None = None,
            equilibrium: Equilibrium | None = None
    ):
        self._traits = KinematicTraits(vptraits, dptraits, vhtraits, dhtraits)
        self._equilibrium = equilibrium

    def vptraits(self) -> tuple[traits.VPTrait, ...]:
        """Return the velocity polar traits."""
        return self._traits.vptraits()

    def dptraits(self) -> tuple[traits.DPTrait, ...]:
        """Return the velocity dispersion polar traits."""
        return self._traits.dptraits()

    def vhtraits(self) -> tuple[traits.VHTrait | None, ...] | None:
        """Return the velocity height traits, if given."""
        return self._traits.vhtraits()

    def dhtraits(self) -> tuple[traits.DHTrait | None, ...] | None:
        """Return the velocity dispersion height traits, if given."""
        return self._traits.dhtraits()

    def equilibrium(self) -> Equilibrium | None:
        """Return the vertical equilibrium, if any."""
        return self._equilibrium


class ConeTraitsKinematics(parseutils.TypedSerializable):
    """
    How a cone moves, as its traits give (see KinematicTraits). A cone
    that emits lines into a spectral cube needs kinematics.

    Parameters
    ----------
    vptraits, dptraits, vhtraits, dhtraits : optional
        The traits (see KinematicTraits).

    Raises
    ------
    ConfigError
        If the traits are not valid (see KinematicTraits).
    """

    @staticmethod
    def type() -> str:
        return 'traits'

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        KinematicTraits.load_options(info)
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type()) | self._traits.dump()

    def __init__(
            self,
            vptraits: traits.VPTrait | Sequence[traits.VPTrait],
            dptraits: traits.DPTrait | Sequence[traits.DPTrait],
            vhtraits:
            traits.VHTrait | Sequence[traits.VHTrait | None] | None = None,
            dhtraits:
            traits.DHTrait | Sequence[traits.DHTrait | None] | None = None
    ):
        self._traits = KinematicTraits(vptraits, dptraits, vhtraits, dhtraits)

    def vptraits(self) -> tuple[traits.VPTrait, ...]:
        """Return the velocity polar traits."""
        return self._traits.vptraits()

    def dptraits(self) -> tuple[traits.DPTrait, ...]:
        """Return the velocity dispersion polar traits."""
        return self._traits.dptraits()

    def vhtraits(self) -> tuple[traits.VHTrait | None, ...] | None:
        """Return the velocity height traits, if given."""
        return self._traits.vhtraits()

    def dhtraits(self) -> tuple[traits.DHTrait | None, ...] | None:
        """Return the velocity dispersion height traits, if given."""
        return self._traits.dhtraits()


density_parser = parseutils.BasicParser(Density)

disk_kinematics_parser = parseutils.TypedParser(
    DiskTraitsKinematics, [DiskTraitsKinematics])

cone_kinematics_parser = parseutils.TypedParser(
    ConeTraitsKinematics, [ConeTraitsKinematics])
