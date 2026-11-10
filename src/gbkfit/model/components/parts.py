"""
The traits of the components made of nodes (rings or shells along a
radius, e.g. disks), which they take as options of their own (see
traits): the checks of their density traits (density_traits) and their
kinematic traits (KinematicTraits, see kinematic_traits).
"""

from collections.abc import Sequence
from typing import Any

from gbkfit.config import TypedParser
from gbkfit.utils import iterutils
from . import traits
from ._nodal import MAX_TRAITS


__all__ = [
    'KinematicTraits',
    'density_traits',
    'dump_traits',
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
    velocity height trait 'mass': the trait 'mass' (see
    traits.VHTraitMass).
    """
    if not isinstance(vptrait, traits.VPTraitMass):
        raise ValueError(
            f"the velocity height trait 'mass' is the rotation in the "
            f"gravity at each height: it pairs with the velocity trait "
            f"'mass' only, not {vptrait.type()!r}")


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
        one. The last can be the vertical equilibrium of a thick disk
        (see traits.DPTraitEquilibrium), whose dispersion its galaxy
        derives and adds in quadrature; the others are fitted, and add
        up.
    vhtraits : VHTrait or Sequence of VHTrait, optional
        The velocity height traits of a thick component: one that every
        velocity polar trait shares (parameters 'vht_*'), or one for each
        (None items or none at all: the velocity does not change with the
        distance from the surface).
    dhtraits : DHTrait or Sequence of DHTrait, optional
        The dispersion height traits of a thick component, one shared or
        one for each fitted dispersion polar trait (as vhtraits; the
        equilibrium, isothermal in z, takes none).

    Raises
    ------
    ValueError
        If there are no polar traits or too many, neither one shared
        height trait nor one for each polar trait, a direction of the
        dispersion without a dispersion trait, several velocity traits
        'mass' (the rotation in the gravity would count more than once),
        a velocity height trait 'mass' that does not pair with the
        velocity trait 'mass' (see traits.VHTraitMass), or an equilibrium
        that is not the last of the dispersion traits.
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
        last = self._dptraits[-1]
        self._equilibrium = last \
            if isinstance(last, traits.DPTraitEquilibrium) else None
        self._fitted = self._dptraits[:-1] if self._equilibrium is not None \
            else self._dptraits
        if any(isinstance(trait, traits.DPTraitEquilibrium)
               for trait in self._fitted):
            raise ValueError(
                "a disk takes at most one dispersion trait 'equilibrium', "
                "the last of its dptraits: its dispersion adds in "
                "quadrature to the sum of the others")
        if self._fitted:
            _check_dispersion_directions(self._fitted)
        self._vhtraits = _height_traits(
            'vhtraits', vhtraits, len(self._vptraits), 'vptraits')
        # (a shared height trait pairs with every velocity trait)
        vhtraits = self._vhtraits or ()
        if len(vhtraits) == 1:
            vhtraits *= len(self._vptraits)
        for vptrait, vhtrait in zip(self._vptraits, vhtraits):
            if isinstance(vhtrait, traits.VHTraitMass):
                _check_rotation_at_heights(vptrait)
        if dhtraits is not None and not self._fitted:
            raise ValueError(
                "dhtraits are the heights of the fitted dispersion traits, "
                "and the equilibrium, isothermal in z, takes none: give "
                "none")
        self._dhtraits = _height_traits(
            'dhtraits', dhtraits, len(self._fitted),
            'dptraits' if self._equilibrium is None
            else 'dptraits but the equilibrium')

    def vptraits(self) -> tuple[traits.VPTrait, ...]:
        """Return the velocity polar traits."""
        return self._vptraits

    def dptraits(self) -> tuple[traits.DPTrait, ...]:
        """Return the velocity dispersion polar traits."""
        return self._dptraits

    def fitted_dptraits(self) -> tuple[traits.DPTrait, ...]:
        """
        Return the velocity dispersion polar traits but the equilibrium:
        those of parameters, which the kernels evaluate.
        """
        return self._fitted

    def equilibrium(self) -> traits.DPTraitEquilibrium | None:
        """
        Return the vertical equilibrium, if any: the last of the
        dispersion traits, whose dispersion the galaxy derives.
        """
        return self._equilibrium

    def vhtraits(self) -> tuple[traits.VHTrait | None, ...] | None:
        """Return the velocity height traits, if given."""
        return self._vhtraits

    def dhtraits(self) -> tuple[traits.DHTrait | None, ...] | None:
        """Return the velocity dispersion height traits, if given."""
        return self._dhtraits
