"""
The parts that disks are made of, besides their rings: their density and
their kinematics, each a set of traits (see traits).
"""

from collections.abc import Sequence
from typing import Any, Self

from gbkfit.utils import iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from . import traits
from ._disk import MAX_TRAITS


__all__ = [
    'Density',
    'Kinematics',
    'density_parser',
    'kinematics_parser'
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
    The density of a disk: its polar traits, the surface density seen
    face-on, and for a thick disk its height traits, the probability
    density of the height of each (see traits). What the density is
    depends on the roles of its component: light for one that emits, the
    optical depth for dust.

    Parameters
    ----------
    ptraits : RPTrait or Sequence of RPTrait
        The polar traits (parameters 'pt_*').
    htraits : RHTrait or Sequence of RHTrait, optional
        The height traits, one for each polar trait (parameters 'ht_*');
        a disk without them is thin.

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


class Kinematics(parseutils.Serializable):
    """
    How a disk moves: the velocity and the velocity dispersion of each of
    its points, given by traits (see traits). The velocity traits add up,
    so motions combine; a rotation in the gravity of the galaxy is the
    velocity trait 'mass' (see traits.VPTraitMass). A disk that emits
    lines into a spectral cube needs kinematics.

    Parameters
    ----------
    vptraits : VPTrait or Sequence of VPTrait
        The velocity polar traits (parameters 'vpt_*').
    dptraits : DPTrait or Sequence of DPTrait
        The velocity dispersion polar traits (parameters 'dpt_*').
    vhtraits : VHTrait or Sequence of VHTrait, optional
        The velocity height traits of a thick disk, one for each velocity
        polar trait (None items or none at all: the velocity does not
        change with height).
    dhtraits : DHTrait or Sequence of DHTrait, optional
        The dispersion height traits of a thick disk, one for each
        dispersion polar trait (as vhtraits).

    Raises
    ------
    ConfigError
        If there are no polar traits or too many, or not one height trait
        for each polar trait.
    """

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        for parser, key, required in (
                (traits.vpt_parser, 'vptraits', True),
                (traits.dpt_parser, 'dptraits', True),
                (traits.vht_parser, 'vhtraits', False),
                (traits.dht_parser, 'dhtraits', False)):
            parseutils.load_option_and_update_info(
                parser, info, key, required=required)
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
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
        self._dptraits = _polar_traits('dptraits', dptraits)
        self._vhtraits = _height_traits(
            'vhtraits', vhtraits, len(self._vptraits), 'vptraits')
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


density_parser = parseutils.BasicParser(Density)

kinematics_parser = parseutils.BasicParser(Kinematics)
