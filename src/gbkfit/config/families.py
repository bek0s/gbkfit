"""
The families of configurations: the base classes whose configurations
choose one of their members by its type (the option 'type'; e.g. the
PSFs), each with the parser that names it (see TypedParser), and their
members.

A family is known once the module of its parser is imported (the
catalogue of GBKFIT, gbkfit.catalogue, imports them all). A member that
needs modules GBKFIT does not always have (e.g. the nautilus fitter, or
the cuda driver; see Serializable.requires) is in its family whether
they are installed or not, with what to install.
"""

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING

from .serializable import TypedSerializable, missing_modules

if TYPE_CHECKING:
    from .parsers import TypedParser


__all__ = [
    'Family',
    'Member',
    'families',
    'family_of'
]


# The parsers of the families, by name (see register_family)
_PARSERS: dict[str, 'TypedParser'] = {}


@dataclasses.dataclass(frozen=True)
class Member:
    """
    A member of a family.

    Parameters
    ----------
    type : str
        Its type (the option 'type' of its configurations).
    cls : type
        Its class.
    missing : Mapping
        The modules it needs that are not installed, each with how to
        install it (see Serializable.requires); none if it is available.
    """

    type: str
    cls: type[TypedSerializable]
    missing: Mapping[str, str]

    def available(self) -> bool:
        """Check whether it can be made: the modules it needs are installed."""
        return not self.missing


@dataclasses.dataclass(frozen=True)
class Family:
    """
    A family: a base class whose configurations choose one of its members
    by its type.

    Parameters
    ----------
    name : str
        Its name (e.g. 'psf').
    parser : TypedParser
        The parser of its configurations.
    members : tuple of Member
        Its members, by type.
    """

    name: str
    parser: 'TypedParser'
    members: tuple[Member, ...]

    def base(self) -> type[TypedSerializable]:
        """Return its base class."""
        return self.parser.cls()


def register_family(name: str, parser: 'TypedParser') -> None:
    """
    Register the parser of a family (see TypedParser, which registers
    those that name their family).

    Raises
    ------
    RuntimeError
        If a family of the name, or of the base class, is registered.
    """
    if name in _PARSERS:
        raise RuntimeError(f"a family '{name}' is already registered")
    if (other := family_of(parser.cls())) is not None:
        raise RuntimeError(
            f"the family '{other.name}' is already of the class "
            f"{parser.cls_name()}")
    _PARSERS[name] = parser


def families() -> dict[str, Family]:
    """
    Return the families registered so far, and their members (see the
    module).

    Returns
    -------
    dict
        The families, by name, in the order of their names.
    """
    return {name: _family(name, _PARSERS[name]) for name in sorted(_PARSERS)}


def family_of(base: type) -> Family | None:
    """
    Return the family of a base class (e.g. that of PSF, for an option of
    the type PSF), or None if it is the base of none.

    Parameters
    ----------
    base : type
        The class.

    Returns
    -------
    Family or None
        The family.
    """
    for name, parser in _PARSERS.items():
        if parser.cls() is base:
            return _family(name, parser)
    return None


def _family(name: str, parser: 'TypedParser') -> Family:
    """Return a family, of its name and parser."""
    members = tuple(
        Member(type_, cls, missing_modules(cls))
        for type_, cls in sorted(parser.registered_classes().items()))
    return Family(name, parser, members)
