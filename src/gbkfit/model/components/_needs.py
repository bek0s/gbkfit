"""
What the components of a galaxy need from it (Need): capabilities of the
galaxy (see Capability: its mass, and what its gravity gives of it),
which the galaxy checks when it is made (see check_needs), and which can
be asked before (see galaxy_gives).
"""

from collections.abc import Collection, Iterable, Sequence
from typing import NamedTuple

from gbkfit.utils.parseutils import ConfigError
from ..gravity import Capability, Gravity
from ..mass import MassModel


__all__ = [
    'Need',
    'check_densities',
    'check_needs',
    'density_needs',
    'galaxy_gives'
]


class Need(NamedTuple):
    """
    Something that a component needs from its galaxy (see
    Component.needs): a capability, what needs it, and the mass
    components whose densities it takes, if any.

    Attributes
    ----------
    capability : Capability
        The capability.
    what : str
        Of the mass alone, the message of the error without it (e.g. "the
        'mass' velocity traits of the component 'gas' need the mass of
        their galaxy"); else, what takes the capability (e.g. "the jeans
        equilibrium of the component 'gas' takes the vertical force of the
        gravity of its galaxy").
    densities : tuple of str
        The names of the mass components whose densities it takes (e.g.
        the pressure support of a disk; see density_needs).
    surface : bool
        Whether it takes their surface densities, not their densities.
    """
    capability: Capability
    what: str
    densities: tuple[str, ...] = ()
    surface: bool = False


def density_needs(
        names: Sequence[str],
        surface: bool,
        what: str
) -> list[Need]:
    """
    Return the needs of the densities of mass components (by name), or of
    their surface densities as self-gravitating sheets in Newtonian
    gravity (none: the surface densities of the sheets only, e.g. of the
    disk itself), of what takes them (e.g. "the hydrostatic pressure
    support of the component 'gas'").
    """
    kind = 'surface density' if surface else 'density'
    needs = [Need(
        Capability.MASS, f"{what} takes the {kind}", tuple(names), surface)] \
        if names else []
    if surface:
        needs.append(Need(
            Capability.SHEETS,
            f"{what} takes the vertical structure of self-gravitating sheets "
            f"in Newtonian gravity"))
    return needs


def galaxy_gives(
        mass_model: MassModel | None,
        gravity: Gravity
) -> dict[Capability, str | None]:
    """
    Return what a galaxy of a mass model (None without mass) and a gravity
    gives its components: of each capability, None if it gives it, or else
    why not.
    """
    if mass_model is None:
        return {capability: "it has no mass" for capability in Capability}
    return {Capability.MASS: None} | gravity.gives(mass_model)


def check_needs(
        needs: Iterable[Need],
        mass_model: MassModel | None,
        gravity: Gravity
) -> None:
    """
    Check that a galaxy of a mass model (None without mass) and a gravity
    gives what its components need (see galaxy_gives). Without mass, only
    the needs of the mass alone fail: those of the rest of what the mass
    gives come with them.

    Raises
    ------
    ConfigError
        If the galaxy does not give a need, saying what needs it.
    """
    gives = galaxy_gives(mass_model, gravity)
    names = [] if mass_model is None \
        else [component.name() for component in mass_model.components()]
    for need in needs:
        if mass_model is None:
            if need.capability is Capability.MASS and not need.densities:
                raise ConfigError(need.what)
            continue
        if unknown := [name for name in need.densities if name not in names]:
            raise ConfigError(
                f"{need.what} of the mass components {unknown}, which its "
                f"galaxy does not have; its mass components: {names}")
        if (reason := gives[need.capability]) is not None:
            raise ConfigError(
                f"{need.what}, which it does not give: {reason}")


def check_densities(
        needs: Iterable[Need],
        mass_model: MassModel,
        thick: Collection[str]
) -> None:
    """
    Check that the mass components whose densities (or surface densities)
    needs take give them (see MassComponent.gives_density and
    gives_surface_density), in a galaxy whose thick components have the
    given names.

    Raises
    ------
    ConfigError
        If one does not, saying what takes it.
    """
    # (at load, not as an error of the first evaluation, which would end a
    # fit: audit 2026-10-31 #30a)
    for need in needs:
        kind = 'surface density' if need.surface else 'density'
        for name in need.densities:
            component = mass_model.component(name)
            gives = component.gives_surface_density() if need.surface \
                else component.gives_density(thick)
            if not gives:
                raise ConfigError(
                    f"{need.what} of the mass component {name!r}, which "
                    f"does not give its {kind} (e.g. a halo of a profile "
                    f"without a projection, a thin disk, a table or a user "
                    f"component)")
