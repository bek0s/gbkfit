"""
What the components of a galaxy need from it (Need): each need checks
itself against the mass and the gravity of the galaxy when the galaxy is
made (see Component.needs).
"""

import abc
from collections.abc import Callable, Sequence
from typing import Any

from gbkfit.utils.parseutils import ConfigError
from ..gravity import Gravity
from ..mass import MassModel


__all__ = [
    'Need',
    'NeedsForces',
    'NeedsMass',
    'NeedsMassDensities',
    'NeedsPotential'
]


class Need(abc.ABC):
    """
    Something that a component needs from its galaxy (see
    Component.needs), which checks itself against the mass and the
    gravity of the galaxy.
    """

    @abc.abstractmethod
    def check(self, mass_model: MassModel | None, gravity: Gravity) -> None:
        """
        Check that a galaxy of the given mass model (None without mass) and
        gravity gives it.

        Raises
        ------
        ConfigError
            If the galaxy does not give it, saying what needs it.
        """
        pass


class NeedsMass(Need):
    """
    The mass of the galaxy (e.g. its circular velocity).

    Parameters
    ----------
    message : str
        What needs it, and why: the message of the error without mass.
    """

    def __init__(self, message: str):
        self._message = message

    def check(self, mass_model: MassModel | None, gravity: Gravity) -> None:
        if mass_model is None:
            raise ConfigError(self._message)


class NeedsForces(Need):
    """
    The forces of the mass in the gravity of the galaxy, off the midplane
    of its disks (see Gravity.check_forces), if it has mass (see NeedsMass
    for the mass itself).

    Parameters
    ----------
    what : str
        What takes them (e.g. "the jeans equilibrium of the component
        'gas' takes the vertical force of the gravity of its galaxy").
    """

    def __init__(self, what: str):
        self._what = what

    def check(self, mass_model: MassModel | None, gravity: Gravity) -> None:
        if mass_model is not None:
            _check_gravity(gravity.check_forces, mass_model, what=self._what)


class NeedsPotential(Need):
    """
    Forces of the mass in the gravity of the galaxy that are the gradient
    of a potential (see Gravity.check_potential), if it has mass (see
    NeedsMass for the mass itself): e.g. for orbits, which keep their
    energy in them.

    Parameters
    ----------
    what : str
        What takes them (e.g. "the orbits of the spheroid 'stars'").
    """

    def __init__(self, what: str):
        self._what = what

    def check(self, mass_model: MassModel | None, gravity: Gravity) -> None:
        if mass_model is not None:
            _check_gravity(
                gravity.check_potential, mass_model, what=self._what)


class NeedsMassDensities(Need):
    """
    The densities of mass components of the galaxy, by name, if it has
    mass (see NeedsMass for the mass itself): their densities, or their
    surface densities as self-gravitating sheets in Newtonian gravity (see
    Gravity.check_self_gravitating_sheets).

    Parameters
    ----------
    names : Sequence of str
        The names of the mass components (none: the surface densities of
        the sheets only, e.g. of the disk itself).
    surface : bool
        Whether they are surface densities of self-gravitating sheets.
    what : str
        What takes them (e.g. "the jeans pressure support of the
        component 'gas'").
    """

    def __init__(self, names: Sequence[str], surface: bool, what: str):
        self._names = tuple(names)
        self._surface = surface
        self._what = what

    def check(self, mass_model: MassModel | None, gravity: Gravity) -> None:
        if mass_model is None:
            return
        names = [component.name() for component in mass_model.components()]
        if unknown := [name for name in self._names if name not in names]:
            kind = 'surface density' if self._surface else 'density'
            raise ConfigError(
                f"{self._what} takes the {kind} of the mass components "
                f"{unknown}, which its galaxy does not have; its mass "
                f"components: {names}")
        if self._surface:
            _check_gravity(
                gravity.check_self_gravitating_sheets,
                what=f"{self._what} takes the vertical structure of "
                     f"self-gravitating sheets in Newtonian gravity")


def _check_gravity(
        check: Callable[..., None], *args: Any, what: str
) -> None:
    """
    Call a check of what a gravity gives (see Gravity) with the given
    arguments, and raise ConfigError with what a feature takes (what) if
    it does not give it.
    """
    try:
        check(*args)
    except ConfigError as e:
        raise ConfigError(
            f"{what}, which it does not give: {e.message}") from None
