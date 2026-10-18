"""
The gravity of a galaxy: how its mass (see MassModel) pulls. Each
gravity gives the field of the mass in one evaluation (see
GravityField), whose circular velocity in the plane moves the
components; what else a field gives (see Gravity), the features that
need it check when the galaxy is made.
"""

import abc
from typing import Any

import numpy as np
import numpy.typing as npt

from gbkfit.params import ParamValues
from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from .mass import MassContext, MassModel


__all__ = [
    'Gravity',
    'GravityField',
    'GravityNewton',
    'gravity_parser'
]


class GravityField(abc.ABC):
    """
    The gravity of the mass of a galaxy in one evaluation, in kpc (see
    Gravity.field).
    """

    @abc.abstractmethod
    def vcirc2(self, R: npt.ArrayLike) -> np.ndarray:
        """
        Return the square of the circular velocity ((km/s)^2) in the plane
        at the radii R (kpc), signed: negative where the mass pulls
        outwards.
        """

    @abc.abstractmethod
    def vertical_force(
            self, R: npt.ArrayLike, z: npt.ArrayLike
    ) -> np.ndarray:
        """
        Return the vertical force per unit mass ((km/s)^2/kpc) at the
        points (R, z) (kpc), if its gravity gives it (see
        Gravity.check_vertical_force).
        """


class Gravity(parseutils.TypedSerializable, abc.ABC):
    """
    How the mass of a galaxy pulls: its parameters, and the field of the
    mass in each evaluation (see GravityField). Every field gives the
    circular velocity in the plane; the features that need more check
    when the galaxy is made that its gravity gives it:

    - the vertical force anywhere, which the jeans equilibrium takes
      (see check_vertical_force and parts.Kinematics)
    - the vertical structure of self-gravitating isothermal sheets,
      which the self_gravitating equilibrium and pressure support take
      (see check_self_gravitating_sheets, parts.Kinematics and
      traits.VPTraitMass)
    - the shares of the components in the squared circular velocity,
      which the dark fraction of a spheroid takes (see check_shares and
      mass.DarkFraction)
    """

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        """Return its parameters, by name; none here."""
        return {}

    @abc.abstractmethod
    def field(
            self,
            mass_model: MassModel,
            params: ParamValues,
            context: MassContext
    ) -> GravityField:
        """
        Return the field of a mass model, at the values of the parameters
        of its galaxy (those of the gravity among them), in its context.
        """

    @abc.abstractmethod
    def check_vertical_force(self, mass_model: MassModel) -> None:
        """
        Raise ConfigError, saying why, unless its fields give the vertical
        force of the mass model anywhere.
        """

    @abc.abstractmethod
    def check_self_gravitating_sheets(self) -> None:
        """
        Raise ConfigError, saying why, unless the vertical structure of a
        self-gravitating isothermal sheet in it is that of Newtonian
        gravity (h ~ sigma^2 / (G Sigma)).
        """

    @abc.abstractmethod
    def check_shares(self) -> None:
        """
        Raise ConfigError, saying why, unless the squared circular
        velocities of the components add up to that of the mass.
        """


class GravityNewton(Gravity):
    """
    Newtonian gravity, by default: the fields of the components of the
    mass add up, and so do their squared circular velocities.
    """

    @staticmethod
    def type() -> str:
        return 'newton'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    def field(
            self,
            mass_model: MassModel,
            params: ParamValues,
            context: MassContext
    ) -> GravityField:
        return _NewtonField(mass_model, params, context)

    def check_vertical_force(self, mass_model: MassModel) -> None:
        if forceless := [
                component.name() for component in mass_model.components()
                if not component.gives_vertical_force()]:
            raise ConfigError(
                f"the mass components {forceless} do not give it (user "
                f"components give their circular velocity only)")

    def check_self_gravitating_sheets(self) -> None:
        pass

    def check_shares(self) -> None:
        pass


class _NewtonField(GravityField):
    """The field of a mass model in Newtonian gravity (see GravityNewton)."""

    def __init__(
            self,
            mass_model: MassModel,
            params: ParamValues,
            context: MassContext
    ):
        self._mass_model = mass_model
        self._params = params
        self._context = context

    def vcirc2(self, R: npt.ArrayLike) -> np.ndarray:
        return self._mass_model.vcirc2(R, self._params, self._context)

    def vertical_force(
            self, R: npt.ArrayLike, z: npt.ArrayLike
    ) -> np.ndarray:
        return self._mass_model.vertical_force(
            R, z, self._params, self._context)


gravity_parser = parseutils.TypedParser(Gravity, [GravityNewton])
