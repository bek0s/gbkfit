"""
Mass models: the components of the mass of a galaxy, which add up.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from .base import MassComponent, MassContext
from .disks import MassExponentialDisk, MassNodewiseDisk, MassTraitDisk
from .halos import (
    MassBurkert, MassEinasto, MassNFW, MassPseudoIsothermal, MassZhao)
from .points import MassPointMass
from .spheroids import MassSersic
from .user import MassUser


__all__ = [
    'MassModel',
    'mass_component_parser',
    'mass_model_parser'
]


mass_component_parser = parseutils.TypedParser(MassComponent, [
    MassPointMass,
    MassExponentialDisk,
    MassNodewiseDisk,
    MassTraitDisk,
    MassSersic,
    MassNFW,
    MassBurkert,
    MassPseudoIsothermal,
    MassEinasto,
    MassZhao,
    MassUser])


class MassModel(parseutils.Serializable):
    """
    The mass of a galaxy: its mass components (e.g. stars, gas, dark
    matter), whose densities, potentials and forces add up, and so do
    their squared circular velocities. Its parameters are those of its
    components, prefixed by 'mass_' and their names (e.g. mass_halo_c).
    It computes gravity only, in kpc (see units): the distance of its
    model converts arcsec to kpc, and the velocity traits 'mass' of the
    components of its model turn its circular velocity into their
    rotation.

    Parameters
    ----------
    components : MassComponent or Sequence of MassComponent
        Its mass components.

    Raises
    ------
    ConfigError
        If there are no components, or their names repeat.
    """

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'MassModel':
        parseutils.load_option_and_update_info(
            mass_component_parser, info, 'components', required=True)
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        return dict(components=mass_component_parser.dump(
            list(self._components)))

    def __init__(self, components: MassComponent | Sequence[MassComponent]):
        components = iterutils.tuplify(components)
        if not components:
            raise ConfigError("a mass model needs at least one component")
        names = [component.name() for component in components]
        if repeated := sorted({n for n in names if names.count(n) > 1}):
            raise ConfigError(
                f"the components of a mass model must have different "
                f"names; repeated: {repeated}")
        self._pdescs, self._mappings = iterutils.merge_with_prefixes(
            [component.pdescs() for component in components],
            [f'mass_{name}_' for name in names])
        self._components = components

    def components(self) -> tuple[MassComponent, ...]:
        """Return its mass components."""
        return self._components

    def pdescs(self) -> dict[str, ParamDesc]:
        """Return its parameters, by name."""
        return self._pdescs

    def vcirc2(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return the square of its circular velocity in the plane, signed
        (see MassComponent.vcirc2).

        Parameters
        ----------
        R : array_like
            The radii (kpc).
        params : ParamValues
            The values of its parameters (and possibly others), by name.
        context : MassContext
            The epoch of the galaxy, and the densities it follows (see
            MassContext).

        Returns
        -------
        np.ndarray
            The squared circular velocity ((km/s)^2) at each radius.
        """
        return sum(self.vcirc2_components(R, params, context).values())

    def vcirc2_components(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> dict[str, np.ndarray]:
        """
        Return the square of the circular velocity of each component in
        the plane, by the name of the component (see vcirc2).
        """
        return self._each(
            lambda component, own: component.vcirc2(R, own, context), params)

    def density(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its density at the points (R, z) (see
        MassComponent.density).
        """
        return sum(self._each(
            lambda component, own: component.density(R, z, own, context),
            params).values())

    def forces(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return its forces per unit mass at the points (R, z) (see
        MassComponent.forces).
        """
        forces = self._each(
            lambda component, own: component.forces(R, z, own, context),
            params).values()
        return (sum(force_R for force_R, _ in forces),
                sum(force_z for _, force_z in forces))

    def potential(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its potential at the points (R, z) (see
        MassComponent.potential).
        """
        return sum(self._each(
            lambda component, own: component.potential(R, z, own, context),
            params).values())

    def _each(
            self,
            function: Callable[[MassComponent, ParamValues], Any],
            params: ParamValues
    ) -> dict[str, Any]:
        """
        Return the result of a function of each component and the values
        of its own parameters, by the name of the component.
        """
        return {
            component.name(): function(component, {
                name: params[full] for name, full in mapping.items()})
            for component, mapping in zip(self._components, self._mappings)}


mass_model_parser = parseutils.BasicParser(MassModel)
