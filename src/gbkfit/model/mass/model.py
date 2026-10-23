"""
Mass models: the components of the mass of a galaxy, which add up.
"""

from collections.abc import Callable, Collection, Sequence
from typing import Any

import dataclasses

import numpy as np
import numpy.typing as npt

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from .base import MassComponent, MassContext
from .disks import MassDisk
from .points import MassPoint
from .spheroids import MassSpheroid
from .tables import MassTable
from .user import MassUser


__all__ = [
    'MassModel',
    'mass_component_parser',
    'mass_model_parser'
]


mass_component_parser = parseutils.TypedParser(MassComponent, [
    MassPoint,
    MassDisk,
    MassSpheroid,
    MassTable,
    MassUser])


class MassModel(parseutils.Serializable):
    """
    The mass of a galaxy: its mass components (e.g. stars, gas, dark
    matter), whose densities, potentials and forces add up, and so do
    their squared circular velocities. Its parameters are those of its
    components, prefixed by 'mass_' and their names (e.g. mass_halo_c).
    It computes gravity only, in kpc (see units): the distance of its
    galaxy converts arcsec to kpc, and the velocity traits 'mass' of the
    components of its galaxy turn its circular velocity into their
    rotation.

    Parameters
    ----------
    components : MassComponent or Sequence of MassComponent
        Its mass components.

    Raises
    ------
    ConfigError
        If there are no components, their names repeat, or a component
        responds to one that the model does not have, to itself, or in a
        loop (see MassComponent.responds_to).
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
        self._check_responses()

    def _check_responses(self) -> None:
        """
        Raise ConfigError if a component responds to one that the model
        does not have, or in a loop (itself included).
        """
        names = [component.name() for component in self._components]
        responses = {
            component.name(): component.responds_to()
            for component in self._components}
        for name, others in responses.items():
            if unknown := sorted(set(others) - set(names)):
                raise ConfigError(
                    f"the mass component {name!r} responds to "
                    f"{unknown}, which the mass model does not have; its "
                    f"components: {names}")

        def visit(name: str, path: tuple[str, ...]) -> None:
            if name in path:
                raise ConfigError(
                    f"the mass components respond to each other in a "
                    f"loop: {' -> '.join(path + (name,))}")
            for other in responses[name]:
                visit(other, path + (name,))
        for name in names:
            visit(name, ())

    def components(self) -> tuple[MassComponent, ...]:
        """Return its mass components."""
        return self._components

    def component(self, name: str) -> MassComponent:
        """
        Return its mass component of a name.

        Raises
        ------
        KeyError
            If it has none of that name.
        """
        return self._components[self._index(name)]

    def component_params(
            self, name: str, params: ParamValues) -> ParamValues:
        """
        Return the values of the parameters of its mass component of a
        name, by their names in the component, from those of its
        parameters (and possibly others).
        """
        mapping = self._mappings[self._index(name)]
        return {own: params[full] for own, full in mapping.items()}

    def _index(self, name: str) -> int:
        """Return the index of its mass component of a name."""
        names = [component.name() for component in self._components]
        if name not in names:
            raise KeyError(
                f"the mass model has no component {name!r}; its "
                f"components: {names}")
        return names.index(name)

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
            lambda component, own, ctx: component.vcirc2(R, own, ctx),
            params, context)

    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its mass (Msun) within the spheres of the radii r (kpc):
        that of all its components (see MassComponent.enclosed_mass).

        Raises
        ------
        NotImplementedError
            If a component does not give it (see gives_enclosed_mass).
        """
        if not self.gives_enclosed_mass():
            raise NotImplementedError(
                "not every mass component gives its enclosed mass")
        return sum(self.enclosed_mass_components(r, params, context).values())

    def enclosed_mass_components(
            self,
            r: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> dict[str, np.ndarray]:
        """
        Return the mass of each component that gives it within the spheres
        of the radii r, by the name of the component (see enclosed_mass).
        """
        return {
            component.name(): component.enclosed_mass(
                r, self.component_params(component.name(), params),
                self._context(component, params, context))
            for component in self._components
            if component.gives_enclosed_mass()}

    def gives_enclosed_mass(self) -> bool:
        """
        Check whether every component gives its enclosed mass (see
        MassComponent.gives_enclosed_mass).
        """
        return all(cmp.gives_enclosed_mass() for cmp in self._components)

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
            lambda component, own, ctx: component.density(R, z, own, ctx),
            params, context).values())

    def cell_masses(
            self,
            r_edges: np.ndarray,
            b_edges: np.ndarray,
            params: ParamValues,
            context: MassContext,
            components: Collection[str] | None = None
    ) -> np.ndarray:
        """
        Return its mass in each cell between spheres and latitudes (see
        MassComponent.cell_masses): that of all its components, or of
        those of the given names.
        """
        total = np.zeros((np.size(r_edges) - 1, np.size(b_edges) - 1))
        for component in self._components:
            name = component.name()
            if components is None or name in components:
                total = total + component.cell_masses(
                    r_edges, b_edges, self.component_params(name, params),
                    self._context(component, params, context))
        return total

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
            lambda component, own, ctx: component.forces(R, z, own, ctx),
            params, context).values()
        return (sum(force_R for force_R, _ in forces),
                sum(force_z for _, force_z in forces))

    def vertical_force(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its vertical force per unit mass at the points (R, z) (see
        MassComponent.vertical_force).
        """
        return sum(self._each(
            lambda component, own, ctx: component.vertical_force(
                R, z, own, ctx), params, context).values())

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
            lambda component, own, ctx: component.potential(
                R, z, own, ctx), params, context).values())

    def _each(
            self,
            function: Callable[
                [MassComponent, ParamValues, MassContext], Any],
            params: ParamValues,
            context: MassContext
    ) -> dict[str, Any]:
        """
        Return the result of a function of each component, the values of
        its own parameters and its context (that of the model, with the
        squared circular velocities of the components it responds to), by
        the name of the component.
        """
        return {
            component.name(): function(
                component, self.component_params(component.name(), params),
                self._context(component, params, context))
            for component in self._components}

    def _context(
            self,
            component: MassComponent,
            params: ParamValues,
            context: MassContext
    ) -> MassContext:
        """
        Return the context of a component: that of the model, with the
        squared circular velocities of the components it responds to.
        """
        if not component.responds_to():
            return context

        def vcirc2(name: str) -> Callable[[np.ndarray], np.ndarray]:
            other = self.component(name)
            own = self.component_params(name, params)
            other_context = self._context(other, params, context)
            return lambda R: other.vcirc2(R, own, other_context)
        return dataclasses.replace(context, vcirc2s={
            name: vcirc2(name) for name in component.responds_to()})


mass_model_parser = parseutils.BasicParser(MassModel)
