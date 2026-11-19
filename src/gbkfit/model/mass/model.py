"""
Mass models: the components of the mass of a galaxy, which add up.
"""

from collections.abc import Callable, Collection, Sequence
from typing import Any

import dataclasses

import numpy as np
import numpy.typing as npt

from gbkfit.config import (
    Serializable, TypedParser, check_options)
from gbkfit.cosmology import Cosmology
from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import iterutils
from ..geometries import geometry_pdesc
from .base import MassComponent, MassContext
from .disks import MassDisk, TracedDisk
from .points import MassPoint
from .sky import MassPlace, MassPlane, SkyMass, SkyPlacement, SkyProjected
from .spheroids import Halo, MassSpheroid, TracedSpheroid
from .tables import MassTable
from .user import MassUser


__all__ = [
    'MassModel',
    'mass_component_parser'
]


mass_component_parser = TypedParser(MassComponent, [
    MassPoint,
    MassDisk,
    TracedDisk,
    Halo,
    MassSpheroid,
    TracedSpheroid,
    MassTable,
    MassUser], family='mass_component')

# The descriptions of the parameters of where the mass is (see
# MassModel.place), as those of the geometries of the components
_PLACE_DESCS = dict(
    xpos="the x coordinate of the centre of the mass on the sky",
    ypos="the y coordinate of the centre of the mass on the sky",
    posa="the position angle of the major axis of the plane of the mass, "
         "from north through east",
    incl="the angle of the axis of the mass from the line of sight (0: "
         "face-on)")


class MassModel(Serializable):
    """
    The mass of a galaxy: its mass components (e.g. stars, gas, dark
    matter), whose densities, potentials and forces add up, and so do
    their squared circular velocities, about one centre and one axis.
    It computes gravity only, in kpc (see units): the distance of its
    galaxy converts arcsec to kpc, and the velocity traits 'mass' of the
    components of its galaxy turn its circular velocity into their
    rotation.

    Its parameters are those of its components, prefixed by 'mass_' and
    their names (e.g. mass_halo_c). Where it is on the sky (see place) is
    given only where its mass is laid out on the sky (e.g. by a lens of
    its mass, see ObservationGroup), by the parameters of its place (see
    place_pdescs): its centre, mass_xpos and mass_ypos, if a component
    lies at it (see takes_mass_centre: e.g. a halo, or a point mass),
    and its plane, mass_posa and mass_incl (as those of disks), if a
    component lies in it (see takes_mass_plane: e.g. a mass disk, or a
    flattened halo). The mass components that follow components of its
    galaxy lie on them (e.g. a traced disk on the rings of its disk).

    Parameters
    ----------
    components : MassComponent or Sequence of MassComponent
        Its mass components.

    Raises
    ------
    ValueError
        If there are no components, their names repeat, or a component
        responds to one that the model does not have, to itself, or in a
        loop (see MassComponent.responds_to), or takes the enclosed mass
        of one that does not give it (see
        MassComponent.takes_enclosed_masses).
    """

    def dump(self) -> dict[str, Any]:
        return dict(components=mass_component_parser.dump(
            list(self._components)))

    def __init__(self, components: MassComponent | Sequence[MassComponent]):
        check_options(MassModel.__init__, locals())
        components = iterutils.tuplify(components)
        if not components:
            raise ValueError("a mass model needs at least one component")
        names = [component.name() for component in components]
        if repeated := sorted({n for n in names if names.count(n) > 1}):
            raise ValueError(
                f"the components of a mass model must have different "
                f"names; repeated: {repeated}")
        self._components = components
        self._check_responses()
        place = (('xpos', 'ypos') if self.takes_mass_centre() else ()) \
            + (('posa', 'incl') if self.takes_mass_plane() else ())
        self._place_pdescs = {
            f'mass_{name}': geometry_pdesc(name, desc=_PLACE_DESCS[name])
            for name in place}
        self._parts_pdescs, self._mappings = iterutils.merge_with_prefixes(
            [component.pdescs() for component in components],
            [f'mass_{name}_' for name in names])

    def _check_responses(self) -> None:
        """
        Raise ValueError if a component responds to one that the model
        does not have, or in a loop (itself included), or takes the
        enclosed mass of one that does not give it.
        """
        names = [component.name() for component in self._components]
        responses = {
            component.name(): component.responds_to()
            for component in self._components}
        for name, others in responses.items():
            if unknown := sorted(set(others) - set(names)):
                raise ValueError(
                    f"the mass component {name!r} responds to "
                    f"{unknown}, which the mass model does not have; its "
                    f"components: {names}")

        def visit(name: str, path: tuple[str, ...]) -> None:
            if name in path:
                raise ValueError(
                    f"the mass components respond to each other in a "
                    f"loop: {' -> '.join(path + (name,))}")
            for other in responses[name]:
                visit(other, path + (name,))
        for name in names:
            visit(name, ())
        for component in self._components:
            if not component.takes_enclosed_masses():
                continue
            if massless := [
                    other for other in component.responds_to()
                    if not self.component(other).gives_enclosed_mass()]:
                raise ValueError(
                    f"the response of the mass component "
                    f"{component.name()!r} takes the masses within spheres "
                    f"of its baryons; {massless} do not give them (tables "
                    f"give their circular velocity in the plane only, user "
                    f"components without a function of their mass): give "
                    f"them as mass disks of their surface densities")

    def check_cosmology(self, cosmology: Cosmology) -> None:
        """
        Raise ValueError, saying why, unless the cosmology of its galaxy
        gives what its components take: the cosmic baryon fraction (see
        MassComponent.takes_baryon_fraction).
        """
        if cosmology.baryon_fraction() is not None:
            return
        if takers := [component.name() for component in self._components
                      if component.takes_baryon_fraction()]:
            raise ValueError(
                f"the mass components {takers} take the cosmic baryon "
                f"fraction, which the cosmology lacks (Omega_b): give the "
                f"setup a cosmology with it")

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

    def check_fluid(self, names: Sequence[str], what: str) -> None:
        """
        Check that the mass components of the given names, the fluid of a
        component of its galaxy (its gas, see Need.fluid), are mass disks,
        of what takes them (e.g. "the jeans pressure support of the
        component 'gas' takes the density of its fluid").

        Raises
        ------
        ValueError
            If one is not.
        """
        components = {mass.name(): mass for mass in self._components}
        for name in names:
            if not isinstance(components.get(name), MassDisk):
                kind = 'no mass component' if name not in components \
                    else f'a {components[name].type()}'
                raise ValueError(
                    f"{what}, which names mass disks; {name!r} is {kind}")

    def mappings(self) -> tuple[dict[str, str], ...]:
        """
        Return the names of the parameters of each of its components among
        its parameters, by the names of the component.
        """
        return tuple(self._mappings)

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
        """
        Return its parameters, those of its components, by name (e.g.
        mass_halo_c): all that its densities, forces and potentials read.
        """
        return self._parts_pdescs

    def place_pdescs(self) -> dict[str, ParamDesc]:
        """
        Return the parameters of where it is on the sky (see place), by
        name: mass_xpos and mass_ypos if it has a centre, and mass_posa
        and mass_incl if it has a plane; none if all its components
        follow components of its galaxy.
        """
        return self._place_pdescs

    def place(self, params: ParamValues) -> MassPlace:
        """
        Return where it is on the sky (see MassPlace), of the values of
        the parameters of its place (see place_pdescs; and possibly
        others), by name: its centre and its plane, if it has them.
        """
        centre = (float(params['mass_xpos']), float(params['mass_ypos'])) \
            if 'mass_xpos' in self._place_pdescs else None
        plane = MassPlane(
            float(params['mass_posa']), float(params['mass_incl'])) \
            if 'mass_posa' in self._place_pdescs else None
        return MassPlace(centre, plane)

    def mass_ratios(self) -> dict[str, str]:
        """
        Return its parameters that are the mass of one of its components
        per unit of the density of a component of its galaxy (see
        MassComponent.mass_ratios), each with the name of the component
        of the galaxy.
        """
        return {
            mapping[name]: followed
            for component, mapping in zip(self._components, self._mappings)
            for name, followed in component.mass_ratios().items()}

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

    def height_potential(
            self,
            R: np.ndarray,
            z: np.ndarray,
            params: ParamValues,
            context: MassContext,
            names: Collection[str] | None = None
    ) -> np.ndarray:
        """
        Return its height potential at the radii R and the heights z (see
        MassComponent.height_potential): of all its components, or of
        those of the given names.
        """
        components = self._components if names is None \
            else [self.component(name) for name in names]
        return sum((
            component.height_potential(
                R, z, self.component_params(component.name(), params),
                self._context(component, params, context))
            for component in components), np.zeros((len(R), len(z))))

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

    def projection_gaps(self) -> str | None:
        """
        Return why its mass cannot be laid out on the sky (see
        sky_mass): the components that do not say where their mass is
        (see sky.SkyProjected); None if it can.
        """
        if gaps := [component.name() for component in self._components
                    if not isinstance(component, SkyProjected)]:
            return (
                f"the mass components {gaps} do not say where their mass is "
                f"(tables give their circular velocity, and functions of the "
                f"user their gravity, not their density): give them as "
                f"disks, spheroids, halos or points")
        return None

    def takes_mass_centre(self) -> bool:
        """
        Check whether a component lies at the centre of the mass (see
        sky.SkyProjected.takes_mass_centre; e.g. a halo, a point mass, or
        a mass disk of its own surface density), which it then has (see
        place): not those that follow components of its galaxy, which lie
        on them.
        """
        return any(
            isinstance(component, SkyProjected)
            and component.takes_mass_centre()
            for component in self._components)

    def takes_mass_plane(self) -> bool:
        """
        Check whether a component lies in the plane of the mass (see
        sky.SkyProjected.takes_mass_plane; e.g. a mass disk, or a
        flattened halo), which it then has (see place). Those that follow
        components of its galaxy lie on them, and the others are round
        (e.g. a spherical halo, or a point mass), or do not say where
        their mass is (tables and functions of the user, whose circular
        velocities are those of the planes of the components that take
        them).
        """
        return any(
            isinstance(component, SkyProjected)
            and component.takes_mass_plane() for component in self._components)

    def sky_mass(
            self,
            params: ParamValues,
            context: MassContext,
            placement: SkyPlacement
    ) -> SkyMass:
        """
        Return its mass laid out on the sky (see sky.SkyMass): the layers of
        each component, of the values of its own parameters and its context
        (with the components it responds to, see MassContext).

        Parameters
        ----------
        params : ParamValues
            The values of its parameters (and possibly others), by name.
        context : MassContext
            The context of the mass, of a galaxy with a distance.
        placement : SkyPlacement
            Where its galaxy places its mass.

        Returns
        -------
        SkyMass
            Its mass on the sky.

        Raises
        ------
        ValueError
            If the context has no distance, or a component cannot lay out
            its mass on the sky (see projection_gaps).
        ModelEvaluationError
            If the values give no layers (see sky.SkyProjected).
        """
        if context.kpc_per_arcsec is None:
            raise ValueError(
                "the mass of a galaxy is laid out on the sky at its "
                "distance, which it lacks")
        if (gaps := self.projection_gaps()) is not None:
            raise ValueError(gaps)
        return SkyMass(context.kpc_per_arcsec, self._each(
            lambda component, own, ctx: component.sky_layers(
                own, ctx, placement), params, context))

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
        squared circular velocities and the masses within spheres (of
        those that give them) of the components it responds to.
        """
        if not component.responds_to():
            return context

        def bound(
                function: Callable[..., np.ndarray],
                own: ParamValues,
                other_context: MassContext
        ) -> Callable[[np.ndarray], np.ndarray]:
            return lambda r: function(r, own, other_context)
        vcirc2s, enclosed_masses = {}, {}
        for name in component.responds_to():
            other = self.component(name)
            own = self.component_params(name, params)
            other_context = self._context(other, params, context)
            vcirc2s[name] = bound(other.vcirc2, own, other_context)
            if other.gives_enclosed_mass():
                enclosed_masses[name] = bound(
                    other.enclosed_mass, own, other_context)
        return dataclasses.replace(
            context, vcirc2s=vcirc2s, enclosed_masses=enclosed_masses)

