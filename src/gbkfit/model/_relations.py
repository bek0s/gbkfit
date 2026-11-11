"""
The relations of the parts of a galaxy (see Galaxy), built and checked
once (see link): the names of their parameters (GalaxyParams), and the
links of its components and its mass (MassLinks), of which each
component sees its own (ComponentLinks).
"""

import dataclasses
from collections.abc import Sequence

from gbkfit.config import config_path, item_prefixes
from gbkfit.params import ParamDesc, pdescs_with_units
from gbkfit.utils import iterutils
from .components import _units
from .components._needs import Need, check_densities, check_needs
from .components.base import Component, ComponentLinks
from .distance import Distance
from .gravity import Capability, Gravity, GravityNewton
from .mass import (
    ComponentDensity, DiskDensity, MassComponent, MassDisk, MassModel,
    SpheroidDensity, mass_component_parser)


__all__ = [
    'GalaxyParams',
    'MassLinks',
    'link'
]


@dataclasses.dataclass(frozen=True)
class GalaxyParams:
    """
    The parameters of a galaxy: the prefix of the parameters and the
    constants of each component (its name and '_', or none for one
    component without a name), the names of the parameters of each
    component among those of the galaxy, by its own names (mappings), and
    all the parameters of the galaxy, by name: those of its components,
    its distance, its mass and its gravity (pdescs).
    """
    prefixes: tuple[str, ...]
    mappings: tuple[dict[str, str], ...]
    pdescs: dict[str, ParamDesc]


@dataclasses.dataclass(frozen=True)
class MassLinks:
    """
    The links of the components of a galaxy and its mass: the names of
    the components whose densities the mass follows (followed: by their
    surface density or their layer), and of those whose layer it follows
    (layers), the names of the components whose densities have a height
    (thick, which the gravity of the mass takes), and the fluid of each
    component that has one (see Component.fluid), by its index: the mass
    disks that it names, or else those that follow it (see
    MassModel.fluid). A galaxy without mass has none.
    """
    followed: frozenset[str] = frozenset()
    layers: frozenset[str] = frozenset()
    thick: tuple[str, ...] = ()
    fluids: dict[int, tuple[str, ...]] = dataclasses.field(
        default_factory=dict)

    def of(self, index: int, component: Component) -> ComponentLinks:
        """Return the links of a component of the galaxy, of its index."""
        return ComponentLinks(
            component.name() in self.followed, self.fluids.get(index, ()))


def link(
        components: Sequence[Component],
        distance: Distance | None,
        mass_model: MassModel | None,
        gravity: Gravity | None
) -> tuple[GalaxyParams, MassLinks]:
    """
    Return the parameters of a galaxy of the given components, distance,
    mass (None without) and gravity (None: Newtonian), and the links of
    its components and its mass, after checking that they go together.

    Raises
    ------
    ValueError
        If there are no components and no mass, several components without
        names, the names of the parameters conflict, components give one
        line different units (see _units.check_line_units_agree), or the
        mass and the components do not go together (see _with_mass).
    """
    if not components and mass_model is None:
        raise ValueError("a galaxy needs at least one component, or mass")
    _units.check_line_units_agree(components)
    prefixes = _component_prefixes(components)
    # Every component has its own geometric parameters (e.g. disk_posa): the
    # galaxy shares none of them between components. Components that share
    # an orientation tie their parameters in the parameter space instead
    # (e.g. stars_posa: disk_posa), which does the same with one mechanism.
    # Shared geometries (2026-10-25 to 2026-11-04) did only that, by merging
    # names here, at the cost of a geometry option on every component, named
    # geometries, and rings that moved to the geometry when it was warped. A
    # better form of them, if one is found, would again only map names here.
    params, mappings = iterutils.merge_with_prefixes(
        [cmp.pdescs() for cmp in components], list(prefixes))
    pdescs, links = _with_mass(
        params, components, distance, mass_model, gravity)
    return GalaxyParams(tuple(prefixes), mappings, pdescs), links


def _with_mass(
        params: dict[str, ParamDesc],
        components: Sequence[Component],
        distance: Distance | None,
        mass_model: MassModel | None,
        gravity: Gravity | None
) -> tuple[dict[str, ParamDesc], MassLinks]:
    """
    Return the parameters of the components of a galaxy, and those of its
    distance, its mass and its gravity (if any; Newtonian by default), with
    the mass per unit of the density of a component (see
    MassModel.mass_ratios) in the unit that the emission of the component
    gives it (see _units.mass_ratio_unit); and the links of its components
    and its mass (see MassLinks). The errors of a component or a mass
    component are of its path in the configuration.
    Raise ValueError if the gravity is given without mass (its only
    user), a component needs what the galaxy does not give (see
    Component.needs: e.g. the mass, or the distance, see
    _distance_needs), a mass
    component follows a component whose density it cannot follow (see
    Component.density_kind), the gravity does not take the mass model (see
    Gravity.check_mass_model) or does not give the shares of a dark
    fraction, or the names of the parameters repeat.
    """
    if gravity is not None and mass_model is None:
        raise ValueError(
            "the gravity of a galaxy is that of its mass; give the galaxy "
            "mass, or no gravity")
    gravity = GravityNewton() if gravity is None else gravity
    if mass_model is None:
        _check_needs(components, MassLinks(), mass_model, gravity)
        # (a distance without mass gives the length of an arcsec to the
        # derived values, e.g. radii in kpc of a disk of an arctan curve)
        return (params if distance is None
                else _with_params(params, distance.pdescs())), MassLinks()
    check_needs(
        _distance_needs(components, mass_model), mass_model, gravity,
        distance is not None)
    kinds = {cmp.name(): cmp.density_kind() for cmp in components
             if cmp.density_kind() is not None}
    for index, mass_component in enumerate(mass_model.components()):
        for name in mass_component.followed_components():
            if kinds.get(name) is not mass_component.followed_kind():
                with config_path('mass', index):
                    raise ValueError(
                        _follow_error(mass_component, name, kinds))
    # (of density height traits, as the densities that mass disks follow:
    # see DiskDensity.thick)
    thick = [cmp.name() for cmp in components if cmp.density_has_height()]
    gravity.check_mass_model(mass_model, thick)
    links = MassLinks(
        followed=frozenset(
            name for mass_component in mass_model.components()
            for name in mass_component.followed_components()),
        layers=frozenset(
            mass.vertical().followed_component()
            for mass in mass_model.components() if isinstance(mass, MassDisk)
        ) - {None},
        thick=tuple(thick))
    links = dataclasses.replace(
        links, fluids=_check_fluids(components, mass_model, links))
    _check_needs(components, links, mass_model, gravity)
    check_needs([
        Need(Capability.SHARES,
             f"the mass component {mass_component.name()!r} takes the "
             f"shares of the components in the squared circular velocity "
             f"(its dark fraction)")
        for mass_component in mass_model.components()
        if mass_component.takes_shares()], mass_model, gravity)
    emissions = {cmp.name(): cmp.emission() for cmp in components}
    mass = pdescs_with_units(mass_model.pdescs(), {
        name: _units.mass_ratio_unit(emissions[followed])
        for name, followed in mass_model.mass_ratios().items()})
    own = {} if distance is None else distance.pdescs()
    return _with_params(params, own | mass | gravity.pdescs()), links


def _check_fluids(
        components: Sequence[Component],
        mass_model: MassModel,
        links: MassLinks
) -> dict[int, tuple[str, ...]]:
    """
    Return the fluid of each component of a galaxy of a mass model that
    has one, by its index (see MassLinks), after checking the fluids and
    the vertical balance of the components, of the given links (see
    Component.fluid and Component.balance_gravity): the mass disks that
    their fluids name, or the mass disks that follow each, whose surface
    density its equilibrium 'sheet' takes; that none balances its rotation
    (the equilibrium 'rotation') next to a mass model, where the vertical
    frequency of a disk is 2.5-3 times its angular speed; and that no mass
    disk follows the layer of one whose thickness is its equilibrium in
    the field of the mass (the mass would take the thickness that its own
    force sets: not yet).

    Raises
    ------
    ValueError
        If one of these does not hold, of the path of the component.
    """
    fluids = {}
    for index, cmp in enumerate(components):
        gravity = cmp.balance_gravity()
        with config_path('components', index):
            label = repr(cmp.name()) if cmp.name() is not None else ''
            if gravity == 'rotation':
                raise ValueError(
                    f"the vertical equilibrium 'rotation' of the component "
                    f"{label} takes the pull of a spherical mass of its "
                    f"rotation, which is not that of a galaxy of mass (a "
                    f"disk pulls 2.5-3 times harder where it dominates): "
                    f"give it the equilibrium 'galaxy' or 'sheet'")
            if cmp.thickness_in_field() and cmp.name() in links.layers:
                raise ValueError(
                    f"a mass disk follows the layer of the component "
                    f"{label}, whose thickness is its equilibrium in the "
                    f"force of the mass: a mass that takes the thickness "
                    f"that its own force sets is not supported yet; give "
                    f"the mass disk a vertical structure of its own, or the "
                    f"disk the equilibrium 'sheet'")
            fluid = mass_model.fluid(cmp.name(), cmp.fluid())
            if fluid and (cmp.fluid() or gravity == 'sheet'):
                fluids[index] = fluid
            if gravity == 'sheet' and not fluid:
                raise ValueError(
                    f"the vertical equilibrium 'sheet' of the component "
                    f"{label} takes the surface density of its fluid: give "
                    f"it a fluid, or a mass disk that follows it")
    return fluids


def _check_needs(
        components: Sequence[Component],
        links: MassLinks,
        mass_model: MassModel | None,
        gravity: Gravity
) -> None:
    """
    Check what the components of a galaxy need from it, of their links
    (see Component.needs): that the galaxy gives it (see check_needs),
    and with mass, that the mass components give the densities that they
    take (see check_densities).

    Raises
    ------
    ValueError
        If one does not hold, of the path of the component.
    """
    for index, cmp in enumerate(components):
        needs = cmp.needs(links.of(index, cmp))
        with config_path('components', index):
            check_needs(needs, mass_model, gravity)
            if mass_model is not None:
                check_densities(needs, mass_model, links.thick)


def _distance_needs(
        components: Sequence[Component],
        mass_model: MassModel
) -> list[Need]:
    """
    Return what needs the distance of a galaxy of the components and the
    mass model (see Capability.DISTANCE): the components that take its
    mass (see Component.uses_mass), whose radii in arcsec meet its kpc,
    and the mass components that need it (see
    MassComponent.needs_distance).
    """
    needs = [
        Need(Capability.DISTANCE, (
            f"the component {cmp.name()!r} moves in the mass of its "
            f"galaxy, and so takes the length of an arcsec in it (its "
            f"distance)") if cmp.name() is not None else (
            "the component moves in the mass of its galaxy, and so takes "
            "the length of an arcsec in it (its distance)"))
        for cmp in components if cmp.uses_mass()]
    return needs + [
        Need(Capability.DISTANCE, (
            f"the mass component {mass_component.name()!r} {why}, and so "
            f"takes the length of an arcsec in its galaxy (its distance)"))
        for mass_component in mass_model.components()
        if (why := mass_component.needs_distance()) is not None]


def _with_params(
        params: dict[str, ParamDesc], taken: dict[str, ParamDesc]
) -> dict[str, ParamDesc]:
    """
    Return the parameters of the components of a galaxy with those of its
    distance, its mass and its gravity (taken), or raise ValueError if
    their names repeat.
    """
    if repeated := sorted(params.keys() & taken.keys()):
        raise ValueError(
            f"the parameters of the distance, the mass and the gravity have "
            f"the names of parameters of the components: {repeated}; rename "
            f"the components")
    return params | taken


# The kinds of the densities that the mass can follow, in words
_KINDS = {DiskDensity: 'a disk', SpheroidDensity: 'a spheroid'}


def _follow_error(
        mass_component: MassComponent,
        name: str,
        kinds: dict[str, type[ComponentDensity]]
) -> str:
    """
    Return the message of the error of a mass component that follows the
    component of the given name, which is not one that it can follow,
    with the kinds of the densities of the components that the mass can
    follow, by their names.
    """
    wanted = _KINDS[mass_component.followed_kind()]
    head = (f"the mass component {mass_component.name()!r} follows the "
            f"density of {wanted}")
    if name not in kinds:
        return (f"{head}, but its galaxy has no component {name!r} whose "
                f"density the mass can follow; those it can: {list(kinds)}")
    types = [cls.type() for cls in mass_component_parser.registered_classes()
             .values() if cls.followed_kind() is kinds[name]]
    return (f"{head}, but {name!r} is {_KINDS[kinds[name]]}: the mass that "
            f"follows it is of the type {' or '.join(map(repr, types))}")


def _component_prefixes(components: Sequence[Component]) -> list[str]:
    """
    Return the prefix of the parameters and the constants of each
    component of a galaxy: its name and '_', or none for one
    component without a name. Raise ValueError if several components do
    not all have names (see item_prefixes).
    """
    names = [cmp.name() for cmp in components]
    if len(names) > 1 and None in names:
        unnamed = [i for i, name in enumerate(names) if name is None]
        raise ValueError(
            f"the components of a galaxy need names when there are "
            f"several; the components {unnamed} have none")
    return item_prefixes(names, 'components', 'cmp', False)
