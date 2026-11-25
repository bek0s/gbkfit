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
    ComponentDensity, DiskDensity, MassComponent, MassModel,
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
    the mass components that follow the density of each component, by
    its name (followers: by its surface density or its layer), the names
    of the components whose layer the mass follows (layers), the names of
    the components whose densities have a height (thick, which the
    gravity of the mass takes), and whether the galaxy has mass (mass).
    A galaxy without mass has none.
    """
    followers: dict[str, tuple[str, ...]] = dataclasses.field(
        default_factory=dict)
    layers: frozenset[str] = frozenset()
    thick: tuple[str, ...] = ()
    mass: bool = False

    @property
    def followed(self) -> frozenset[str]:
        """The names of the components whose densities the mass follows."""
        return frozenset(self.followers)

    def of(self, component: Component) -> ComponentLinks:
        """Return the links of a component of the galaxy."""
        name = component.name()
        return ComponentLinks(
            self.followers.get(name, ()) if name is not None else (),
            self.mass, name in self.layers)


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
    MassModel.mass_ratios) in the unit that the light of the component
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
    followers: dict[str, tuple[str, ...]] = {}
    for mass_component in mass_model.components():
        for name in mass_component.followed_components():
            followers[name] = followers.get(name, ()) + (
                mass_component.name(),)
    links = MassLinks(
        followers=followers,
        layers=frozenset(
            mass.followed_layer() for mass in mass_model.components()
        ) - {None},
        thick=tuple(thick), mass=True)
    _check_needs(components, links, mass_model, gravity)
    lights = {cmp.name(): cmp.light() for cmp in components}
    mass = pdescs_with_units(mass_model.pdescs(), {
        name: _units.mass_ratio_unit(lights[followed])
        for name, followed in mass_model.mass_ratios().items()})
    own = {} if distance is None else distance.pdescs()
    return _with_params(params, own | mass | gravity.pdescs()), links


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
        with config_path('components', index):
            needs = cmp.needs(links.of(cmp))
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
    MassComponent.needs).
    """
    needs = [
        Need(Capability.DISTANCE, (
            f"the component {cmp.name()!r} moves in the mass of its "
            f"galaxy, and so takes the length of an arcsec in it (its "
            f"distance)") if cmp.name() is not None else (
            "the component moves in the mass of its galaxy, and so takes "
            "the length of an arcsec in it (its distance)"))
        for cmp in components if cmp.needs()]
    return needs + [
        need for mass_component in mass_model.components()
        for need in mass_component.needs()]


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
