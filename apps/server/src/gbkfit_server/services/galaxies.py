"""
The galaxies of the projects, as the library describes them without
devices or observations (see gbkfit.setup.describe_galaxy): the
parameters a galaxy has, and its problems. A galaxy the library does not
make as a whole (it has a problem) is described part by part, so that a
problem in one part does not hide the parameters of the others.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

from gbkfit.config import config_leaves
from gbkfit.model import Galaxy, MassComponent
from gbkfit.setup import describe_galaxy

from .workspace import Workspace


__all__ = [
    'Galaxies',
    'GalaxyDescription'
]


@dataclasses.dataclass
class GalaxyDescription:
    """
    A galaxy as the library describes it: its parameters (their names,
    titles and descriptions for people, sizes, defaults, units, valid
    ranges and periods, and the paths of the options that make them in the
    galaxy, see Galaxy.param_owners; in its order), its problems (their
    severities, paths in the galaxy and messages), and whether the library
    made it as a whole (else its parameters are those of its parts made
    on their own; see Galaxies.describe).
    """
    parameters: list[dict[str, Any]]
    problems: list[dict[str, Any]]
    whole: bool


class Galaxies:
    """
    The galaxies of the projects of a workspace.

    Parameters
    ----------
    workspace : Workspace
        The workspace.
    """

    def __init__(self, workspace: Workspace):
        self._workspace = workspace

    def describe(
            self, project: str, galaxy: Mapping[str, Any]
    ) -> GalaxyDescription:
        """
        Describe a galaxy of the options of a piece of a project (its files
        relative to the project): its parameters, and its problems (none
        but those of the galaxy: the values of its parameters are not
        given). Of a galaxy not made as a whole, the parameters are those
        of its parts made on their own (see _parameters_by_part).
        """
        root = self._workspace.path(project)
        description = describe_galaxy(galaxy, root)
        made = description.galaxy
        parameters = _parameters(made) if made is not None \
            else _parameters_by_part(root, galaxy, description.problems)
        problems = [
            dict(severity=p.severity, path=list(p.path), message=p.message)
            for p in description.problems]
        return GalaxyDescription(parameters, problems, made is not None)


def _parameters(galaxy: Galaxy) -> list[dict[str, Any]]:
    """
    Return the parameters of a galaxy, in its order (see
    GalaxyDescription).
    """
    owners = galaxy.param_owners()
    return [
        dict(name=name, title=desc.title(), desc=desc.desc(),
             size=desc.size(), default=desc.default(),
             unit=None if desc.unit() is None else str(desc.unit()),
             range=str(desc.valid_range()), period=desc.period(),
             owner=list(owners.get(name, ())))
        for name, desc in galaxy.pdescs().items()]


def _parameters_by_part(
        root: str, galaxy: Mapping[str, Any], problems: list[Any]
) -> list[dict[str, Any]]:
    """
    Return the parameters of the parts of a galaxy that the library does
    not make as a whole: of each of its components that it makes on its
    own (with the parts of its mass that follow it or no component, and
    have no problem), then of its distance, gravity and mass (with the
    components made on their own, and the parts of the mass that follow
    none of the others and have no problem), each with the path of its
    option in the whole galaxy. A part made without the mass is made
    without its gravity too (which can follow the mass). The parameters of
    a part that is not made are not known. A component keeps its name on
    its own, and so the names of its parameters.
    """
    components = _listed(galaxy.get('components'))
    mass = _listed(galaxy.get('mass'))
    names = [component.get('name') for component in components]
    broken = {
        p.path[1] for p in problems
        if p.path[:1] == ['mass'] and len(p.path) > 1}

    def kept(made: set[str]) -> list[int]:
        # (the parts of the mass with no problem that follow only the
        # components made)
        return [
            j for j, part in enumerate(mass)
            if j not in broken and _followed(part) <= made]

    parameters = []
    made = []
    for index, component in enumerate(components):
        found = _made_part(root, galaxy, [component], mass, kept(
            {names[index]}))
        if found is None:
            continue
        made.append(index)
        parameters += [
            dict(p, owner=['components', index, *p['owner'][2:]])
            for p in found if p['owner'][:1] == ['components']]
    indices = kept({names[i] for i in made})
    found = _made_part(
        root, galaxy, [components[i] for i in made], mass, indices)
    for p in found or []:
        if p['owner'][:1] == ['mass']:
            parameters.append(
                dict(p, owner=['mass', indices[p['owner'][1]],
                               *p['owner'][2:]]))
        elif p['owner'][:1] != ['components']:
            parameters.append(p)
    return parameters


def _made_part(
        root: str, galaxy: Mapping[str, Any],
        components: list[dict[str, Any]], mass: list[dict[str, Any]],
        indices: list[int]
) -> list[dict[str, Any]] | None:
    """
    Return the parameters of a galaxy of the given components and parts
    of its mass (by their indices), else of the components alone (without
    mass and gravity); None if neither is made.
    """
    part = {
        key: value for key, value in galaxy.items()
        if key not in ('components', 'mass')}
    part['components'] = components
    if indices:
        made = describe_galaxy(
            part | dict(mass=[mass[j] for j in indices]), root).galaxy
        if made is not None:
            return _parameters(made)
    part.pop('gravity', None)
    made = describe_galaxy(part, root).galaxy
    return None if made is None else _parameters(made)


def _listed(value: Any) -> list[Any]:
    """Return the items of an option of one item or a list of them."""
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _followed(part: Any) -> set[str]:
    """
    Return the names of the components that a part of the mass names (e.g.
    the component it follows; see config_leaves).
    """
    return {
        leaf.value for leaf in config_leaves(part, MassComponent)
        if leaf.kind == 'components'}
