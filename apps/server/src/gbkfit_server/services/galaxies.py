"""
The galaxies of the projects, as the library describes them without
devices (see gbkfit.describe): the parameters a galaxy has, and its
problems. A galaxy is described in a setup of its own, seen by the
observation of the template of observations (of a cube; else of an
image), with no values of its parameters.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

import ruamel.yaml

import gbkfit
import gbkfit.project

from .workspace import Workspace


__all__ = [
    'GalaxyDescription',
    'Galaxies'
]


_yaml = ruamel.yaml.YAML(typ='safe')

# The observations a galaxy is described through, in turn: that of the
# template (a cube), and an image (e.g. for a galaxy of a band)
_OBSERVATIONS = (
    _yaml.load(gbkfit.project.TEMPLATES['observations']),
    dict(type='sky', observable=dict(type='pixel_brightness', size=[48, 48]),
         line_of_sight=dict(type='flat')))


@dataclasses.dataclass
class GalaxyDescription:
    """
    A galaxy as the library describes it: its parameters (their names,
    sizes, defaults, units, valid ranges and periods, in its order), and
    its problems (their severities, paths in the galaxy and messages).
    """
    parameters: list[dict[str, Any]]
    problems: list[dict[str, Any]]


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
        given).
        """
        root = self._workspace.path(project)
        description = None
        for observation in _OBSERVATIONS:
            description = gbkfit.describe(dict(
                version=gbkfit.project.FORMAT_VERSION, root=root,
                driver=dict(type='host'), galaxies=[dict(galaxy)],
                observations=[observation], params=dict(properties={})))
            if description.group is not None:
                break
        parameters = [] if description.group is None else [
            dict(name=name, size=desc.size(), default=desc.default(),
                 unit=None if desc.unit() is None else str(desc.unit()),
                 range=str(desc.valid_range()), period=desc.period())
            for name, desc in description.group.pdescs().items()]
        problems = [
            dict(severity=p.severity, path=p.path[2:], message=p.message)
            for p in description.problems
            if p.path[:2] == ['galaxies', 0]
            or not p.path and description.group is None]
        return GalaxyDescription(parameters, problems)
