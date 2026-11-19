"""
The setups of the projects (the recipes of their fits), as the library
describes them without devices (see gbkfit.setup.describe): their
parameters, each with the file that gives it its properties, and their
problems.
"""

import dataclasses
import io
from collections.abc import Mapping
from typing import Any

import ruamel.yaml

import gbkfit.setup
from gbkfit.observation import ObservationGroup

from .workspace import Workspace


__all__ = [
    'SetupDescription',
    'Setups',
    'setup_text'
]


@dataclasses.dataclass
class SetupDescription:
    """
    A setup as the library describes it: its parameters (in its names: their
    titles and descriptions for people, sizes, defaults and units; the item
    they are of, by its name in the setup, '' for one galaxy without a
    name; their properties as its layers merge them, if any; and the file
    of the layer that gives them, relative to the project, if any: e.g. the
    params piece of a galaxy, or the setup itself), in the order of the
    library; and its problems (their severities, paths in the setup and
    messages).
    """
    parameters: list[dict[str, Any]]
    problems: list[dict[str, Any]]


class Setups:
    """
    The setups of the projects of a workspace.

    Parameters
    ----------
    workspace : Workspace
        The workspace.
    """

    def __init__(self, workspace: Workspace):
        self._workspace = workspace

    def describe(
            self, path: str, setup: Mapping[str, Any]
    ) -> SetupDescription:
        """
        Describe a setup of the given options as the file of a path (of a
        project, relative to the workspace; e.g. as it is edited, not saved
        yet): its parameters and its problems.
        """
        filename = self._workspace.path(path)
        description = gbkfit.setup.describe(
            filename, {filename: setup_text(setup)})
        problems = [
            dict(severity=p.severity, path=list(p.path), message=p.message)
            for p in description.problems]
        space, group = description.params, description.group
        pdescs = space.pdescs() if space is not None \
            else group.pdescs() if group is not None else {}
        params = (description.config or {}).get('params')
        properties = params.get('properties', {}) \
            if isinstance(params, Mapping) else {}
        sources = description.sources.get('parameters', {})
        owners = _owners(group) if group is not None else []
        parameters = []
        for name, desc in pdescs.items():
            # (the keys of its properties: its name, or with elements)
            keys = [
                key for key in properties
                if key.split('[', 1)[0].strip() == name]
            parameters.append(dict(
                name=name, title=desc.title(), desc=desc.desc(),
                size=desc.size(), default=desc.default(),
                unit=None if desc.unit() is None else str(desc.unit()),
                item=_owner(name, owners),
                properties={key: properties[key] for key in keys},
                source=next(
                    (sources[key] for key in keys if key in sources), None)))
        return SetupDescription(parameters, problems)


def setup_text(setup: Mapping[str, Any]) -> str:
    """Return the text of a setup of the given options (YAML)."""
    stream = io.StringIO()
    yaml = ruamel.yaml.YAML(typ='safe')
    yaml.default_flow_style = None
    yaml.dump(dict(setup), stream)
    return stream.getvalue()


def _owners(group: ObservationGroup) -> list[str]:
    """
    Return the names of the items of a group of a setup (its galaxies,
    lenses and observations), which prefix the names of their parameters;
    longest first.
    """
    names = [
        item.name() for item in (
            *group.galaxies(), *group.lenses(), *group.observations())]
    return sorted(
        {name for name in names if name}, key=len, reverse=True)


def _owner(name: str, owners: list[str]) -> str:
    """
    Return the item of a setup a parameter is of, by its name (see
    _owners): '' for one galaxy without a name.
    """
    return next((owner for owner in owners if name.startswith(f'{owner}_')),
                '')
