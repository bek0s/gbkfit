"""
The rules of version 4 of the format (2026-11-04 on), which rewrite
configurations of version 3.
"""

import copy
import re
from collections.abc import Mapping, MutableMapping
from typing import Any

from ruamel.yaml.comments import CommentedMap

from .rules import (
    EXPRESSIONS, Change, Rule, components, flow, items, rename, replaced)


__all__ = [
    'RULES',
    'OwnGeometries',
    'RingNodes'
]


# The options of the rings of a disk, which a warped geometry gave the
# disks that shared it
_RINGS = ('rnmax', 'rnsep', 'rnlen', 'rnodes')

class OwnGeometries(Rule):
    """
    Every component has a geometry of its own (2026-11-04): the galaxy
    options geometry and geometries and the component option geometry
    are gone. A component that shared a geometry takes its warps and, if
    it was warped, its rings, and its geometric parameters are its own
    (e.g. disk_posa, was posa or {geometry}_posa): in a params section of
    the configuration, the shared parameter is renamed to that of the
    first of its components, and those of the others are tied to it
    (e.g. stars_posa: disk_posa), which keeps the model; without one,
    the changes say what to rename and tie by hand. The rings of a warped
    geometry, which expressions name (e.g. rnodes, {geometry}_rnodes),
    are those of its first component (e.g. disk_rnodes). The galaxies of
    a project, which have no names, have their parameters renamed by hand:
    which params are of which galaxy is in the setups.
    """

    @staticmethod
    def version() -> int:
        return 4

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        galaxies = list(items(config, 'galaxies'))
        # Galaxies without names among several are the pieces of a project
        # (those of one configuration have names), each of its own setups:
        # which params are of which is not known here, so that their
        # parameters are renamed by hand
        alone = len(galaxies) == 1
        for path, galaxy in galaxies:
            shared, rings, galaxy_changes = self._galaxy(galaxy, path)
            changes += galaxy_changes
            sections = _sections(config) if alone or 'name' in galaxy \
                else []
            for old, new in shared.items():
                changes += _share(sections, path, old, new)
            for old, new in rings.items():
                if not sections:
                    changes.append(Change(path, (
                        f"by hand: rename {old} to {new} in expressions")))
                for params_path, params in sections:
                    changes += _renamed_in_expressions(
                        params, params_path, old, new)
        return changes

    @staticmethod
    def _galaxy(
            galaxy: MutableMapping[str, Any], path: list[str | int]
    ) -> tuple[dict[str, list[str]], dict[str, str], list[Change]]:
        """
        Give each component of a galaxy its own geometry, in place, and
        return the names of the parameters it shared, each with those of
        the components that shared it, in their order, and the names of
        the rings of its warped geometries, each with that of the rings of
        their first component.
        """
        changes = []
        galaxy_prefix = _prefix(galaxy)
        own = galaxy.pop('geometry', None)
        named = galaxy.pop('geometries', None)
        if own is not None or named is not None:
            changes.append(Change(path, (
                "geometry and geometries removed: every component has its "
                "own geometry")))
        own = own if isinstance(own, Mapping) else {}
        named = named if isinstance(named, list) else [named] \
            if isinstance(named, Mapping) else []
        by_name = {
            g.get('name'): g for g in named if isinstance(g, Mapping)}
        shared: dict[str, list[str]] = {}
        rings: dict[str, str] = {}
        for cmp_path, component in components(galaxy):
            cmp_path = path + cmp_path
            choice = component.pop('geometry', 'galaxy')
            if choice == 'own':
                changes.append(Change(cmp_path, (
                    "geometry: own removed (every component has its own)")))
                continue
            geometry = own if choice == 'galaxy' else by_name.get(choice)
            if not isinstance(geometry, Mapping):
                changes.append(Change(cmp_path, (
                    f"by hand: the geometry {choice!r} is unknown; give "
                    f"the component the warps and rings it had")))
                continue
            if choice != 'galaxy':
                changes.append(Change(cmp_path, (
                    f"geometry: {choice} removed (every component has its "
                    f"own)")))
            changes += _take_warps(component, geometry, cmp_path)
            names = _geometric_params(component)
            if names is None:
                changes.append(Change(cmp_path, (
                    f"by hand: tie the geometric parameters of this "
                    f"component of the type {component.get('type')!r} to "
                    f"those of the other components of its geometry")))
                continue
            given = geometry.get('params')
            geometry_prefix = '' if choice == 'galaxy' else f'{choice}_'
            if geometry.get('loose') or geometry.get('tilted'):
                rings.setdefault(
                    f'{galaxy_prefix}{geometry_prefix}rnodes',
                    f'{galaxy_prefix}{_prefix(component)}rnodes')
            for name in names:
                if given is None or name in given:
                    shared.setdefault(
                        f'{galaxy_prefix}{geometry_prefix}{name}', []).append(
                        f'{galaxy_prefix}{_prefix(component)}{name}')
        return shared, rings, changes


def _prefix(item: Mapping[str, Any]) -> str:
    """Return the prefix of the parameters of a galaxy or a component."""
    return f"{item['name']}_" if 'name' in item else ''


def _take_warps(
        component: MutableMapping[str, Any],
        geometry: Mapping[str, Any],
        path: list[str | int]
) -> list[Change]:
    """
    Give a component the warps of the geometry it shared and, if warped,
    its rings, in place; only disks could share a warped one.
    """
    warps = [w for w in ('loose', 'tilted') if geometry.get(w)]
    if not warps:
        return []
    for warp in warps:
        component[warp] = True
    # A copy each: YAML would write one list of several components as an
    # anchor and its aliases
    rings = {k: copy.deepcopy(geometry[k]) for k in _RINGS if k in geometry}
    for key in _RINGS:
        component.pop(key, None)
    component.update(rings)
    return [Change(path, (
        f"{', '.join(warps)} and the rings {list(rings)} of its geometry "
        f"added"))]


def _geometric_params(component: Mapping[str, Any]) -> list[str] | None:
    """
    Return the geometric parameters of a component of version 3, as the
    components of that version had them; None for an unknown type. The
    rule keeps its own copy of these rules: they are those of version 3,
    and must not change with the components.
    """
    kind = component.get('type')
    kinematics = component.get('kinematics')
    kinematics = kinematics if isinstance(kinematics, Mapping) else {}
    emission = component.get('emission')
    lines = isinstance(emission, Mapping) and emission.get('type') == 'lines'
    # Disks and cones have a systemic velocity with velocity traits only
    if kind in ('disk', 'cone'):
        return ['xpos', 'ypos', 'posa', 'incl'] \
            + (['vsys'] if kinematics.get('vptraits') else [])
    if kind == 'point':
        return ['xpos', 'ypos'] + (['vsys'] if lines else [])
    if kind == 'particles':
        return ['xpos', 'ypos', 'posa', 'incl'] + (['vsys'] if lines else [])
    if kind == 'spheroid':
        # An axis unless it is a sphere without kinematics; a profile
        # without a shape, mge, has the axis ratios of its terms
        shape = component.get('shape')
        profile = component.get('profile')
        mge = isinstance(profile, Mapping) and profile.get('type') == 'mge'
        sphere = not mge and (
            not isinstance(shape, Mapping) or shape.get('type') == 'sphere')
        moving = component.get('kinematics') is not None
        return ['xpos', 'ypos'] \
            + (['posa', 'incl'] if moving or not sphere else []) \
            + (['vsys'] if moving else [])
    return None


def _share(
        sections: list[tuple[list[str | int], MutableMapping[str, Any]]],
        galaxy_path: list[str | int],
        old: str,
        new: list[str]
) -> list[Change]:
    """
    Rewrite a parameter that components shared, in the params sections
    of its galaxy: renamed to that of the first component (new[0]), with
    the others tied to it; or say what to do by hand without one.
    """
    first, others = new[0], new[1:]
    if first == old and not others:
        return []
    if not sections:
        ties = ''.join(f", {other}: {first}" for other in others)
        return [Change(galaxy_path, (
            f"by hand: rename the parameter {old} to {first}{ties}"))]
    changes = []
    for path, params in sections:
        if first != old:
            changes += _renamed(params, path, old, first)
        properties = params['properties']
        for other in reversed(others):
            _insert_after(properties, first, other, first)
        if others:
            changes.append(Change(path + ['properties'], (
                f"{', '.join(others)} tied to {first} (they shared {old})")))
    return changes


def _sections(
        config: Mapping[str, Any]
) -> list[tuple[list[str | int], MutableMapping[str, Any]]]:
    """Return the params sections of a configuration with properties."""
    return [
        (path, params) for path, params in items(config, 'params')
        if isinstance(params.get('properties'), MutableMapping)]


def _renamed(
        params: MutableMapping[str, Any],
        path: list[str | int],
        old: str,
        new: str
) -> list[Change]:
    """
    Rename a parameter in a params section, in place: the keys of its
    properties (e.g. posa, posa[1:]) and its modes, and the name in its
    expressions (see _renamed_in_expressions); the function of its
    transforms is code, which the changes say to check by hand.
    """
    changes = []
    for option in ('properties', 'modes'):
        mapping = params.get(option)
        if not isinstance(mapping, MutableMapping):
            continue
        renamed = [
            key for key in list(mapping)
            if key == old or key.startswith(f'{old}[')]
        for key in renamed:
            rename(mapping, key, new + key[len(old):])
            changes.append(Change(
                path + [option], f"{key} -> {new}{key[len(old):]}"))
    if 'transforms' in params:
        changes.append(Change(path + ['transforms'], (
            f"by hand: rename {old} to {new} in the transforms function")))
    return changes + _renamed_in_expressions(params, path, old, new)


def _renamed_in_expressions(
        params: MutableMapping[str, Any],
        path: list[str | int],
        old: str,
        new: str
) -> list[Change]:
    """
    Rename a parameter or a constant in the expressions of a params
    section, in place (e.g. of ties, derived values and joint priors).
    """
    pattern = re.compile(rf'(?<![\w.]){re.escape(old)}(?!\w)')
    changes = []
    # Only the options of expressions: others hold e.g. file names
    for key in [k for k in EXPRESSIONS if k in params]:
        params[key], found = replaced(params[key], pattern, new)
        if found:
            changes.append(Change(path + [key], (
                f"{old} -> {new} in its expressions")))
    return changes


def _insert_after(
        mapping: MutableMapping[str, Any], after: str, key: str, value: Any
) -> None:
    """
    Set a key of a dictionary in place, after the key after (or last if
    it has none), unless it has the key already.
    """
    if key in mapping:
        return
    keys = list(mapping)
    position = keys.index(after) + 1 if after in keys else len(keys)
    if isinstance(mapping, CommentedMap):
        mapping.insert(position, key, value)
        return
    pairs = list(mapping.items())
    pairs.insert(position, (key, value))
    mapping.clear()
    mapping.update(pairs)


class RingNodes(Rule):
    """
    The radial nodes of a component (the rings of a disk, the shells of a
    cone, the nodes of the traits of a spheroid) are one option, rnodes
    (2026-11-04): a list, as before, or radii evenly spaced, {max, sep}
    (were rnmax and rnsep) or {max, len} (were rnmax and rnlen).
    """

    _OLD = ('rnmax', 'rnsep', 'rnlen')
    _FORMS = {
        frozenset({'rnmax', 'rnsep'}): dict(rnmax='max', rnsep='sep'),
        frozenset({'rnmax', 'rnlen'}): dict(rnmax='max', rnlen='len')}

    @staticmethod
    def version() -> int:
        return 4

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, galaxy in items(config, 'galaxies'):
            changes += self._walk(galaxy, path)
        return changes

    def _walk(self, value: Any, path: list[str | int]) -> list[Change]:
        """Rewrite the nodes in a value and its items, in place."""
        changes = []
        if isinstance(value, MutableMapping):
            given = frozenset(k for k in self._OLD if k in value)
            if given:
                changes.append(self._rewrite(value, given, path))
            for key, item in value.items():
                changes += self._walk(item, path + [key])
        elif isinstance(value, list):
            for i, item in enumerate(value):
                changes += self._walk(item, path + [i])
        return changes

    def _rewrite(
            self,
            mapping: MutableMapping[str, Any],
            given: frozenset[str],
            path: list[str | int]
    ) -> Change:
        """Rewrite the nodes of one mapping, in place."""
        names = self._FORMS.get(given)
        if names is None or 'rnodes' in mapping:
            options = sorted(given | ({'rnodes'} & set(mapping)))
            return Change(path, (
                f"by hand: give rnodes alone, a list or {{max, sep}} or "
                f"{{max, len}}; given: {options}"))
        nodes = flow({new: mapping[old] for old, new in names.items()})
        rename(mapping, 'rnmax', 'rnodes')
        mapping['rnodes'] = nodes
        for old in given - {'rnmax'}:
            del mapping[old]
        return Change(path, (
            f"{', '.join(sorted(given))} -> rnodes: "
            f"{{{', '.join(names.values())}}}"))


# The rules, in the order of the changes: the rings that OwnGeometries
# gives the components are rewritten by RingNodes
RULES: tuple[Rule, ...] = (
    OwnGeometries(),
    RingNodes(),
)
