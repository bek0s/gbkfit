"""
The rules of version 8 of the format (2026-11-11 on), which rewrite
configurations of version 7.
"""

import re
from collections.abc import Mapping, MutableMapping
from typing import Any

from ruamel.yaml.comments import CommentedMap

from .rules import Change, Rule, components, items, rename


__all__ = [
    'RULES',
    'FlatComponents',
    'TracedMass',
    'TraitParameterNames'
]


# The options of params sections that hold names of parameters
_EXPRESSIONS = ('properties', 'derived', 'priors')


def _splice(
        mapping: MutableMapping[str, Any], key: str,
        pairs: list[tuple[str, Any]]
) -> None:
    """
    Replace a key of a dictionary in place by the given pairs, at its
    position (with the comments of the other keys, of YAML).
    """
    position = list(mapping).index(key)
    mapping.pop(key)
    if isinstance(mapping, CommentedMap):
        for offset, (name, value) in enumerate(pairs):
            mapping.insert(position + offset, name, value)
    else:
        old = list(mapping.items())
        mapping.clear()
        mapping.update(old[:position] + pairs + old[position:])


class FlatComponents(Rule):
    """
    A component's type says which options it has (2026-11-11), so the
    groups of options without a choice of kind are gone:

    - disks and cones take their traits as options of their own: the
      density polar and height traits (density: {ptraits, htraits}) are
      rptraits and rhtraits, and the traits and the equilibrium of the
      kinematics (kinematics: {type: traits, ...}) are on the component;
    - a spheroid's type says how its stars move: the kinematics jeans is
      the type jeans_spheroid, schwarzschild (with norbits, turns and
      seed) the type orbit_spheroid, and traits (with their nodes) the
      type spheroid, with those options on the component.
    """

    _NODAL = ('thin_disk', 'thick_disk', 'cone')

    @staticmethod
    def version() -> int:
        return 8

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for galaxy_path, galaxy in items(config, 'galaxies'):
            for path, component in components(galaxy):
                kind = component.get('type')
                if kind in self._NODAL:
                    changes += self._density(component, galaxy_path + path)
                if kind in (*self._NODAL, 'spheroid'):
                    changes += self._kinematics(
                        component, galaxy_path + path)
        return changes

    @staticmethod
    def _density(
            component: MutableMapping[str, Any], path: list[str | int]
    ) -> list[Change]:
        """Take the traits of the density of a component up, in place."""
        density = component.get('density')
        if not isinstance(density, Mapping):
            return []
        names = dict(ptraits='rptraits', htraits='rhtraits')
        pairs = [(names.get(key, key), value)
                 for key, value in density.items()]
        _splice(component, 'density', pairs)
        return [Change(
            path, "density: {ptraits, htraits} -> rptraits, rhtraits")]

    @staticmethod
    def _kinematics(
            component: MutableMapping[str, Any], path: list[str | int]
    ) -> list[Change]:
        """
        Take the options of the kinematics of a component up, in place,
        and give a spheroid the type of its kinematics.
        """
        kinematics = component.get('kinematics')
        if not isinstance(kinematics, Mapping):
            return []
        kind = kinematics.get('type')
        pairs = [(key, value) for key, value in kinematics.items()
                 if key != 'type']
        _splice(component, 'kinematics', pairs)
        changes = [Change(path, (
            f"kinematics: {{type: {kind}, ...}} -> its options on the "
            f"component"))]
        if component.get('type') == 'spheroid':
            new = dict(jeans='jeans_spheroid', schwarzschild='orbit_spheroid')
            if kind in new:
                component['type'] = new[kind]
                changes.append(Change(path, f"type spheroid -> {new[kind]}"))
        return changes


class TraitParameterNames(Rule):
    """
    The parameters of traits are named after their kind, one code per
    kind (2026-11-11): those of the density traits rpt_* and rht_* (were
    pt_* and ht_*); unnamed traits are numbered from the second, so that
    a trait added at the end renames none (rpt, rpt2, rpt3: were pt, pt1,
    pt2); and named traits keep their kind (rpt_ring_*: was ring_*), so
    that traits of other kinds can share a name. The coefficients of line
    opacity are numbered so too (op, op2: were op, op1). In the params
    sections of the configuration, the parameters are renamed in their
    properties and modes and in expressions; a prefix that two galaxies
    of a project rename in different ways is renamed by hand.
    """

    # The options of the traits of the components (of version 8: this
    # rule comes after FlatComponents), and the old prefix of each kind
    _KINDS = dict(
        rptraits=('rpt', 'pt'), rhtraits=('rht', 'ht'),
        vptraits=('vpt', 'vpt'), vhtraits=('vht', 'vht'),
        dptraits=('dpt', 'dpt'), dhtraits=('dht', 'dht'),
        zptraits=('zpt', 'zpt'), sptraits=('spt', 'spt'))
    _NODAL = (
        'thin_disk', 'thick_disk', 'cone', 'spheroid', 'jeans_spheroid',
        'orbit_spheroid')

    @staticmethod
    def version() -> int:
        return 8

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        # The renames of every galaxy, applied to every params section: a
        # galaxy alone or with a name owns its parameters, and the
        # galaxies of a project without names (whose params are in its
        # setups) are renamed alike, unless two of them rename a prefix
        # in different ways (which is changed by hand)
        changes = []
        renames: dict[str, str] = {}
        conflicts: dict[str, set[str]] = {}
        for path, galaxy in items(config, 'galaxies'):
            for old, new in self._galaxy(galaxy).items():
                if renames.setdefault(old, new) != new:
                    conflicts.setdefault(old, {renames[old]}).add(new)
        for old, news in conflicts.items():
            del renames[old]
            changes.append(Change(['galaxies'], (
                f"by hand: the galaxies rename the parameters {old}* "
                f"differently ({', '.join(sorted(news))}); rename them in "
                f"the params of the setups of each")))
        if renames:
            for params_path, params in items(config, 'params'):
                if isinstance(params.get('properties'), MutableMapping):
                    changes += _renamed_prefixes(
                        params, params_path, renames)
        return changes

    def _galaxy(self, galaxy: Mapping[str, Any]) -> dict[str, str]:
        """
        Return the old and the new prefix of the parameters of each trait
        of a galaxy whose name changes (e.g. 'disk_pt1_': 'disk_rpt2_').
        """
        galaxy_prefix = f"{galaxy['name']}_" if 'name' in galaxy else ''
        found = [c for _, c in components(galaxy)]
        names = [c.get('name') for c in found]
        # (of version 7: the first unnamed component has no prefix, the
        # others cmp1, cmp2, ...; all or none have names)
        prefixes = [f'{n}_' for n in names] if all(names) else [
            f'cmp{i}_' if i else '' for i in range(len(found))]
        renames = {}
        for prefix, component in zip(prefixes, found):
            if component.get('type') not in self._NODAL:
                continue
            base = galaxy_prefix + prefix
            for key, (kind, old_kind) in self._KINDS.items():
                renames |= _trait_renames(
                    component.get(key), kind, old_kind, base)
            opacity = component.get('opacity')
            if isinstance(opacity, Mapping) and opacity.get('type') == 'line':
                renames |= _opacity_renames(opacity.get('coefficients'), base)
        return {old: new for old, new in renames.items() if old != new}


def _as_list(value: Any) -> list[Any]:
    """Return the items of a list option (one item, or none, as a list)."""
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _trait_renames(
        value: Any, kind: str, old_kind: str, base: str, suffix: str = ''
) -> dict[str, str]:
    """
    Return the old and the new prefix of the parameters of each trait of a
    list option, of the given kind and old prefix, after the prefix base
    (of the galaxy and the component) and with the given suffix (of a line
    of line opacity).
    """
    traits = _as_list(value)
    names = [t.get('name') if isinstance(t, Mapping) else None
             for t in traits]
    if names and all(names):
        return {f'{base}{name}_{suffix}': f'{base}{kind}_{name}_{suffix}'
                for name in names}
    return {
        f'{base}{old_kind}{i or ""}_{suffix}':
            f'{base}{kind}{i + 1 if i else ""}_{suffix}'
        for i in range(len(traits))}


def _opacity_renames(coefficients: Any, base: str) -> dict[str, str]:
    """
    Return the renames of the coefficient traits of line opacity: those of
    one line without a name, or of each line of a mapping.
    """
    if isinstance(coefficients, Mapping):
        renames = {}
        for line, traits in coefficients.items():
            renames |= _trait_renames(traits, 'op', 'op', base, f'{line}_')
        return renames
    return _trait_renames(coefficients, 'op', 'op', base)


def _renamed_prefixes(
        params: MutableMapping[str, Any],
        path: list[str | int],
        renames: dict[str, str]
) -> list[Change]:
    """
    Rename the parameters of a params section of the given prefixes, in
    place, at once (so that 'pt1_' -> 'rpt2_' and 'pt2_' -> 'rpt3_' do not
    chain): the keys of its properties (e.g. pt_a, pt_a[1:]) and its
    modes, and the names in its expressions; the function of its
    transforms is code, which the changes say to check by hand.
    """
    changes = []
    # (the longest prefix first: 'pt_ring_' before 'pt_')
    olds = sorted(renames, key=len, reverse=True)
    for option in ('properties', 'modes'):
        mapping = params.get(option)
        if not isinstance(mapping, MutableMapping):
            continue
        for key in list(mapping):
            old = next((o for o in olds if key.startswith(o)), None)
            if old is not None:
                new = renames[old] + key[len(old):]
                rename(mapping, key, new)
                changes.append(Change(path + [option], f"{key} -> {new}"))
    if 'transforms' in params:
        changes.append(Change(path + ['transforms'], (
            "by hand: rename the parameters of traits in the transforms "
            "function (e.g. pt_a -> rpt_a, pt1_a -> rpt2_a)")))
    pattern = re.compile(
        rf'(?<![\w.])({"|".join(re.escape(o) for o in olds)})(?=\w)')
    for key in [k for k in _EXPRESSIONS if k in params]:
        params[key], replaced = _replaced(
            params[key], pattern, lambda m: renames[m.group(1)])
        if replaced:
            changes.append(Change(path + [key], (
                "the parameters of traits renamed in its expressions")))
    return changes


def _replaced(value: Any, pattern: re.Pattern, new) -> tuple[Any, bool]:
    """
    Return a value with the matches of a pattern replaced in its strings
    and those of its items (in place for lists and dictionaries), and
    whether any was.
    """
    if isinstance(value, str):
        result = pattern.sub(new, value)
        return result, result != value
    replaced = False
    if isinstance(value, MutableMapping):
        for key in list(value):
            value[key], one = _replaced(value[key], pattern, new)
            replaced |= one
    elif isinstance(value, list):
        for i, item in enumerate(value):
            value[i], one = _replaced(item, pattern, new)
            replaced |= one
    return value, replaced


class TracedMass(Rule):
    """
    The mass that traces a component has types of its own (2026-11-11):
    a mass disk whose surface density was that of a component (surface:
    {type: component, component: C, rstep}) is a traced_disk of the
    component C (with its rstep), in the layer of C unless its vertical
    structure is another (a law, or the layer of another component, which
    it keeps); and light_spheroid is traced_spheroid. A disk mass takes
    surface laws only.
    """

    @staticmethod
    def version() -> int:
        return 8

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for galaxy_path, galaxy in items(config, 'galaxies'):
            for path, mass in items(galaxy, 'mass'):
                path = galaxy_path + path
                kind = mass.get('type')
                if kind == 'light_spheroid':
                    mass['type'] = 'traced_spheroid'
                    changes.append(Change(
                        path, "type light_spheroid -> traced_spheroid"))
                elif kind == 'disk':
                    changes += self._disk(mass, path)
        return changes

    @staticmethod
    def _disk(
            mass: MutableMapping[str, Any], path: list[str | int]
    ) -> list[Change]:
        """Rewrite a mass disk that follows a component, in place."""
        surface = mass.get('surface')
        if not isinstance(surface, Mapping) \
                or surface.get('type') != 'component':
            return []
        component = surface.get('component')
        rstep = surface.get('rstep')
        vertical = mass.get('vertical')
        own = isinstance(vertical, Mapping) \
            and vertical.get('type') == 'component' \
            and vertical.get('component') == component \
            and vertical.get('rstep') == rstep
        pairs = [('type', 'traced_disk')] + [
            (key, value) for key, value in mass.items()
            if key not in ('type', 'surface', 'vertical')]
        pairs.append(('component', component))
        if rstep is not None:
            pairs.append(('rstep', rstep))
        if vertical is not None and not own:
            pairs.append(('vertical', vertical))
        mass.clear()
        mass.update(pairs)
        what = 'in its layer' if own else 'with its vertical structure'
        return [Change(path, (
            f"type disk of the surface of the component {component!r} -> "
            f"traced_disk, {what}"))]


# The rules, in the order of the changes: TraitParameterNames reads the
# trait options that FlatComponents takes up
RULES: tuple[Rule, ...] = (
    FlatComponents(),
    TraitParameterNames(),
    TracedMass(),
)
