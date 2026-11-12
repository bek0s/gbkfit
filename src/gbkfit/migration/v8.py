"""
The rules of version 8 of the format (2026-11-11 on), which rewrite
configurations of version 7.
"""

import re
from collections.abc import Mapping, MutableMapping
from typing import Any

from ruamel.yaml.comments import CommentedMap

import gbkfit.cosmology
from gbkfit.config import ConfigError
from .rules import (
    EXPRESSIONS, Change, PiecesView, Rule, components, items, rename,
    renamed_params, replaced)


__all__ = [
    'RULES',
    'AssumedDistances',
    'Cosmologies',
    'FlatComponents',
    'Halos',
    'TracedMass',
    'TraitParameterNames',
    'VerticalBalance'
]


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
        if 'kinematics' in component and kinematics is None:
            del component['kinematics']
            return [Change(path, "kinematics: null removed")]
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
    for key in [k for k in EXPRESSIONS if k in params]:
        params[key], found = replaced(
            params[key], pattern, lambda m: renames[m.group(1)])
        if found:
            changes.append(Change(path + [key], (
                "the parameters of traits renamed in its expressions")))
    return changes


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
        # (the layer of the same component, at the subnodes of the surface
        # density: those of the vertical structure, if others, differed
        # by little)
        own = isinstance(vertical, Mapping) \
            and vertical.get('type') == 'component' \
            and vertical.get('component') == component
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


# The profiles of halos (of v7 and v8: nfw_truncated is truncated_nfw)
_HALO_PROFILES = frozenset({
    'nfw', 'burkert', 'pseudo_isothermal', 'einasto', 'zhao', 'dekel_zhao',
    'core_nfw', 'nfw_truncated', 'truncated_nfw'})

# The parameterisations of the dark fractions of v7, by the parameter
# that they solve for (the mass, or its velocity; the scale density)
_DARK_FRACTIONS = dict(
    log_m='dark_fraction_concentration',
    v='dark_fraction_concentration',
    log_rho='dark_fraction_radius')


class Halos(Rule):
    """
    Halos (2026-11-11): a mass spheroid of the profile of a halo, or
    that responds to baryons or is given by a dark fraction, is a halo
    (type halo).
    - The parameters of its profile are its parameterisation, and the
      overdensity of its profile is that of the halo.
    - A response is a contraction; the response sidm is sidm: true, with
      the contraction that it nested.
    - A dark fraction that solved for the mass (log_m, or v) is the
      parameterisation dark_fraction_concentration, and one that solved
      for the scale density (log_rho) dark_fraction_radius; its bounds
      go. Others (that solved for c or a shape parameter) are changed by
      hand, as is a spheroid of a luminous profile with a response or a
      dark fraction.
    - The profile nfw_truncated is truncated_nfw, and baryons that
      nothing used are removed (they are an error).
    - Parameters are renamed in the params sections, as
      TraitParameterNames renames those of traits: the inner slope of a
      dekel_zhao profile is gamma (was a), the radius of a dark fraction
      rfdm (was r_fdm), and the age of an SIDM halo age (was t_age).
    - The contraction gnedin04 has the parameters a0 and w, added at the
      values it had (0.85 and 0.8) to the params sections that give the
      other parameters of its halo.
    """

    @staticmethod
    def version() -> int:
        return 8

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        renames: dict[str, str] = {}
        additions: dict[str, tuple[dict[str, float], str]] = {}
        for galaxy_path, galaxy in items(config, 'galaxies'):
            galaxy_prefix = f"{galaxy['name']}_" if 'name' in galaxy else ''
            for path, mass in items(galaxy, 'mass'):
                if mass.get('type') != 'spheroid':
                    continue
                path = galaxy_path + path
                prefix = f"{galaxy_prefix}mass_{mass.get('name')}_"
                renames |= _halo_renames(mass, prefix)
                if _gnedin04(mass):
                    additions[prefix] = (_GNEDIN04, (
                        "the parameters of the contraction gnedin04, at "
                        "its values before"))
                changes += _halo(mass, path)
        for params_path, params in items(config, 'params'):
            if isinstance(params.get('properties'), MutableMapping):
                if renames:
                    changes += renamed_params(
                        params, params_path, renames, 'halos')
                changes += _added_names(params, params_path, additions)
        return changes


# The parameters of the contraction gnedin04, at the values of v7
_GNEDIN04 = dict(a0=0.85, w=0.8)


def _gnedin04(mass: Mapping[str, Any]) -> bool:
    """
    Check whether a mass spheroid of v7 contracts as gnedin04 (by its
    response, or the contraction of its response sidm).
    """
    response = mass.get('response')
    if isinstance(response, Mapping) and response.get('type') == 'sidm':
        response = response.get('contraction')
    return isinstance(response, Mapping) and response.get('type') == 'gnedin04'


def _added_names(
        params: MutableMapping[str, Any],
        path: list[str | int],
        additions: dict[str, tuple[dict[str, Any], str]]
) -> list[Change]:
    """
    Add parameters to the properties of a params section, in place: of
    each prefix (e.g. of a halo) whose parameters it gives, the
    parameters of the given names and values, after its last one, unless
    it gives them, with why (e.g. "the parameters of the contraction
    gnedin04, at its values before").
    """
    changes = []
    properties = params['properties']
    for prefix, (values, why) in additions.items():
        keys = [key for key in properties if key.startswith(prefix)]
        missing = [(prefix + name, value) for name, value in values.items()
                   if prefix + name not in properties]
        if keys and missing:
            _splice(properties, keys[-1],
                    [(keys[-1], properties[keys[-1]])] + missing)
            changes.append(Change(path + ['properties'], (
                f"{', '.join(name for name, _ in missing)} added, {why}")))
    return changes


def _halo_renames(
        mass: Mapping[str, Any], prefix: str) -> dict[str, str]:
    """
    Return the renames of the parameters of a mass spheroid of v7 (see
    Halos), of the prefix of its parameters.
    """
    renames = {}
    profile = mass.get('profile')
    if isinstance(profile, Mapping) and profile.get('type') == 'dekel_zhao':
        renames[f'{prefix}a'] = f'{prefix}gamma'
    if mass.get('dark_fraction') is not None:
        renames[f'{prefix}r_fdm'] = f'{prefix}rfdm'
    response = mass.get('response')
    if isinstance(response, Mapping) and response.get('type') == 'sidm':
        renames[f'{prefix}t_age'] = f'{prefix}age'
    return renames


def _halo(mass: MutableMapping[str, Any], path: list) -> list[Change]:
    """
    Rewrite a mass spheroid of v7 in place, as a halo if it is one (see
    Halos), and return the changes.
    """
    changes = []
    profile = mass.get('profile')
    if not isinstance(profile, MutableMapping):
        profile = {}
    kind = profile.get('type')
    if kind == 'nfw_truncated':
        profile['type'] = 'truncated_nfw'
        changes.append(Change(
            path + ['profile'], "type nfw_truncated -> truncated_nfw"))
    response = mass.get('response')
    dark_fraction = mass.get('dark_fraction')
    # (baryons that nothing used, which are an error now)
    if mass.get('baryons') and response is None and dark_fraction is None:
        del mass['baryons']
        changes.append(Change(path, (
            "baryons removed: neither a response nor a dark fraction used "
            "them")))
    if kind not in _HALO_PROFILES:
        if response is not None or dark_fraction is not None:
            changes.append(Change(path, (
                f"by hand: a spheroid of the profile {kind} has a response "
                f"or a dark fraction, which only halos have: make it a halo "
                f"of the profile of a halo, or leave them out")))
        return changes
    mass['type'] = 'halo'
    changes.append(Change(path, "type spheroid -> halo"))
    if 'parameters' in profile:
        rename(profile, 'parameters', 'parameterisation')
        changes.append(Change(
            path + ['profile'], "parameters -> parameterisation"))
    if 'overdensity' in profile:
        _splice(mass, 'profile', [
            ('profile', profile), ('overdensity', profile.pop('overdensity'))])
        changes.append(Change(
            path, "the overdensity of the profile is that of the halo"))
    if isinstance(response, MutableMapping):
        if response.get('type') == 'sidm':
            contraction = response.get('contraction')
            _splice(mass, 'response', (
                [] if contraction is None else [('contraction', contraction)])
                + [('sidm', True)])
            changes.append(Change(
                path, "response sidm -> sidm: true (and its contraction)"))
        else:
            rename(mass, 'response', 'contraction')
            changes.append(Change(path, "response -> contraction"))
    if isinstance(dark_fraction, Mapping):
        solve = dark_fraction.get('solve')
        parameterisation = _DARK_FRACTIONS.get(solve)
        if parameterisation is None:
            changes.append(Change(path + ['dark_fraction'], (
                f"by hand: a dark fraction sets the mass or the scale "
                f"density of a halo now (see the parameterisations "
                f"dark_fraction_concentration and dark_fraction_radius); "
                f"this one solved for {solve}")))
        else:
            del mass['dark_fraction']
            if 'parameterisation' in profile:
                profile['parameterisation'] = parameterisation
            else:
                _splice(profile, 'type', [
                    ('type', profile['type']),
                    ('parameterisation', parameterisation)])
            changes.append(Change(path, (
                f"dark_fraction (solve {solve}) -> the parameterisation "
                f"{parameterisation} of the profile; its bounds are "
                f"dropped")))
    return changes


class Cosmologies(Rule):
    """
    The cosmology (2026-11-11) is that of the setup (its section
    cosmology), not of the distances of its galaxies.
    - The cosmology of the distances of the galaxies of a configuration
      is its cosmology, {type: published, name: ...}, but Planck18 (its
      default); one whose galaxies have different cosmologies is refused
      (a ConfigError: give them one). Galaxy pieces leave theirs to the
      setups (see PiecesView).
    - The distance direct is parameter, and redshift cosmological. One by
      a redshift of its own is parameter, its value added to the params
      sections that give the parameters of the galaxy, fixed at the
      angular diameter distance of that redshift in the cosmology; its
      epoch, that redshift, is now that of the galaxy.
    """

    @staticmethod
    def version() -> int:
        return 8

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        additions: dict[str, tuple[dict[str, float], str]] = {}
        cosmologies: dict[str, list] = {}
        for galaxy_path, galaxy in items(config, 'galaxies'):
            distance = galaxy.get('distance')
            if not isinstance(distance, MutableMapping):
                continue
            path = galaxy_path + ['distance']
            name = distance.pop('cosmology', 'Planck18')
            cosmologies.setdefault(name, []).append(galaxy_path)
            kind = distance.get('type')
            if kind == 'direct':
                distance['type'] = 'parameter'
                changes.append(Change(path, "type direct -> parameter"))
            elif kind == 'redshift' and 'redshift' in distance:
                redshift = distance.pop('redshift')
                value = gbkfit.cosmology.CosmologyPublished(
                    name).angular_diameter_distance(redshift)
                distance['type'] = 'parameter'
                prefix = f"{galaxy['name']}_" if 'name' in galaxy else ''
                additions[prefix] = (dict(distance=round(value, 6)), (
                    f"fixed: the angular diameter distance (Mpc) of the "
                    f"redshift {redshift} of the distance"))
                changes.append(Change(path, (
                    f"type redshift of the redshift {redshift} -> "
                    f"parameter, fixed at {value:.6g} Mpc; its epoch is "
                    f"now the redshift of the galaxy")))
            elif kind == 'redshift':
                distance['type'] = 'cosmological'
                changes.append(Change(path, "type redshift -> cosmological"))
        for params_path, params in items(config, 'params'):
            if isinstance(params.get('properties'), MutableMapping):
                changes += _added_names(params, params_path, additions)
        changes += _setup_cosmology(config, cosmologies)
        return changes


def _setup_cosmology(
        config: MutableMapping[str, Any],
        cosmologies: dict[str, list]
) -> list[Change]:
    """
    Give a configuration of version 7 the cosmology of the distances of
    its galaxies, of their paths by its name (see Cosmologies), in place,
    and return the changes.

    Raises
    ------
    ConfigError
        If its galaxies have different cosmologies.
    """
    if isinstance(config, PiecesView):
        return [
            Change(path + ['distance'], (
                f"cosmology {name} -> that of the setups of the galaxy "
                f"(section cosmology)"))
            for name, paths in cosmologies.items() if name != 'Planck18'
            for path in paths]
    if len(cosmologies) > 1:
        raise ConfigError(
            f"the distances of the galaxies of a configuration have "
            f"different cosmologies ({', '.join(sorted(cosmologies))}); a "
            f"configuration now has one (its section cosmology): give its "
            f"galaxies one, then migrate it")
    if not cosmologies or 'Planck18' in cosmologies:
        return []
    name = next(iter(cosmologies))
    config['cosmology'] = dict(type='published', name=name)
    return [Change(['cosmology'], (
        f"the cosmology of the distances of its galaxies ({name})"))]


class AssumedDistances(Rule):
    """
    The distance that the radii of a table or a profile were measured at
    (2026-11-11), D0, is their assumed_distance (was distance): of the
    mass tables of the galaxies, of the profile observations, and of the
    data of profiles (inline, or pieces).
    """

    @staticmethod
    def version() -> int:
        return 8

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for galaxy_path, galaxy in items(config, 'galaxies'):
            for path, mass in items(galaxy, 'mass'):
                if mass.get('type') == 'table':
                    changes += _assumed(mass, galaxy_path + path)
        for path, observation in items(config, 'observations'):
            if observation.get('type') != 'profile':
                continue
            changes += _assumed(observation, path)
            data = observation.get('data')
            if isinstance(data, MutableMapping):
                changes += _assumed(data, path + ['data'])
        # (the data pieces of a project, of their type)
        for path, data in items(config, 'data'):
            if data.get('type') == 'profile':
                changes += _assumed(data, path)
        return changes


def _assumed(
        options: MutableMapping[str, Any], path: list[str | int]
) -> list[Change]:
    """
    Rename the option distance of a table or a profile assumed_distance,
    in place, and return the change.
    """
    if 'distance' not in options:
        return []
    rename(options, 'distance', 'assumed_distance')
    return [Change(path, "distance -> assumed_distance")]


class VerticalBalance(Rule):
    """
    The vertical balance of a disk is stated once (2026-11-11). The
    dispersion of the vertical equilibrium of a thick disk (equilibrium:
    {type, direction, mass}; a thin disk took none, and keeps it to be
    changed by hand, see v4) is its last dispersion trait, {type:
    equilibrium, gravity, direction}: jeans balances the gravity galaxy;
    self_gravitating the sheet of the surface density of its fluid (fluid:
    its mass); central its rotation, in a galaxy without mass (next to a
    mass, the pull of a spherical mass of its rotation is not
    expressible). The pressure support of its velocity trait 'mass' is an
    option of the disk, and the mass of the support the fluid of the disk.
    The self_gravitating support of a thin disk is the jeans support of
    the disk in the sheet of its fluid (rhtraits: {type: equilibrium,
    gravity: sheet}), which without a mass is that of the mass disks that
    follow it; on a thick disk of fitted heights, it is not expressible.
    """

    _GRAVITIES = dict(
        jeans='galaxy', self_gravitating='sheet', central='rotation')

    @staticmethod
    def version() -> int:
        return 8

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for galaxy_path, galaxy in items(config, 'galaxies'):
            mass = [item for _, item in items(galaxy, 'mass')]
            for path, component in components(galaxy):
                if component.get('type') in ('thin_disk', 'thick_disk'):
                    changes += self._disk(
                        component, galaxy_path + path, mass)
        return changes

    def _disk(
            self,
            disk: MutableMapping[str, Any],
            path: list[str | int],
            mass: list[Any]
    ) -> list[Change]:
        """
        Rewrite the balance of a disk of a galaxy of the given mass
        components (none without mass), in place.

        Raises
        ------
        ConfigError
            If it is in the central equilibrium next to a mass, its
            pressure support and its equilibrium take different masses, or
            it is a thick disk of the self_gravitating support, or a thin
            one of it without a mass disk.
        """
        massive = bool(mass)
        changes = []
        fluids = []
        sheet = False
        vptraits = disk.get('vptraits')
        for trait in vptraits if isinstance(vptraits, list) else [vptraits]:
            if not isinstance(trait, MutableMapping) \
                    or 'pressure_support' not in trait:
                continue
            support = dict(trait.pop('pressure_support'))
            if (names := support.pop('mass', None)) is not None:
                fluids.append(names)
            if support.get('type') == 'self_gravitating':
                if disk.get('type') == 'thick_disk':
                    raise ConfigError(
                        f"the disk at {path} is thick, of the "
                        f"self_gravitating pressure support, which is no "
                        f"longer expressible with heights of its own: give "
                        f"it the pressure jeans (in its heights), or the "
                        f"heights of the equilibrium sheet (rhtraits: "
                        f"{{type: equilibrium, gravity: sheet, zmax}}), "
                        f"then migrate it")
                support['type'] = 'jeans'
                sheet = True
            disk['pressure_support'] = support
            changes.append(Change(path, (
                "the pressure support of the velocity trait mass -> "
                "pressure_support of the disk")))
        equilibrium = disk.get('equilibrium')
        if disk.get('type') == 'thick_disk' \
                and isinstance(equilibrium, Mapping):
            kind = equilibrium.get('type')
            if kind == 'central' and massive:
                raise ConfigError(
                    f"the disk at {path} is in the central equilibrium of "
                    f"a galaxy with mass, which is no longer expressible: "
                    f"the pull of a spherical mass of its rotation is not "
                    f"that of the galaxy. Give it the equilibrium jeans "
                    f"(the vertical force of the mass), or remove the "
                    f"mass, then migrate it")
            if (names := equilibrium.get('mass')) is not None:
                fluids.append(names)
            trait = dict(
                type='equilibrium', gravity=self._GRAVITIES.get(kind, kind))
            if 'direction' in equilibrium:
                trait['direction'] = equilibrium['direction']
            dptraits = disk.get('dptraits')
            dptraits = list(dptraits) if isinstance(dptraits, list) \
                else [dptraits]
            # (the traits of a list are named all or none)
            if any(isinstance(item, Mapping) and 'name' in item
                   for item in dptraits):
                trait['name'] = 'equilibrium'
            disk['dptraits'] = dptraits + [trait]
            del disk['equilibrium']
            changes.append(Change(path, (
                f"equilibrium {kind} -> the last dispersion trait, "
                f"equilibrium of the gravity {trait['gravity']}")))
        names = [
            tuple(value) if isinstance(value, list) else (value,)
            for value in fluids]
        if len(set(names)) > 1:
            raise ConfigError(
                f"the pressure support and the equilibrium of the disk at "
                f"{path} take different masses ({fluids}); a disk now has "
                f"one fluid: give them one, then migrate it")
        if fluids:
            disk['fluid'] = fluids[0]
            changes.append(Change(path, (
                "the mass of its pressure support or equilibrium -> fluid")))
        if sheet:
            name = disk.get('name')
            followed = any(
                isinstance(item, Mapping) and name is not None and (
                    item.get('type') == 'traced_disk'
                    and item.get('component') == name
                    or isinstance(item.get('vertical'), Mapping)
                    and item['vertical'].get('component') == name)
                for item in mass)
            if not fluids and not followed:
                raise ConfigError(
                    f"the thin disk at {path} is of the self_gravitating "
                    f"pressure support, now the pressure jeans in the sheet "
                    f"of the surface density of its fluid, a mass disk; it "
                    f"has none: give it a mass disk that follows it, or its "
                    f"fluid (fluid: <name>), then migrate it")
            disk['rhtraits'] = dict(type='equilibrium', gravity='sheet')
            changes.append(Change(path, (
                "pressure support self_gravitating -> jeans, in the sheet "
                "of its fluid (rhtraits: equilibrium sheet)")))
        return changes


# The rules, in the order of the changes: TraitParameterNames reads the
# trait options that FlatComponents takes up
RULES: tuple[Rule, ...] = (
    FlatComponents(),
    TraitParameterNames(),
    TracedMass(),
    Halos(),
    Cosmologies(),
    AssumedDistances(),
    VerticalBalance(),
)
