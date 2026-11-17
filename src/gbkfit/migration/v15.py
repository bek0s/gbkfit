"""
The rules of version 15 of the format (2026-11-17 on), which rewrite
configurations of version 14: gravitational lenses are a section of
their own, beside the galaxies, not the foreground of an observation.
"""

import re
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from ruamel.yaml.comments import CommentedSeq

from gbkfit.config import ConfigError
from .rules import (
    EXPRESSIONS, Change, PiecesView, Rule, ViewedPiece, flow, items, rename,
    replaced)
from .v4 import _insert_after


__all__ = [
    'RULES',
    'LensesSection'
]


# The name of the lens that the lenses of the observations become
_LENS = 'lens'

# The parameters of the masses of the lenses of version 14, by type
_MASS_PARAMS = dict(
    sie=('theta_e', 'q', 'posa', 'xpos', 'ypos'),
    epl=('theta_e', 'slope', 'q', 'posa', 'xpos', 'ypos'),
    nfw=('kappa_s', 'r_s', 'q', 'posa', 'xpos', 'ypos'),
    multipole=('theta_e', 'a', 'posa', 'xpos', 'ypos'),
    shear=('gamma', 'posa'))

# The masses that multipoles can deviate (their shapes), and the
# parameters of a multipole that are those of its shape (tied to them)
_SHAPES = ('sie', 'epl')
_SHAPE_TIES = ('theta_e', 'xpos', 'ypos')


class LensesSection(Rule):
    """
    The lens of an observation (its option foreground) is a lens of the
    section 'lenses' (2026-11-17), shared by every observation of the
    galaxies behind it, at a redshift; its masses are its pieces:

    - identical lenses of several observations become one, 'lens'
      (their grids of the source plane aside: the grids of the planes
      behind lenses are automatic); differing ones stop the migration (a
      lens is one for all the observations);
    - a multipole is the option multipoles of the shape (SIE or EPL) that
      its ties name (its theta_e, xpos and ypos those of the shape; its a
      and posa become a{m} and posa{m} of the shape); one tied to no shape
      stops the migration;
    - every mass was the deflection of the light of the galaxies it bent,
      quoted for them: a deflection map is quoted (quoted: true), as the
      shapes are; it loses its option scaled and its parameter scale
      (which scaled a map made for another source, whose redshift is to
      be given by hand: the redshifts scale it);
    - the parameters '{observation}_lens_{mass}_{p}' are those of the
      lens, 'lens_{piece}_{p}' ('lens_{p}' for one piece; the pieces
      without names are named 'mass{i}' when there are several); a joint
      fit's ties of one to the other and equal values become one, and
      conflicting ones stop the migration;
    - the redshift of the lens is to be given by hand, and the redshift
      of the source of its quoted strengths (source_redshift) is that of
      the galaxies it bent, if they have one, else by hand; every galaxy
      needs a redshift.

    In a project, the observation pieces lose their lens, which is the
    piece lenses/lens.yaml, and the setups that name them list it.
    """

    @staticmethod
    def version() -> int:
        return 15

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        lensed = []
        for path, observation in items(config, 'observations'):
            if 'foreground' not in observation:
                continue
            foreground = observation.pop('foreground')
            lens = foreground.get('lens') \
                if isinstance(foreground, Mapping) else None
            if not isinstance(observation, ViewedPiece):
                changes.append(Change(
                    path + ['foreground'],
                    "-> section lenses" if isinstance(lens, Mapping)
                    else "removed (no lens)"))
            if isinstance(lens, Mapping):
                lensed.append((path, observation, lens))
        if not lensed:
            return changes
        lens, renames, tied, notes = _merged_lens(config, lensed)
        for params_path, params in items(config, 'params'):
            changes += _renamed(params, params_path, renames, tied)
        changes += [
            Change(path + ['redshift'], (
                "by hand: give the galaxy its redshift: with lenses, the "
                "redshifts order the galaxies and the lenses"))
            for path, galaxy in items(config, 'galaxies')
            if not isinstance(galaxy, ViewedPiece)
            and 'redshift' not in galaxy]
        if isinstance(config, PiecesView):
            config.setdefault('lenses', []).append(lens)
            index = len(config['lenses']) - 1
            return changes + [Change(['lenses', index], (
                f"the lens of the observation pieces, as the piece "
                f"{_LENS!r}"))] + [
                Change(['lenses', index] + path, note)
                for path, note in notes]
        if all(isinstance(observation, ViewedPiece)
               for _, observation, _ in lensed):
            # (the lens piece, which the pieces of the project give)
            _insert_after(config, 'galaxies', 'lenses', [_LENS])
            return changes + [Change(
                ['lenses'], f"the lens piece {_LENS!r} of the project")]
        _insert_after(config, 'galaxies', 'lenses', [lens])
        return changes + [Change(['lenses'], "the lens of the observations")] \
            + [Change(['lenses', 0] + path, note) for path, note in notes]


def _merged_lens(
        config: Mapping[str, Any],
        lensed: Sequence[tuple[list[str | int], Mapping[str, Any],
                               Mapping[str, Any]]]
) -> tuple[dict[str, Any], dict[str, str], set[str],
           list[tuple[list[str | int], str]]]:
    """
    Return the lens of the section 'lenses' that the lenses of version 14
    of the observations of a configuration (with their paths) become, the
    new names of their parameters, by the old ones, the old names of the
    parameters of multipoles tied to their shapes (gone with them), and
    the notes of the migration (paths and messages).

    Raises
    ------
    ConfigError
        If the lenses differ, or a multipole is tied to no shape.
    """
    first_path, _, first = lensed[0]
    masses = _listed(first.get('masses'))
    for path, _, lens in lensed[1:]:
        if _listed(lens.get('masses')) != masses:
            raise ConfigError(
                f"the observations have different lenses (their masses "
                f"differ from those of {_shown(first_path)}): a lens is now "
                f"one for every observation (the section lenses); make "
                f"them one by hand", path + ['foreground', 'lens'])
    names = [m.get('name') if isinstance(m, Mapping) else None
             for m in masses]
    # (the prefixes of their parameters of version 14: their names if all
    # have names, else 'mass{i}', none for the first)
    old = [f'{n}_' for n in names] if all(names) else [
        f'mass{i}_' if i else '' for i in range(len(masses))]
    params = _all_properties(config)
    hosts = _multipole_hosts(lensed, masses, old, params)
    kept = [i for i in range(len(masses)) if i not in hosts]
    piece_names = [
        names[i] if names[i] is not None else f'mass{i}' for i in kept]
    new = {
        i: '' if len(kept) == 1 else f'{name}_'
        for i, name in zip(kept, piece_names)}
    pieces = []
    notes = []
    for i, name in zip(kept, piece_names):
        piece = {
            key: value for key, value in masses[i].items()
            if key not in ('name', 'scaled')}
        if len(kept) > 1 or names[i] is not None:
            piece = dict(name=name) | piece
        if masses[i].get('type') == 'deflection_map':
            piece['quoted'] = True
        orders = [masses[m]['order'] for m, h in hosts.items() if h == i]
        if orders:
            piece['multipoles'] = _flow_list(orders)
        pieces.append(flow(piece))
    renames: dict[str, str] = {}
    tied: set[str] = set()
    for _, observation, _ in lensed:
        base = f"{observation.get('name')}_lens_"
        for i, mass in enumerate(masses):
            kind = mass.get('type')
            if i in hosts:
                host = f'{_LENS}_{new[hosts[i]]}'
                m = mass['order']
                renames[f'{base}{old[i]}a'] = f'{host}a{m}'
                renames[f'{base}{old[i]}posa'] = f'{host}posa{m}'
                for p in _SHAPE_TIES:
                    renames[f'{base}{old[i]}{p}'] = f'{host}{p}'
                    tied.add(f'{base}{old[i]}{p}')
                continue
            own = list(_MASS_PARAMS.get(kind, ()))
            if kind == 'deflection_map' and mass.get('shifted'):
                own += ['xpos', 'ypos']
            renames |= {
                f'{base}{old[i]}{p}': f'{_LENS}_{new[i]}{p}' for p in own}
            if kind == 'deflection_map' and mass.get('scaled'):
                tied.add(f'{base}{old[i]}scale')
    lens: dict[str, Any] = dict(name=_LENS)
    notes.append((['redshift'], (
        "by hand: give the lens its redshift (redshift): that of its plane, "
        "which orders it among the galaxies")))
    # (every piece is quoted, for the galaxies it bent)
    redshift = _common_redshift(config, [obs for _, obs, _ in lensed])
    if redshift is not None:
        lens['source_redshift'] = redshift
    else:
        notes.append((['source_redshift'], (
            "by hand: give the lens the redshift of the source its "
            "strengths are quoted for (source_redshift): that of the "
            "galaxies it bent, so that they keep their meaning")))
    lens['pieces'] = pieces if len(pieces) > 1 else pieces[0]
    if any(old_lens.get('source_size') is not None
           or old_lens.get('source_step') is not None
           for _, _, old_lens in lensed):
        notes.append(([], (
            "the grids of the planes behind the lens are automatic (see "
            "the option source_grid); source_size and source_step are "
            "gone")))
    if any(masses[i].get('type') == 'deflection_map'
           and not masses[i].get('shifted') for i in kept):
        notes.append(([], (
            "the deflection maps that are not shifted now move with the "
            "offsets of each observation, as everything on the sky")))
    if any(masses[i].get('type') == 'deflection_map'
           and masses[i].get('scaled') for i in kept):
        notes.append((['pieces'], (
            "by hand: the option scaled of the deflection maps and their "
            "parameter scale are gone; a scaled map was made for a source "
            "of another redshift than that of the galaxies it bent: give "
            "it a lens of its own at the redshift of this one (lenses at "
            "one redshift add), quoted for the redshift of that source "
            "(source_redshift)")))
    if any(masses[h].get('type') == 'epl' for h in hosts.values()):
        notes.append(([], (
            "the multipoles of an EPL now fall as its convergence (with "
            "its slope), not as that of an SIE")))
    notes.append(([], (
        "the light of the galaxies in front of the lens, and at its "
        "redshift (e.g. the lens galaxy), is no longer bent by it")))
    return lens, renames, tied, notes


def _multipole_hosts(
        lensed: Sequence[tuple[list[str | int], Mapping[str, Any],
                               Mapping[str, Any]]],
        masses: Sequence[Mapping[str, Any]],
        old: Sequence[str],
        params: Mapping[str, Any]
) -> dict[int, int]:
    """
    Return the shape (SIE or EPL) that each multipole of the masses of a
    lens deviates (both by index): the one that its ties name, in the
    parameters of every observation of the lens (see LensesSection).

    Raises
    ------
    ConfigError
        If a multipole is tied to no shape, or two of one order to one.
    """
    shapes = [i for i, m in enumerate(masses) if m.get('type') in _SHAPES]
    hosts: dict[int, int] = {}
    for i, mass in enumerate(masses):
        if mass.get('type') != 'multipole':
            continue
        found = None
        for path, observation, _ in lensed:
            base = f"{observation.get('name')}_lens_"
            ties = [params.get(f'{base}{old[i]}{p}') for p in _SHAPE_TIES]
            named = [
                h for h in shapes
                if all(isinstance(tie, str)
                       and tie.strip() == f'{base}{old[h]}{p}'
                       for tie, p in zip(ties, _SHAPE_TIES))]
            if not named or found not in (None, named[0]):
                raise ConfigError(
                    f"the multipole {mass.get('name') or i} of the lens is "
                    f"not tied to a shape (its {', '.join(_SHAPE_TIES)} "
                    f"tied to those of an SIE or an EPL of the lens): "
                    f"multipoles are now the option multipoles of a shape; "
                    f"make it one by hand", path + ['foreground', 'lens'])
            found = named[0]
        orders = [masses[m]['order'] for m, h in hosts.items() if h == found]
        if mass.get('order') in orders:
            raise ConfigError(
                f"two multipoles of order {mass.get('order')} deviate one "
                f"shape of the lens; make them one by hand",
                lensed[0][0] + ['foreground', 'lens'])
        hosts[i] = found
    return hosts


def _renamed(
        params: MutableMapping[str, Any],
        path: list[str | int],
        renames: Mapping[str, str],
        tied: set[str]
) -> list[Change]:
    """
    Rename the parameters of the lenses of a params section, in place:
    drop those that are gone (tied, e.g. the centre of a multipole, its
    shape's now), rename the names in its expressions, and the keys of
    its properties and modes, of which those that become one must be the
    same, or ties of one to the other (see LensesSection).

    Raises
    ------
    ConfigError
        If parameters of several observations become one parameter of
        the lens, and differ.
    """
    changes = []
    for option in ('properties', 'modes'):
        mapping = params.get(option)
        if not isinstance(mapping, MutableMapping):
            continue
        for key in [k for k in mapping if k.split('[')[0] in tied]:
            del mapping[key]
            changes.append(Change(path + [option, key], (
                "dropped: of the shape of its multipole, or the scale of a "
                "map, which the redshifts give")))
    pattern = re.compile(
        rf'(?<![\w.])({"|".join(re.escape(o) for o in renames)})(?!\w)')
    for key in [k for k in EXPRESSIONS if k in params]:
        params[key], found = replaced(
            params[key], pattern, lambda m: renames[m.group(1)])
        if found:
            changes.append(Change(path + [key], (
                "the parameters of the lens renamed in its expressions")))
    for option in ('properties', 'modes'):
        mapping = params.get(option)
        if not isinstance(mapping, MutableMapping):
            continue
        targets: dict[str, list[str]] = {}
        for key in list(mapping):
            name = key.split('[')[0]
            targets.setdefault(
                renames.get(name, name) + key[len(name):], []).append(key)
        for new, keys in targets.items():
            # (a tie of one observation to the other is the one parameter)
            values = [
                k for k in keys
                if not (isinstance(mapping[k], str)
                        and mapping[k].strip() == new)] or keys[:1]
            if any(mapping[k] != mapping[values[0]] for k in values[1:]):
                raise ConfigError(
                    f"the parameters {keys} become one, {new!r}, of the "
                    f"lens that the observations share, and differ; make "
                    f"them one by hand", path + [option])
            for key in keys:
                if key != values[0]:
                    del mapping[key]
                    changes.append(Change(
                        path + [option, key], f"-> {new} (one for all)"))
            if values[0] != new:
                rename(mapping, values[0], new)
                changes.append(Change(
                    path + [option], f"{values[0]} -> {new}"))
    return changes


def _all_properties(config: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return the properties of the parameters of the params sections of a
    configuration (those of later ones last), by name without subscripts.
    """
    properties = {}
    for _, params in items(config, 'params'):
        given = params.get('properties')
        if isinstance(given, Mapping):
            properties |= {
                key.split('[')[0]: value for key, value in given.items()}
    return properties


def _common_redshift(
        config: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]
) -> float | None:
    """
    Return the redshift of the galaxies that observations of a
    configuration see (the galaxies they name, or its one galaxy), if
    they all have one, the same; None otherwise.
    """
    galaxies = [galaxy for _, galaxy in items(config, 'galaxies')]
    seen = []
    for observation in observations:
        names = observation.get('galaxies')
        if names is None:
            seen += galaxies[:1] if len(galaxies) == 1 else [None]
        else:
            by_name = {g.get('name'): g for g in galaxies}
            seen += [by_name.get(name) for name in _listed(names)]
    redshifts = {
        None if galaxy is None else galaxy.get('redshift')
        for galaxy in seen}
    if len(redshifts) != 1 or None in redshifts:
        return None
    return redshifts.pop()


def _flow_list(values: list[Any]) -> CommentedSeq:
    """Return a list as one of the flow style of YAML."""
    result = CommentedSeq(values)
    result.fa.set_flow_style()
    return result


def _listed(value: Any) -> list[Any]:
    """Return a value as a list (a list, or one item)."""
    return list(value) if isinstance(value, list) else [value]


def _shown(path: Sequence[str | int]) -> str:
    """Return a path in a configuration as text."""
    return '.'.join(str(part) for part in path)


RULES: tuple[Rule, ...] = (
    LensesSection(),
)
