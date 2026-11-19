"""
The rules of version 17 of the format (2026-11-18 on), which rewrite
configurations of version 16: the mass of a galaxy that a lens bends
light by has a place on the sky of its own, its centre mass_xpos and
mass_ypos if a mass component lies at it, and its plane mass_posa and
mass_incl if one lies in it.
"""

from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from typing import Any

from .rules import Change, Rule, components, items


__all__ = [
    'RULES',
    'MassPlace'
]


# The types of the components of version 16 that have a plane (a
# position angle and an inclination) at every value of their options
_PLANED = ('thin_disk', 'thick_disk')

# The values of where a mass is that no component gives (see MassPlace):
# those of the geometries of components by default
_DEFAULTS = dict(xpos=0, ypos=0, posa=0, incl=45)


class MassPlace(Rule):
    """
    The mass of a galaxy that a piece of a lens is the mass of (a piece
    of the type 'mass') has a place on the sky of its own (2026-11-18):
    its centre, mass_xpos and mass_ypos, if a mass component lies at it
    (a point, a halo, a spheroid, or a disk of its own surface density:
    not one that follows a component, which lies on it), and its plane,
    mass_posa and mass_incl, if one lies in it (a disk of its own surface
    density, or a flattened halo or spheroid). In version 16 its centre
    was that of the components that its mass moves, holds up or follows,
    which had to agree, and its plane that of the disks and the spheroids
    of an axis among them. In the params sections, of each such galaxy:

    - mass_xpos and mass_ypos are tied to the xpos and ypos of the first
      component that the mass follows, else of the first that has a
      velocity trait 'mass' (that the mass moves), else of the first
      component; without components, they are 0;
    - mass_posa and mass_incl are tied to the posa and incl of the first
      disk in the same order; without one, they are 0 and 45, to be
      checked by hand.

    The models stay as they were: the components that the mass moves,
    holds up or follows were at its centre, and in its plane.
    """

    @staticmethod
    def version() -> int:
        return 17

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        # (once each: a project sees a galaxy piece and the copies of it in
        # its setups, see migrate_project)
        lensing = {
            piece['galaxy'] for _, lens in items(config, 'lenses')
            for piece in _listed(lens.get('pieces'))
            if isinstance(piece, Mapping) and piece.get('type') == 'mass'}
        places: dict[str, tuple[Any, str]] = {}
        for _, galaxy in items(config, 'galaxies'):
            if galaxy.get('name') in lensing:
                places |= _place(galaxy)
        changes = []
        if places:
            for path, params in items(config, 'params'):
                changes += _added_params(params, path, places)
        return changes


def _place(galaxy: Mapping[str, Any]) -> dict[str, tuple[Any, str]]:
    """
    Return the values of the parameters of where the mass of a galaxy is
    (see MassPlace), by their names in the params sections (prefixed by
    the name of the galaxy, if it has one), each with the message of its
    change; none without mass.
    """
    parts = list(_mass_parts(galaxy))
    if not parts:
        return {}
    prefix = f"{galaxy['name']}_" if 'name' in galaxy else ''
    named = [component for _, component in components(galaxy)]
    prefixes = [
        '' if len(named) == 1 and 'name' not in component
        else f"{component.get('name')}_" for component in named]
    followed = {name for part in parts for name in _followed(part)}
    order = (
        [i for i, c in enumerate(named) if c.get('name') in followed]
        + [i for i, c in enumerate(named) if _moved(c)]
        + list(range(len(named))))
    place: dict[str, tuple[Any, str]] = {}
    names = ('xpos', 'ypos') if any(map(_centred, parts)) else ()
    if order:
        first = order[0]
        place |= {
            f'{prefix}mass_{name}': (
                f'{prefix}{prefixes[first]}{name}',
                f"the centre of the mass: tied to the {name} of the "
                f"component {_label(named, first)!r}")
            for name in names}
    else:
        place |= {
            f'{prefix}mass_{name}': (
                _DEFAULTS[name],
                "the centre of the mass: 0 (the galaxy has no components)")
            for name in names}
    if any(_planar(part) for part in parts):
        disks = [i for i in order if named[i].get('type') in _PLANED]
        if disks:
            place |= {
                f'{prefix}mass_{name}': (
                    f'{prefix}{prefixes[disks[0]]}{name}',
                    f"the plane of the mass: tied to the {name} of the disk "
                    f"{_label(named, disks[0])!r}")
                for name in ('posa', 'incl')}
        else:
            place |= {
                f'{prefix}mass_{name}': (
                    _DEFAULTS[name],
                    f"by hand: set it to the {name} of the plane of the "
                    f"mass (no disk gives it)")
                for name in ('posa', 'incl')}
    return place


def _mass_parts(galaxy: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Yield the mass components of a galaxy that are dictionaries."""
    mass = galaxy.get('mass')
    if isinstance(mass, Mapping):
        mass = mass.get('components', [mass])
    if isinstance(mass, Sequence) and not isinstance(mass, str):
        yield from (part for part in mass if isinstance(part, Mapping))


def _followed(part: Mapping[str, Any]) -> Iterator[str]:
    """
    Yield the names of the components that a mass component follows: of
    a traced disk or spheroid, and of a surface density or a vertical
    structure of the type 'component'.
    """
    if part.get('type') in ('traced_disk', 'traced_spheroid') \
            and isinstance(part.get('component'), str):
        yield part['component']
    for option in ('surface', 'vertical'):
        value = part.get(option)
        if isinstance(value, Mapping) and value.get('type') == 'component' \
                and isinstance(value.get('component'), str):
            yield value['component']


def _moved(component: Mapping[str, Any]) -> bool:
    """
    Check whether a component has a velocity trait 'mass' (the mass moves
    it).
    """
    return any(
        isinstance(trait, Mapping) and trait.get('type') == 'mass'
        for option in ('vptraits', 'vhtraits')
        for trait in _listed(component.get(option)))


def _centred(part: Mapping[str, Any]) -> bool:
    """
    Check whether a mass component lies at the centre of the mass: a
    point, a halo, a spheroid, or a disk of its own surface density.
    """
    kind = part.get('type')
    surface = part.get('surface')
    return kind in ('point', 'halo', 'spheroid') or kind == 'disk' and not (
        isinstance(surface, Mapping) and surface.get('type') == 'component')


def _planar(part: Mapping[str, Any]) -> bool:
    """
    Check whether a mass component lies in the plane of the mass: a disk
    of its own surface density, or a flattened halo or spheroid.
    """
    kind = part.get('type')
    if kind == 'disk':
        surface = part.get('surface')
        return not (isinstance(surface, Mapping)
                    and surface.get('type') == 'component')
    shape = part.get('shape')
    return kind in ('halo', 'spheroid') and isinstance(shape, Mapping) \
        and shape.get('type') == 'flattened'


def _listed(value: Any) -> list[Any]:
    """Return an option of one item or several as a list."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        return list(value)
    return [] if value is None else [value]


def _label(named: Sequence[Mapping[str, Any]], index: int) -> str:
    """Return the name of a component, or its position."""
    return named[index].get('name', f'component{index}')


def _added_params(
        params: MutableMapping[str, Any],
        path: list[str | int],
        place: Mapping[str, tuple[Any, str]]
) -> list[Change]:
    """
    Add the parameters of where the masses are to the properties of a
    params section, in place, where it does not give them (see
    MassPlace).
    """
    properties = params.get('properties')
    if not isinstance(properties, MutableMapping):
        return []
    changes = []
    for name, (value, message) in place.items():
        if name in properties:
            continue
        properties[name] = value
        changes.append(Change(path + ['properties', name], message))
    return changes


RULES: tuple[Rule, ...] = (
    MassPlace(),
)
