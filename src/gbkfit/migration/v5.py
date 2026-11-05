"""
The rules of version 5 of the format (2026-11-05 on), which rewrite
configurations of version 4.
"""

from collections.abc import Mapping, MutableMapping
from typing import Any

from .rules import Change, Rule, components, items


__all__ = [
    'RULES',
    'DiskThickness'
]


# The options of a thick disk that a thin one does not take, by the
# option of the disk they are in ('': the disk itself); the density height
# traits, which make a disk thick, are not among them
_THICK_OPTIONS = {
    '': ('opacity', 'zptraits', 'clouds'),
    'kinematics': ('vhtraits', 'dhtraits', 'equilibrium')}


class DiskThickness(Rule):
    """
    A disk is thin or thick by its type (2026-11-05): the component type
    'disk' is 'thick_disk' if its density has height traits (htraits), and
    'thin_disk' if not. A thin disk takes none of the options of a thick
    one (its opacity, vertical distortions and clouds, and the height
    traits and the equilibrium of its kinematics; they were errors on a
    disk without height traits): those that are none (null, or an empty
    list) are removed, and the others are left to change by hand.
    """

    @staticmethod
    def version() -> int:
        return 5

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for galaxy_path, galaxy in items(config, 'galaxies'):
            for path, component in components(galaxy):
                if component.get('type') == 'disk':
                    changes += self._disk(component, galaxy_path + path)
        return changes

    @staticmethod
    def _disk(
            component: MutableMapping[str, Any], path: list[str | int]
    ) -> list[Change]:
        """Rewrite a component of the type 'disk', in place."""
        density = component.get('density')
        if isinstance(density, Mapping) \
                and density.get('htraits') is not None:
            component['type'] = 'thick_disk'
            return [Change(path, (
                "type disk -> thick_disk (its density has height traits)"))]
        component['type'] = 'thin_disk'
        changes = [Change(path, "type disk -> thin_disk")]
        if isinstance(density, MutableMapping) and 'htraits' in density:
            del density['htraits']
            changes.append(Change(path + ['density'], "htraits: null removed"))
        for option, names in _THICK_OPTIONS.items():
            part = component.get(option) if option else component
            if not isinstance(part, MutableMapping):
                continue
            part_path = path + ([option] if option else [])
            for name in [n for n in names if n in part]:
                if part[name] is None or part[name] == []:
                    changes.append(Change(
                        part_path, f"{name}: {_none(part.pop(name))} removed"))
                else:
                    changes.append(Change(part_path, (
                        f"by hand: a thin disk takes no {name}: remove it, "
                        f"or give the density height traits (htraits) and "
                        f"make it a thick_disk")))
        return changes


def _none(value: Any) -> str:
    """Return a value that is none (None, or an empty list) as YAML."""
    return 'null' if value is None else '[]'


# The rules, in the order of the changes
RULES: tuple[Rule, ...] = (
    DiskThickness(),
)
