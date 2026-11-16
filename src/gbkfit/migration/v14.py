"""
The rules of version 14 of the format (2026-11-16 on), which rewrite
configurations of version 13: the vertical equilibrium of the dispersion
of a thick disk is its option equilibrium, not a dispersion trait.
"""

from collections.abc import MutableMapping
from typing import Any

from .rules import Change, Rule, components, items, like
from .v4 import _insert_after


__all__ = [
    'RULES',
    'EquilibriumOption'
]


class EquilibriumOption(Rule):
    """
    The dispersion trait 'equilibrium' is gone (2026-11-16): the
    dispersion that holds a thick disk up is its option equilibrium
    (gravity and direction, as the trait had; the trait had no
    parameters, so its name goes), after its dptraits, which keep the
    other traits (dropped if none is left).
    """

    @staticmethod
    def version() -> int:
        return 14

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for galaxy_path, galaxy in items(config, 'galaxies'):
            for path, component in components(galaxy):
                dptraits = component.get('dptraits')
                listed = isinstance(dptraits, list)
                traits_ = dptraits if listed else [dptraits]
                found = [
                    trait for trait in traits_
                    if isinstance(trait, MutableMapping)
                    and trait.get('type') == 'equilibrium']
                if not found:
                    continue
                trait = found[0]
                equilibrium = {
                    key: value for key, value in trait.items()
                    if key not in ('type', 'name')}
                rest = [item for item in traits_ if item is not trait]
                _insert_after(
                    component, 'dptraits', 'equilibrium', equilibrium)
                if rest:
                    component['dptraits'] = like(dptraits, rest)
                else:
                    del component['dptraits']
                changes.append(Change(
                    galaxy_path + path + ['dptraits'],
                    "dispersion trait 'equilibrium' -> option equilibrium"))
        return changes


RULES: tuple[Rule, ...] = (
    EquilibriumOption(),
)
