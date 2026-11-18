"""
The rules of version 16 of the format (2026-11-18 on), which rewrite
configurations of version 15: the parameter alpha of Palatini f(R)
gravity has the sign of f(R) = R + alpha R^2.
"""

import re
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from .rules import EXPRESSIONS, Change, Rule, items, rename, replaced


__all__ = [
    'RULES',
    'PalatiniAlpha'
]


# The priors that a negated parameter negates: the mean, and the range
# turned around
_NEGATED_PRIORS = ('uniform', 'gauss', 'gauss_trunc')


class PalatiniAlpha(Rule):
    """
    The parameter gravity_alpha of Palatini f(R) gravity is the alpha of
    f(R) = R + alpha R^2 (2026-11-18): its correction had the opposite
    sign (Toniato et al. 2020, eq. 32), so the alpha of version 15 is
    -alpha, and the models stay as they were. In the params sections, of
    each galaxy of Palatini gravity:

    - a number is negated, an expression too (-(...)), and a free alpha
      has its value negated and its bounds turned around (min: -max), as
      its prior (uniform, gauss or gauss_trunc: the mean negated, the
      range turned around); any other prior, a solved alpha and one of
      the transforms function are negated by hand;
    - the expressions that read it read -alpha.
    """

    @staticmethod
    def version() -> int:
        return 16

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        # (once each: a project sees a galaxy piece and the copies of it in
        # its setups, see migrate_project)
        names = list(dict.fromkeys(
            (f"{galaxy['name']}_" if 'name' in galaxy else '')
            + 'gravity_alpha'
            for _, galaxy in items(config, 'galaxies')
            if isinstance(gravity := galaxy.get('gravity'), Mapping)
            and gravity.get('type') == 'palatini'))
        changes = []
        if names:
            for path, params in items(config, 'params'):
                changes += _negated_params(params, path, names)
        return changes


def _negated_params(
        params: MutableMapping[str, Any],
        path: list[str | int],
        names: Sequence[str]
) -> list[Change]:
    """
    Negate parameters of a params section, of the given names, in place
    (see PalatiniAlpha).
    """
    changes = []
    pattern = re.compile(
        rf'(?<![\w.])({"|".join(re.escape(n) for n in names)})(?!\w)')
    for key in [k for k in EXPRESSIONS if k in params]:
        params[key], found = replaced(
            params[key], pattern, lambda m: f'(-{m.group(1)})')
        if found:
            changes.append(Change(path + [key], (
                "the expressions read -gravity_alpha (Palatini f(R): its "
                "alpha has the opposite sign)")))
    properties = params.get('properties')
    if not isinstance(properties, MutableMapping):
        return changes
    for name in [n for n in names if n in properties]:
        properties[name], by_hand = _negated(properties[name])
        changes.append(Change(path + ['properties', name], (
            f"by hand: negate {by_hand} (Palatini f(R): alpha has the "
            f"opposite sign)" if by_hand else
            "negated (Palatini f(R): alpha has the opposite sign)")))
    if 'transforms' in params:
        changes.append(Change(path + ['transforms'], (
            f"by hand: negate {', '.join(names)} in the transforms "
            f"function, if it gives them (Palatini f(R): alpha has the "
            f"opposite sign)")))
    return changes


def _negated(value: Any) -> tuple[Any, str | None]:
    """
    Return the property of a parameter negated (see PalatiniAlpha), and
    what is to be negated by hand, if anything.
    """
    if value is None:
        # (of the transforms function, see _negated_params)
        return value, None
    if isinstance(value, (int, float)):
        return -value, None
    if isinstance(value, str):
        return f'-({value})', None
    if not isinstance(value, MutableMapping) or 'solve' in value:
        return value, "it"
    if 'value' in value:
        value['value'] = -value['value']
    _turned(value)
    prior = value.get('prior')
    if prior is None:
        return value, None
    if not isinstance(prior, MutableMapping) \
            or prior.get('type') not in _NEGATED_PRIORS:
        return value, "its prior"
    if 'mean' in prior:
        prior['mean'] = -prior['mean']
    _turned(prior)
    return value, None


def _turned(options: MutableMapping[str, Any]) -> None:
    """
    Turn the range of options around, in place, where it is: min: -max,
    max: -min (either alone too).
    """
    if 'min' in options and 'max' in options:
        options['min'], options['max'] = -options['max'], -options['min']
    elif 'min' in options:
        options['min'] = -options['min']
        rename(options, 'min', 'max')
    elif 'max' in options:
        options['max'] = -options['max']
        rename(options, 'max', 'min')


RULES: tuple[Rule, ...] = (
    PalatiniAlpha(),
)
