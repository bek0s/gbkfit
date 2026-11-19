"""
Layers of parameters: params configurations (see ParamSpace.load) merged
in order (see merge_layers), e.g. those of the section params of a setup
of a project (see gbkfit.project.resolve).
"""

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from gbkfit.config import ConfigError, config_path


__all__ = [
    'merge_layers'
]


def merge_layers(
        layers: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, int]]:
    """
    Merge layers of parameters (params configurations, see
    ParamSpace.load), in order: e.g. those of the section params of a
    setup (see gbkfit.project.resolve), each of whose layers can be the
    parameters of an item of it (see prefixed_params).

    The properties of a parameter in a layer (by its name, with or
    without subscripts) replace all those of the parameter in the layers
    before it, as do its mode, and a derived value of a name. The priors
    of the layers add up. Any other option is that of the last layer
    that has it (e.g. transforms).

    Parameters
    ----------
    layers : Sequence of Mapping
        The layers.

    Returns
    -------
    dict
        The parameters.
    dict
        The layer of the properties of each parameter (its index), by
        name.

    Raises
    ------
    ConfigError
        If a layer is not a dictionary, or its options 'properties',
        'modes' or 'derived' are not dictionaries, or 'priors' is not a
        list.
    """
    merged: dict[str, Any] = {}
    owners: dict[str, int] = {}
    for i, layer in enumerate(layers):
        with config_path(i):
            if not isinstance(layer, Mapping):
                raise ConfigError("a layer of parameters must be a dictionary")
            for key, value in layer.items():
                with config_path(key):
                    merged[key] = _merge_option(key, merged.get(key), value)
            names = {_parameter(key) for key in layer.get('properties', {})}
            owners |= dict.fromkeys(names, i)
    return copy.deepcopy(merged), owners


def _merge_option(key: str, merged: Any, value: Any) -> Any:
    """Return an option of the merged layers with that of the next one."""
    if key == 'priors':
        if not isinstance(value, list):
            raise ConfigError("must be a list")
        return (merged or []) + value
    if key not in ('properties', 'modes', 'derived'):
        return value
    if not isinstance(value, Mapping):
        raise ConfigError("must be a dictionary")
    # (the properties of a parameter go by its name, without subscripts)
    name_of = _parameter if key == 'properties' else str
    replaced = {name_of(name) for name in value}
    kept = {
        name: v for name, v in (merged or {}).items()
        if name_of(name) not in replaced}
    return kept | dict(value)


def _parameter(key: str) -> str:
    """Return the name of the parameter of a key of the properties."""
    return key.split('[', 1)[0].strip()
