"""
Helpers shared by the components.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from gbkfit.utils import iterutils, numutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from .lines import line_parser

if TYPE_CHECKING:
    from .base import Component
    from .geometries import Geometry
    from .lines import Lines


def dump_lines(lines_: 'Lines') -> dict[str, Any]:
    """Return the lines option of a spectral component: none if not given."""
    if lines_.lines() is None:
        return {}
    return dict(lines=line_parser.dump(list(lines_.lines())))


def dump_name(component: 'Component') -> dict[str, Any]:
    """Return the name option of a component: none if it has no name."""
    name = component.name()
    return dict(name=name) if name is not None else {}


def dump_geometry(component: 'Component') -> dict[str, Any]:
    """
    Return the geometry option of a component: the name of its geometry,
    or none if it has its own.
    """
    geometry = component.geometry()
    return dict(geometry=geometry.name()) if geometry is not None else {}


def parse_nodes(
        prefix: str,
        nmin: float | None,
        nmax: float | None,
        nsep: float | None,
        nlen: int | None,
        nodes: Any
) -> tuple[float, ...]:
    """
    Return the nodes given by one of: nodes; nmin, nmax and nsep (from
    nmin every nsep, to the first node at or beyond nmax); nmin, nmax and
    nlen (nlen nodes from nmin to nmax). The prefix names the options in
    messages (e.g. 'r' for rnodes). Raise ConfigError unless exactly one
    of them is given, or the nodes are not at least two, ascending, unique
    and not negative.
    """
    nodes_list = nodes is not None
    nodes_arange = [nmin, nmax, nsep].count(None) == 0
    nodes_linspace = [nmin, nmax, nlen].count(None) == 0
    if [nodes_list, nodes_arange, nodes_linspace].count(True) != 1:
        raise ConfigError(
            f"only one of the following sets of options "
            f"must be defined: "
            f"(1) {prefix}nodes; "
            f"(2) {prefix}nmin, {prefix}nmax, {prefix}nsep; "
            f"(3) {prefix}nmin, {prefix}nmax, {prefix}nlen")
    if (nodes_arange or nodes_linspace) and not (0 <= nmin < nmax):
        raise ConfigError(
            f"the following expression must be true: "
            f"0 <= {prefix}nmin < {prefix}nmax")
    if nodes_arange and not (0 < nsep <= nmax - nmin):
        raise ConfigError(
            f"the following expression must be true: "
            f"0 < {prefix}nsep <= {prefix}nmax - {prefix}nmin")
    if nlen is not None and not 2 <= nlen:
        raise ConfigError(
            f"the following expression must be true: "
            f"2 =< {prefix}nlen")
    if nodes_arange:
        # From nmin every nsep, to the first node at or beyond nmax; the
        # count tolerates the rounding of (nmax - nmin) / nsep, which made
        # np.arange to nmax + nsep add a node (e.g. 2.4 for 2.2 in 0.2s)
        count = int(np.ceil((nmax - nmin) / nsep - 1e-9)) + 1
        nodes = (nmin + nsep * np.arange(count)).tolist()
    elif nodes_linspace:
        nodes = np.linspace(nmin, nmax, nlen).tolist()
    nodes = tuple(nodes)
    if len(nodes) < 2:
        raise ConfigError(f"at least two {prefix}nodes must be provided")
    if not iterutils.is_ascending(nodes):
        raise ConfigError(f"{prefix}nodes must be ascending")
    if not numutils.all_positive(nodes, include_zero=True):
        raise ConfigError(f"{prefix}nodes must not be negative")
    if not iterutils.all_unique(nodes):
        raise ConfigError(f"{prefix}nodes must be unique")
    return nodes


def load_geometries(
        info: dict[str, Any], keys: tuple[str, ...]
) -> None:
    """
    Load the geometries of the configuration of a model (its option
    geometries), in place: the geometry option of each of its components
    (those of the options keys, e.g. 'components') becomes the geometry
    of that name, and the option geometries goes. Raise ConfigError for
    an unknown geometry, repeated names, or a geometry that no component
    uses.
    """
    from .geometries import geometry_parser
    infos = info.pop('geometries', None)
    if infos is None:
        infos = []
    with parseutils.config_path('geometries'):
        geometries = geometry_parser.load(iterutils.listify(infos))
    names = [geometry.name() for geometry in geometries]
    if repeated := sorted({n for n in names if names.count(n) > 1}):
        raise ConfigError(
            f"the geometries must have different names; repeated: "
            f"{repeated}")
    by_name = dict(zip(names, geometries))
    used = set()
    for key in keys:
        value = info.get(key)
        if value is None:
            continue
        is_list = isinstance(value, (list, tuple))
        components = [dict(item) for item in value] if is_list \
            else [dict(value)]
        for i, component in enumerate(components):
            name = component.get('geometry')
            if name is None:
                continue
            if name not in by_name:
                path = (key, i) if is_list else (key,)
                raise ConfigError(
                    f"unknown geometry {name!r}; the geometries are "
                    f"{names}", path + ('geometry',))
            component['geometry'] = by_name[name]
            used.add(name)
        info[key] = components if is_list else components[0]
    if unused := [name for name in names if name not in used]:
        raise ConfigError(f"no component uses the geometries {unused}")


def geometry_params(component: 'Component') -> tuple[str, ...]:
    """Return the geometric parameters of a component, of its pdescs."""
    from .geometries import GEOMETRY_PARAMS
    pdescs = component.pdescs()
    return tuple(name for name in GEOMETRY_PARAMS if name in pdescs)


def shared_params(
        geometry: 'Geometry', components: list['Component']
) -> tuple[str, ...]:
    """
    Return the parameters that a geometry shares: those it was given, or
    all the geometric parameters of the components that use it. Raise
    ConfigError if it was given one that none of them has.
    """
    from .geometries import GEOMETRY_PARAMS
    params = set()
    for component in components:
        params |= set(geometry_params(component))
    if geometry.params() is None:
        return tuple(name for name in GEOMETRY_PARAMS if name in params)
    if unused := [p for p in geometry.params() if p not in params]:
        raise ConfigError(
            f"no component of the geometry {geometry.name()!r} has its "
            f"parameters {unused}")
    return geometry.params()
