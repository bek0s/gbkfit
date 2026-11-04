"""
The renders of galaxies, to see a galaxy as it is built: its maps (its
light, velocity and dispersion, of each component and of all) through an
ideal observation (fine pixels, no PSF, LSF or noise), seen as given or
face-on or edge-on, and its curves (the rotation and dispersion of its
disks, and the circular velocity of its mass), of its options as a piece
of a project gives them (their files relative to it) and of values of its
parameters.
"""

import dataclasses
import math
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from gbkfit.config import ConfigError, error_message, paths_relative_to
from gbkfit.driver.drivers.host import DriverHost
from gbkfit.model import (
    BandEmission, Extras, Galaxy, LineEmission, galaxy_parser)
from gbkfit.model.components import GalaxyFunctions
from gbkfit.observation import ObservationGroup, observation_parser
from gbkfit.params import ParamSpace

from .errors import Invalid
from .workspace import Workspace


__all__ = [
    'GalaxyRender',
    'Renders',
    'View'
]


# How a galaxy is seen: at the orientation of its parameters, or with every
# component turned face-on or edge-on (its posa and incl replaced; the
# major axis of an edge-on one along x)
View = Literal['sky', 'face-on', 'edge-on']
_ORIENTATIONS = {
    'face-on': dict(posa=0, incl=0), 'edge-on': dict(posa=90, incl=90)}

# The pixels across a render, and the most
_SIZE = 160
_MOST = 512

# The field of a render: this times the largest radius of the galaxy (the
# last node of its disks and cones, or the bounding radius of the others),
# or _FIELD arcsec without one
_MARGIN = 1.15
_FIELD = 20.0

# The radii a curve is drawn at, to the largest radius, from just off the
# centre (where some are not defined, e.g. the slope of a density)
_NCURVE = 96

# The maps of each component, by the names of the extra outputs of its
# plan (see Extras): its light, its velocity on the line of sight and in
# the plane of the sky (x, y), and its dispersion
_MAPS = dict(
    light='bdata', velocity='vdata', vx='vxdata', vy='vydata',
    dispersion='ddata')


@dataclasses.dataclass
class GalaxyRender:
    """
    A galaxy rendered: the size of its pixels (arcsec) and their number
    across, its maps by name (ny, nx; 'light', 'velocity', 'vx', 'vy' and
    'dispersion' of all its components, and '{component}/{map}' of each
    that has the map; NaN where there is no light), and its curves by name
    ('radius', arcsec, and the functions of radii of the galaxy at them,
    e.g. 'rotation_gas' and 'vcirc_halo'; see GalaxyFunctions), with the
    problems of the curves that could not be drawn.
    """
    step: float
    size: int
    maps: dict[str, np.ndarray]
    curves: dict[str, list[float]]
    problems: list[str]


class Renders:
    """
    The renders of the galaxies of the projects of a workspace.

    Parameters
    ----------
    workspace : Workspace
        The workspace.
    """

    def __init__(self, workspace: Workspace):
        self._workspace = workspace

    def galaxy(
            self,
            project: str,
            options: Mapping[str, Any],
            properties: Mapping[str, Any],
            view: View = 'sky',
            size: int = _SIZE
    ) -> GalaxyRender:
        """
        Render a galaxy of the options of a piece of a project, at the
        values of its parameters of the given properties (numbers, or
        lists of them for vectors; the defaults for the others).

        Raises
        ------
        Invalid
            If the galaxy or the values are not valid, or it cannot be
            rendered.
        """
        if view not in ('sky', *_ORIENTATIONS):
            raise Invalid(f"unknown view {view!r}")
        if not 16 <= size <= _MOST:
            raise Invalid(f"the size must be from 16 to {_MOST}; it is {size}")
        root = self._workspace.path(project)
        try:
            with paths_relative_to(root):
                galaxy = galaxy_parser.load(dict(options))
            radius = _radius(galaxy)
            step = 2 * _MARGIN * radius / size
            observations = _observations(galaxy, size, step)
            if not observations:
                raise Invalid("the galaxy has no component that emits light")
            group = ObservationGroup([galaxy], observations)
            values = _values(group, properties)
            plan = group.plan(DriverHost())
            extras = Extras()
            plan.model_h(_oriented(galaxy, values, view), extras)
        # A mistake of the options or the values, or of the model at them
        except (ConfigError, ValueError, TypeError, RuntimeError) as e:
            raise Invalid(error_message(e)) from e
        maps = _maps(galaxy, extras)
        curves, problems = _curves(galaxy, values, radius)
        return GalaxyRender(step, size, maps, curves, problems)


def _radius(galaxy: Galaxy) -> float:
    """
    Return the largest radius of a galaxy (arcsec): of the last node of its
    disks and cones, or the bounding radius of its other components.
    """
    radii = [
        float(nodes[-1]) for name, nodes in galaxy.constants().items()
        if name == 'rnodes' or name.endswith('_rnodes')]
    radii += [cmp.bounding_radius() for cmp in galaxy.components()]
    radius = max(radii, default=0.0)
    return radius if radius > 0 else _FIELD


def _observations(galaxy: Galaxy, size: int, step: float) -> list[Any]:
    """
    Return the ideal observations of a galaxy, of size pixels of the given
    step: a cube of the components that emit lines (one line of no rest,
    whose velocities its maps hold whatever its channels), and an image
    of those that emit in bands; each with a depth axis if a component it
    sees is thick.
    """
    components = galaxy.components()
    infos = []
    for kind, observable in (
            (LineEmission, dict(
                type='pixel_spectra',
                spectral=dict(size=2, step=100, convention='relativistic'))),
            (BandEmission, dict(type='pixel_brightness'))):
        seen = [c for c in components if isinstance(c.emission(), kind)]
        if not seen:
            continue
        thick = any(c.thick(kind) for c in seen)
        infos.append(dict(
            type='sky',
            observable=observable | dict(size=[size, size], step=[step, step]),
            line_of_sight=dict(type='grid' if thick else 'flat')))
    return observation_parser.load(infos)


def _values(
        group: ObservationGroup, properties: Mapping[str, Any]
) -> dict[str, Any]:
    """
    Return the values of the parameters of a group: those of the given
    properties (numbers or lists of them), and the defaults of the others.

    Raises
    ------
    ConfigError
        If a property is not a value of a parameter of the group.
    """
    pdescs = group.pdescs()
    given = {name: value for name, value in properties.items()}
    if unknown := sorted(set(given) - set(pdescs)):
        raise ConfigError(f"unknown parameters: {unknown}")
    filled = {
        name: given.get(name, _default(desc))
        for name, desc in pdescs.items()}
    space = ParamSpace.from_properties(pdescs, filled)
    return dict(space.evaluate())


def _default(desc: Any) -> Any:
    """Return the default value of a parameter, a list for a vector."""
    default = desc.default()
    return [default] * desc.size() if desc.size() > 1 else default


def _oriented(galaxy: Galaxy, values: dict[str, Any], view: View) -> dict:
    """
    Return the values of a galaxy seen in a view: with the posa and incl
    of each component replaced, face-on or edge-on (each element of a
    node-wise one).
    """
    if view == 'sky':
        return values
    values = dict(values)
    for index in range(len(galaxy.components())):
        names = galaxy.component_params(index)
        for param, value in _ORIENTATIONS[view].items():
            if param in names:
                name = names[param]
                values[name] = np.full_like(
                    np.asarray(values[name], dtype=float), value)
    return values


def _maps(galaxy: Galaxy, extras: Extras) -> dict[str, np.ndarray]:
    """
    Return the maps of the components of a galaxy from the extra outputs
    of its ideal observations, and those of all of them: their total light,
    and their velocities and dispersion weighted by the light of those
    that have them; NaN where there is no light.
    """
    maps: dict[str, np.ndarray] = {}
    sums: dict[str, np.ndarray] = {}
    weights = None
    total = None
    for component in galaxy.components():
        name = component.name() or 'component'
        own = {}
        for key, extra in _MAPS.items():
            found = [x for x in extras if x.name == extra
                     and x.ids.get('component', 'component') == name]
            if found:
                own[key] = np.asarray(found[0].value.data, dtype=float)
        if 'light' not in own:
            continue
        light = own['light']
        if light.ndim == 3:
            own = {key: _projected(value, light) if key != 'light'
                   else value.sum(axis=0) for key, value in own.items()}
            light = own['light']
        total = light if total is None else total + light
        dark = ~(np.abs(light) > 0)
        for key, value in own.items():
            maps[f'{name}/{key}'] = value if key == 'light' \
                else np.where(dark, np.nan, value)
        if 'velocity' in own:
            weight = np.abs(light)
            weights = weight if weights is None else weights + weight
            for key in ('velocity', 'vx', 'vy', 'dispersion'):
                if key in own:
                    value = np.nan_to_num(own[key]) * weight
                    sums[key] = value if key not in sums else sums[key] + value
    if total is not None:
        maps['light'] = total
    if weights is not None:
        dark = ~(weights > 0)
        for key, value in sums.items():
            maps[key] = np.where(
                dark, np.nan, value / np.where(dark, 1, weights))
    return maps


def _projected(value: np.ndarray, light: np.ndarray) -> np.ndarray:
    """
    Return a map of a 3d grid (z, y, x) projected along the line of sight,
    weighted by the light of each voxel.
    """
    weight = np.abs(light)
    total = weight.sum(axis=0)
    return np.where(
        total > 0, (np.nan_to_num(value) * weight).sum(axis=0)
        / np.where(total > 0, total, 1), np.nan)


def _curves(
        galaxy: Galaxy, values: Mapping[str, Any], radius: float
) -> tuple[dict[str, list[float]], list[str]]:
    """
    Return the curves of a galaxy from just off its centre to its largest
    radius (arcsec): 'radius', and each of its functions of radii at them
    (e.g. the rotation of a disk; those of its mass, of radii in kpc, at
    the same radii on the sky), with the problems of those that failed.
    """
    radii = np.linspace(radius / _NCURVE, radius, _NCURVE)
    curves = dict(radius=radii.tolist())
    problems = []
    try:
        bound = galaxy.plan_functions().bind(values)
    except (ValueError, TypeError, RuntimeError) as e:
        return curves, [error_message(e)]
    kpc = bound.get('kpc_per_arcsec')
    for name, function in bound.items():
        units = GalaxyFunctions.radial_units(name)
        if units is None:
            continue
        in_kpc = units[1].is_equivalent('kpc')
        if in_kpc and kpc is None:
            continue
        at = radii * kpc if in_kpc else radii
        try:
            result = np.asarray(function(at), dtype=float)
        except (ValueError, TypeError, RuntimeError) as e:
            problems.append(f"{name}: {error_message(e)}")
            continue
        curves[name] = [
            float(v) if math.isfinite(v) else None for v in result]
    return curves, problems
