"""
The renders of galaxies, to see a galaxy as it is built: its maps (its
light, velocity and dispersion, of each component and of all) through an
ideal observation (fine pixels, no PSF, LSF or noise), seen as given or
face-on or edge-on, and its curves (the rotation and dispersion of its
disks, and the circular velocity of its mass), and its cube (of its
components that emit lines, through an ideal cube of channels that hold
their light), of its options as a piece of a project gives them (their
files relative to it) and of values of its parameters; and a galaxy as
an observation of its project sees it (its model, on the grid of the
observation, through its instrument; and its data, if it has them).
"""

import dataclasses
import logging
import math
import os
import threading
from collections.abc import Callable, Mapping
from typing import Any, Literal

import astropy.constants
import astropy.units
import numpy as np

import gbkfit.project
import gbkfit.setup
import gbkfit.units
from gbkfit.config import ConfigError, error_message, paths_relative_to
from gbkfit.driver.drivers.host import DriverHost
from gbkfit.model import (
    BandEmission, Extras, Galaxy, GalaxyFunctions, LineEmission,
    galaxy_parser)
from gbkfit.observation import ObservationGroup, observation_parser
from gbkfit.params import ParamSpace

from .errors import Invalid
from .views import Cube
from .workspace import Workspace


__all__ = [
    'GalaxyRender',
    'Observed',
    'ObservedArray',
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

# The most channels of a cube, the light of its lines beyond their mean
# velocities that its channels hold (this many times their dispersion),
# and its channels without any (km/s)
_NCHANNELS = 128
_SIGMAS = 4
_NO_VELOCITIES = (-300.0, 300.0)

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
    that has the map, by its name, or its position if it has none, e.g.
    'component0/light'; NaN where there is no light), and its curves by name
    ('radius', arcsec, and the functions of radii of the galaxy at them,
    e.g. 'rotation_gas' and 'vcirc_halo'; see GalaxyFunctions), with the
    problems of the curves that could not be drawn.
    """
    step: float
    size: int
    maps: dict[str, np.ndarray]
    curves: dict[str, list[float]]
    problems: list[str]


@dataclasses.dataclass
class ObservedArray:
    """
    An array of a galaxy observed: its kind ('model', or 'data' and
    'error'), its item (e.g. 'spectra', 'brightness', 'moment1'), its values (a
    cube, channels by ny by nx, or a map, ny by nx) and their unit ('' if
    none is known).
    """
    kind: str
    item: str
    values: np.ndarray
    unit: str


@dataclasses.dataclass
class Observed:
    """
    A galaxy as an observation sees it (see Renders.observed): the type of
    the observable of the observation, its arrays (the model of each item,
    and the data and errors of the data, if it has them), the velocities of
    its channels (km/s; None without a spectral axis), and the warnings of
    the library as it set them up (e.g. a PSF narrower than the pixels),
    and its pixels on the sky: their size (x, y; arcsec) and the pixel (x,
    y; 0-based) that the model measures positions from.
    """
    observable: str
    arrays: list[ObservedArray]
    velocities: np.ndarray | None
    notes: list[str]
    step: tuple[float, float]
    rpix: tuple[float, float]


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
            size: int = _SIZE,
            progress: Callable[[str], None] | None = None
    ) -> GalaxyRender:
        """
        Render a galaxy of the options of a piece of a project, at the
        values of its parameters of the given properties (numbers, or
        lists of them for vectors; the defaults for the others), telling
        progress the stage it reaches, if given (e.g. 'evaluating').

        Raises
        ------
        Invalid
            If the galaxy or the values are not valid, or it cannot be
            rendered.
        """
        tell = progress or _quiet
        tell('setting up')
        galaxy, radius, step = self._load(project, options, view, size)
        try:
            observations = _observations(galaxy, size, step)
            if not observations:
                raise Invalid("the galaxy has no component that emits light")
            group = ObservationGroup([galaxy], observations)
            values = _values(group, properties)
            plan = group.plan(DriverHost())
            extras = Extras()
            tell('evaluating')
            plan.model_h(_oriented(galaxy, values, view), extras)
        # A mistake of the options or the values, or of the model at them
        except (ConfigError, ValueError, TypeError, RuntimeError) as e:
            raise Invalid(error_message(e)) from e
        maps = _maps(galaxy, extras)
        tell('drawing the curves')
        curves, problems = _curves(galaxy, values, radius)
        return GalaxyRender(step, size, maps, curves, problems)

    def cube(
            self,
            project: str,
            options: Mapping[str, Any],
            properties: Mapping[str, Any],
            view: View = 'sky',
            size: int = _SIZE,
            progress: Callable[[str], None] | None = None
    ) -> Cube:
        """
        Render the cube of a galaxy (as galaxy renders its maps): of its
        components that emit lines, through an ideal cube of the pixels of
        its maps, whose channels of velocities (km/s; of the rest of the
        first line named, if any) hold their light: the velocities of its
        maps, each within _SIGMAS times its dispersion, of each line (at
        most _NCHANNELS channels, of a round width); telling progress the
        stage it reaches, if given.

        Raises
        ------
        Invalid
            If the galaxy or the values are not valid, or it has no
            component that emits lines, or it cannot be rendered.
        """
        tell = progress or _quiet
        tell('setting up')
        galaxy, _, step = self._load(project, options, view, size)
        try:
            seen = [
                c for c in galaxy.components()
                if isinstance(c.emission(), LineEmission)]
            if not seen:
                raise Invalid(
                    "the galaxy has no component that emits lines: a cube "
                    "sees lines")
            observations = _observations(galaxy, size, step)
            group = ObservationGroup([galaxy], observations)
            values = _oriented(galaxy, _values(group, properties), view)
            extras = Extras()
            tell('evaluating the maps')
            group.plan(DriverHost()).model_h(values, extras)
            low, high = _velocity_range(_maps(galaxy, extras))
            offsets = _line_offsets(seen)
            low, high = low + min(offsets), high + max(offsets)
            width = _round_width((high - low) / (_NCHANNELS - 1))
            low = math.floor(low / width) * width
            nchannels = min(_NCHANNELS, math.ceil((high - low) / width) + 1)
            spectral = dict(size=nchannels, step=width, rpix=0, rval=low)
            cube = _observations(galaxy, size, step, spectral)[0]
            group = ObservationGroup([galaxy], [cube])
            tell('evaluating the cube')
            data = group.plan(DriverHost()).model_h(values)[0]
        except (ConfigError, ValueError, TypeError, RuntimeError) as e:
            raise Invalid(error_message(e)) from e
        cube_values = np.array(data['spectra'].data, dtype=np.float32)
        velocities = low + width * np.arange(nchannels)
        return Cube.of(cube_values, velocities, 'km/s')

    def observed(
            self,
            project: str,
            options: Mapping[str, Any],
            properties: Mapping[str, Any],
            observation: str,
            progress: Callable[[str], None] | None = None
    ) -> Observed:
        """
        Evaluate a galaxy of the options of a piece of a project, at the
        values of its parameters of the given properties (the defaults for
        the others, also of the observation), through an observation of
        the project (by name: the pieces it names, its dataset, PSF and
        LSF, in place): its model on the grid of the observation, through
        its instrument and line of sight, and its data, if it has them;
        telling progress the stage it reaches, if given.

        Raises
        ------
        Invalid
            If the galaxy, the observation or the values are not valid, or
            the observation cannot see the galaxy (e.g. a thick disk along
            a flat line of sight).
        """
        tell = progress or _quiet
        tell('setting up')
        root = self._workspace.path(project)
        notes: list[str] = []
        try:
            with _noted(notes):
                resolved = gbkfit.project.resolve(
                    dict(observations=[observation]),
                    os.path.join(root, 'observe.yaml'),
                    gbkfit.project.open(root))
                with paths_relative_to(root):
                    galaxy = galaxy_parser.load(dict(options))
                    seen = observation_parser.load(
                        resolved.config['observations'])
                group = ObservationGroup([galaxy], seen)
                values = _values(group, properties)
                plan = group.plan(DriverHost())
                tell('evaluating')
                outputs = gbkfit.setup.evaluation_outputs(
                    plan, None, values, True, False, False) \
                    | gbkfit.setup.data_outputs(plan)
        except (ConfigError, ValueError, TypeError, RuntimeError) as e:
            raise Invalid(error_message(e)) from e
        observable = seen[0].observable()
        light = seen[0].data_unit()
        arrays = []
        for output in outputs.values():
            data = getattr(output.value, 'data', output.value)
            item = output.ids.get('item')
            if output.kind in ('model', 'data', 'error') and item \
                    and isinstance(data, np.ndarray) and data.ndim in (2, 3):
                arrays.append(ObservedArray(
                    output.kind, item, np.asarray(data, dtype=np.float32),
                    _item_unit(item, light)))
        spectral = observable.spectral()
        velocities = None if spectral is None or spectral.size() is None \
            else spectral.rval() + spectral.step() * (
                np.arange(spectral.size()) - spectral.rpix())
        # (a model with no light at all: say where it may have gone)
        models = [a.values for a in arrays if a.kind == 'model']
        if models and not any(np.nanmax(np.abs(m), initial=0) > 0
                              for m in models):
            where = '' if velocities is None else (
                f", or its velocities out of the channels of the "
                f"observation ({velocities.min():g} to "
                f"{velocities.max():g} km/s; see its systemic velocity)")
            notes.append(
                f"the model has no light in the observation: the galaxy "
                f"may be out of its field, or too small for its pixels"
                f"{where}")
        step, rpix = observable.step(), observable.rpix()
        return Observed(
            observable.type(), arrays, velocities, notes,
            (float(step[0]), float(step[1])), (float(rpix[0]), float(rpix[1])))

    def _load(
            self, project: str, options: Mapping[str, Any], view: View,
            size: int
    ) -> tuple[Galaxy, float, float]:
        """
        Return a galaxy of the options of a piece of a project, its
        largest radius (arcsec) and the size of the pixels of its renders
        of size pixels across (arcsec).

        Raises
        ------
        Invalid
            If the view or the size is not one, or the galaxy is not
            valid.
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
        except (ConfigError, ValueError, TypeError, RuntimeError) as e:
            raise Invalid(error_message(e)) from e
        return galaxy, radius, 2 * _MARGIN * radius / size


def _quiet(stage: str) -> None:
    """Hear the stage of a render, and do nothing with it."""


class _noted:
    """
    Keep the warnings that the library logs in this thread while in it, as
    notes (e.g. of an observation, as it is set up).
    """

    def __init__(self, notes: list[str]):
        self._notes = notes
        self._handler = logging.Handler(logging.WARNING)
        thread = threading.get_ident()
        self._handler.addFilter(lambda record: record.thread == thread)
        self._handler.emit = lambda record: notes.append(record.getMessage())

    def __enter__(self) -> None:
        logging.getLogger('gbkfit').addHandler(self._handler)

    def __exit__(self, *_: object) -> None:
        logging.getLogger('gbkfit').removeHandler(self._handler)


def _item_unit(item: str, light: Any) -> str:
    """
    Return the unit of an item of an observation, of the unit of its
    light (None: unknown): of a cube or an image, that; of a moment map,
    the light times km/s (moment 0), or km/s (moments 1 and 2), or none.
    """
    if item in ('moment1', 'moment2'):
        return 'km/s'
    unit = getattr(light, 'unit', light)
    if unit is None or str(unit) == '':
        return ''
    return f'{unit} km/s' if item == 'moment0' else str(unit)


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


def _observations(
        galaxy: Galaxy, size: int, step: float,
        spectral: dict[str, Any] | None = None
) -> list[Any]:
    """
    Return the ideal observations of a galaxy, of size pixels of the given
    step: a cube of the components that emit lines (of the channels of the
    options of a spectral axis given, or else of two, whose velocities its
    maps hold whatever its channels; at the rest of the first line named,
    if any), and an image of those that emit in
    bands; each with a depth axis if a component it sees is thick or has
    an opacity, and with radiative transfer if one absorbs its own lines.
    """
    components = galaxy.components()
    infos = []
    for kind, observable in (
            (LineEmission, dict(
                type='pixel_spectra',
                spectral=(spectral or dict(size=2, step=100))
                | dict(convention='relativistic'))),
            (BandEmission, dict(type='pixel_brightness'))):
        seen = [c for c in components if isinstance(c.emission(), kind)]
        if not seen:
            continue
        # (the convention then that of the rest)
        if kind is LineEmission and (rest := _rest(seen)) is not None:
            observable = observable | dict(spectral={
                key: value for key, value in observable['spectral'].items()
                if key != 'convention'} | dict(rest=rest))
        opacities = [c.opacity() for c in seen if c.opacity() is not None]
        if any(o.type() == 'line' for o in opacities):
            line_of_sight = 'transfer'
        elif opacities or any(c.thick(kind) for c in seen):
            line_of_sight = 'grid'
        else:
            line_of_sight = 'flat'
        infos.append(dict(
            type='sky',
            observable=observable | dict(size=[size, size], step=[step, step]),
            line_of_sight=dict(type=line_of_sight)))
    return observation_parser.load(infos)


def _rest(components: list[Any]) -> dict[str, Any] | None:
    """
    Return the rest of the first line named by components that emit lines,
    as an option of a spectral axis; None if they name none.
    """
    for component in components:
        if lines := component.emission().lines().lines():
            rest = lines[0].rest()
            return dict(
                value=float(rest.value),
                unit=gbkfit.units.format_unit(rest.unit))
    return None


def _velocity_range(maps: Mapping[str, np.ndarray]) -> tuple[float, float]:
    """
    Return the velocities (km/s) that hold the light of the maps of a
    galaxy (see _maps): those of each component, each within _SIGMAS times
    its dispersion; _NO_VELOCITIES if it has none.
    """
    lows, highs = [], []
    for name, velocity in maps.items():
        if not name.endswith('velocity'):
            continue
        dispersion = maps.get(name.replace('velocity', 'dispersion'))
        spread = _SIGMAS * np.nan_to_num(dispersion) \
            if dispersion is not None else 0
        finite = np.isfinite(velocity)
        if finite.any():
            lows.append(float(np.min((velocity - spread)[finite])))
            highs.append(float(np.max((velocity + spread)[finite])))
    if not lows or max(highs) <= min(lows):
        return _NO_VELOCITIES
    return min(lows), max(highs)


def _line_offsets(components: list[Any]) -> list[float]:
    """
    Return the velocities (km/s) of the lines that components emit
    relative to the first line named (of its rest: optical velocities of a
    wavelength, radio ones of a frequency): [0] if they name none.
    """
    rests = [
        line.rest() for component in components
        for line in component.emission().lines().lines() or ()]
    if not rests:
        return [0.0]
    first = rests[0]
    c = astropy.constants.c.to_value('km/s')
    offsets = []
    for rest in rests:
        ratio = rest.to_value(first.unit, astropy.units.spectral()) \
            / first.value
        offsets.append(
            c * (ratio - 1) if first.unit.physical_type == 'length'
            else c * (1 - ratio))
    return offsets


def _round_width(width: float) -> float:
    """Return a round width at least the one given: 1, 2 or 5 times 10^n."""
    power = 10 ** math.floor(math.log10(width))
    for lead in (1, 2, 5, 10):
        if lead * power >= width * (1 - 1e-9):
            return lead * power
    return 10 * power


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
        names = galaxy.component_param_names(index)
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
    for i, component in enumerate(galaxy.components()):
        # (the label of the component in the extras: its name, or its
        # position, e.g. 'component0')
        name = component.name() or f'component{i}'
        own = {}
        for key, extra in _MAPS.items():
            found = [x for x in extras if x.name == extra
                     and x.ids.get('component') == name]
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
