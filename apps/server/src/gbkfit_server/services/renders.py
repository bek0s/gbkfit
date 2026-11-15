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
from gbkfit.grid import doppler
from gbkfit.model import (
    BandEmission, Extras, Galaxy, GalaxyFunctions, LineEmission,
    galaxy_parser)
from gbkfit.observation import ObservationGroup, observation_parser
from gbkfit.params import ParamSpace

from .errors import Invalid
from .views import Cube, CubeAxis
from .workspace import Workspace


__all__ = [
    'GalaxyExtra',
    'GalaxyRender',
    'GalaxyVolume',
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

# The directions of the speeds of the disks (see GalaxyFunctions; the
# planar traits, along none, are left out) and of their dispersions
_SPEED_DIRECTIONS = ('tan', 'rad', 'ver', 'away', 'out', 'los')
_DISPERSION_DIRECTIONS = ('rad', 'tan', 'ver')

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

# The reach of the curves of a galaxy of mass alone, which has no
# components to give it a largest radius (kpc): about that of the rotation
# curve of a galaxy
_MASS_REACH = 30.0

# The maps of each component, by the names of the extra outputs of its
# plan (see Extras): its light, its velocity on the line of sight and in
# the plane of the sky (x, y), and its dispersion
_MAPS = dict(
    light='bdata', velocity='vdata', vx='vxdata', vy='vydata',
    dispersion='ddata')


class _NoLight(Invalid):
    """A galaxy that emits no light (e.g. of mass alone): nothing to map."""


@dataclasses.dataclass
class GalaxyRender:
    """
    A galaxy rendered: the size of its pixels (arcsec) and their number
    across, its maps by name (ny, nx; 'light', 'velocity', 'vx', 'vy' and
    'dispersion' of all its components, and '{component}/{map}' of each
    that has the map, by its name, or its position if it has none, e.g.
    'component0/light'; NaN where there is no light; none, of a galaxy
    that emits no light), and its curves by name
    ('radius', arcsec, and the functions of radii of the galaxy at them,
    e.g. 'rotation_gas' and 'vcirc_halo'; see GalaxyFunctions; and of each
    disk its speeds along each direction it has traits along,
    'speed_gas:rad', and its dispersion along each direction if it is not
    isotropic, 'dispersion_gas:tan'), with the problems of the curves that
    could not be drawn; and the extra outputs of the evaluation (see
    gbkfit.model.Extras: of each, its name, ids, title, description, unit
    and number of dimensions; see Renders.extra).
    """
    step: float
    size: int
    maps: dict[str, np.ndarray]
    curves: dict[str, list[float]]
    problems: list[str]
    extras: list[dict[str, Any]]


@dataclasses.dataclass
class GalaxyVolume:
    """
    The light of a galaxy in 3d, face-on (see Renders.volume): its values
    (nz, ny, nx; the axis z along its axis of rotation, from below its
    plane to above it), the size of its voxels across (step) and along z
    (depth_step; arcsec), and the voxel of its plane along z (zero, 0-based;
    fractional).
    """
    step: float
    depth_step: float
    zero: float
    values: np.ndarray


@dataclasses.dataclass
class GalaxyExtra:
    """
    An extra output of the evaluation of a galaxy (see Renders.extra): its
    name, ids, title, description and unit (as the library gives them; ''
    if none), its values (an array: e.g. a map, ny by nx, or a cube with a
    depth axis, nz by ny by nx), and the size of its pixels across (step;
    arcsec) and along its depth (depth_step; None without a depth axis).
    """
    name: str
    ids: dict[str, str]
    title: str
    description: str
    unit: str
    values: np.ndarray
    step: float
    depth_step: float | None


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
    the observation (e.g. 'pixel_spectra'), its arrays (the model of each item,
    and the data and errors of the data, if it has them), the velocities of
    its channels (km/s; None without a spectral axis), and the warnings of
    the library as it set them up (e.g. a PSF narrower than the pixels),
    and its pixels on the sky: their size (x, y; arcsec) and the pixel (x,
    y; 0-based) that the model measures positions from.
    """
    type: str
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
        progress the stage it reaches, if given (e.g. 'evaluating'). A
        galaxy that emits no light (e.g. of mass alone) has its curves
        but no maps (see _unlit).

        Raises
        ------
        Invalid
            If the galaxy or the values are not valid, or it cannot be
            rendered.
        """
        tell = progress or _quiet
        try:
            galaxy, values, extras, radius, step = self._evaluate(
                project, options, properties, view, size, tell)
        except _NoLight:
            return self._unlit(project, options, properties, view, size, tell)
        maps = _maps(galaxy, extras)
        tell('drawing the curves')
        curves, problems = _curves(galaxy, values, radius)
        listed = [
            dict(name=x.name, ids=x.ids, title=x.title,
                 description=x.description, unit=_extra_unit(x.value),
                 ndim=int(np.ndim(_extra_values(x.value))))
            for x in extras if _extra_values(x.value) is not None]
        return GalaxyRender(step, size, maps, curves, problems, listed)

    def _unlit(
            self,
            project: str,
            options: Mapping[str, Any],
            properties: Mapping[str, Any],
            view: View,
            size: int,
            tell: Callable[[str], None]
    ) -> GalaxyRender:
        """
        Render a galaxy that emits no light (e.g. of mass alone, as the
        rotation curve of a galaxy is fitted by): its curves, to its
        largest radius (of the components it has, which only dim; else
        _MASS_REACH kpc), and no maps.

        Raises
        ------
        Invalid
            If the galaxy or the values are not valid.
        """
        galaxy, radius, step = self._load(project, options, view, size)
        try:
            values = _values(galaxy, properties)
        except (ConfigError, ValueError, TypeError, RuntimeError) as e:
            raise Invalid(error_message(e)) from e
        tell('drawing the curves')
        curves, problems = _curves(
            galaxy, values, radius if galaxy.components() else None)
        return GalaxyRender(step, size, {}, curves, problems, [])

    def volume(
            self,
            project: str,
            options: Mapping[str, Any],
            properties: Mapping[str, Any],
            size: int = _SIZE,
            progress: Callable[[str], None] | None = None
    ) -> GalaxyVolume:
        """
        Render the light of a galaxy in 3d (as galaxy renders its maps),
        face-on, so that z is along its axis of rotation: the brightness of
        each of its components in each voxel of an ideal observation with
        a depth axis (those in a plane, e.g. thin disks, in the voxels of
        its plane); telling progress the stage it reaches, if given.

        Raises
        ------
        Invalid
            If the galaxy or the values are not valid, or it cannot be
            rendered.
        """
        tell = progress or _quiet
        galaxy, _, extras, _, step = self._evaluate(
            project, options, properties, 'face-on', size, tell)
        tell('gathering the volume')
        cubes, planes = [], []
        for extra in extras:
            if extra.name != 'bdata' or 'component' not in extra.ids:
                continue
            data = np.asarray(extra.value.data, dtype=np.float32)
            (cubes if data.ndim == 3 else planes).append((extra.value, data))
        if cubes:
            coords = cubes[0][0].coords
            depth_step, zero = float(coords.step[2]), float(coords.rpix[2])
            values = sum(data for _, data in cubes)
        else:
            depth_step, zero = step, 0.0
            values = np.zeros((1, size, size), dtype=np.float32)
        for _, data in planes:
            values[int(round(min(max(zero, 0), len(values) - 1)))] += data
        return GalaxyVolume(step, depth_step, zero, values.astype(np.float32))

    def extra(
            self,
            project: str,
            options: Mapping[str, Any],
            properties: Mapping[str, Any],
            name: str,
            ids: Mapping[str, str],
            view: View = 'sky',
            size: int = _SIZE,
            progress: Callable[[str], None] | None = None
    ) -> GalaxyExtra:
        """
        Render an extra output of the evaluation of a galaxy (as galaxy
        renders its maps; see gbkfit.model.Extras): that of the name and
        the ids given (e.g. 'vdata' of the component 'gas'), telling
        progress the stage it reaches, if given.

        Raises
        ------
        Invalid
            If the galaxy or the values are not valid, it cannot be
            rendered, or its evaluation has no such output.
        """
        tell = progress or _quiet
        _, _, extras, _, step = self._evaluate(
            project, options, properties, view, size, tell)
        found = [
            x for x in extras if x.name == name and x.ids == dict(ids)
            and _extra_values(x.value) is not None]
        if not found:
            raise Invalid(f"the galaxy has no extra output {name} of {ids}")
        extra = found[0]
        coords = getattr(extra.value, 'coords', None)
        depth = None if coords is None or len(coords.step) < 3 \
            else float(coords.step[2])
        return GalaxyExtra(
            extra.name, extra.ids, extra.title, extra.description,
            _extra_unit(extra.value),
            np.asarray(_extra_values(extra.value), dtype=np.float32),
            step, depth)

    def _evaluate(
            self,
            project: str,
            options: Mapping[str, Any],
            properties: Mapping[str, Any],
            view: View,
            size: int,
            tell: Callable[[str], None]
    ) -> tuple[Galaxy, dict[str, Any], Extras, float, float]:
        """
        Evaluate a galaxy of the options of a piece of a project through
        its ideal observations (see _observations), seen as given: return
        the galaxy, the values of its parameters, the extra outputs of the
        evaluation, its largest radius and the size of its pixels (arcsec).

        Raises
        ------
        Invalid
            If the galaxy or the values are not valid, or it cannot be
            evaluated (e.g. it has no component that emits light).
        """
        tell('setting up')
        galaxy, radius, step = self._load(project, options, view, size)
        try:
            observations = _observations(galaxy, size, step)
            if not observations:
                raise _NoLight("the galaxy has no component that emits light")
            group = ObservationGroup([galaxy], observations)
            values = _values(group, properties)
            plan = group.plan(DriverHost())
            extras = Extras()
            tell('evaluating')
            plan.model_h(_oriented(galaxy, values, view), extras)
        # A mistake of the options or the values, or of the model at them
        except (ConfigError, ValueError, TypeError, RuntimeError) as e:
            raise Invalid(error_message(e)) from e
        return galaxy, values, extras, radius, step

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
        maps, each within _SIGMAS times its dispersion, of each line, at
        its redshift (see _cube_axis; at most _NCHANNELS
        channels, of a round width); telling progress the stage it
        reaches, if given.

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
            axis = _cube_axis(galaxy.redshift(), seen)
            low, high = _observed_velocities(
                _velocity_range(_maps(galaxy, extras)), axis)
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
        return Cube.of(cube_values, velocities, 'km/s', step, axis)

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
        observation = seen[0]
        light = observation.data_unit()
        arrays = []
        for output in outputs.values():
            data = getattr(output.value, 'data', output.value)
            item = output.ids.get('item')
            if output.kind in ('model', 'data', 'error') and item \
                    and isinstance(data, np.ndarray) and data.ndim in (2, 3):
                arrays.append(ObservedArray(
                    output.kind, item, np.asarray(data, dtype=np.float32),
                    _item_unit(item, light)))
        spectral = observation.spectral()
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
        step, rpix = observation.step(), observation.rpix()
        return Observed(
            observation.type(), arrays, velocities, notes,
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
    bands; each evaluated along the line of sight as what it sees needs
    (see Scene.evaluation of gbkfit.model).
    """
    components = galaxy.components()
    infos = []
    for kind, info in (
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
            info = info | dict(spectral={
                key: value for key, value in info['spectral'].items()
                if key != 'convention'} | dict(rest=rest))
        infos.append(info | dict(size=[size, size], step=[step, step]))
    return observation_parser.load(infos)


def _rest(components: list[Any]) -> str | None:
    """
    Return the rest of the first line named by components that emit lines,
    as an option of a spectral axis (its text, in full); None if they name
    none.
    """
    for component in components:
        if lines := component.emission().lines().lines():
            rest = lines[0].rest()
            unit = gbkfit.units.format_unit(rest.unit)
            return f"{float(rest.value)!r} {unit}"
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


def _cube_axis(redshift: float, components: list[Any]) -> CubeAxis:
    """
    Return how the velocities of a galaxy land on the spectral axis of its
    cube (see _observations), as its lines are placed (see LinePlacement
    and Lines.factors): at the 1 + z of its redshift, in the convention of
    the rest of the first line named by the components that emit lines
    (optical for a wavelength, radio for a frequency); without names, one
    line at the rest of the axis whatever the redshift, on an axis of
    relativistic velocities.
    """
    rests = [
        lines[0].rest() for component in components
        if (lines := component.emission().lines().lines())]
    if not rests:
        return CubeAxis('relativistic', 1.0)
    return CubeAxis(
        'optical' if rests[0].unit.physical_type == 'length' else 'radio',
        1 + redshift)


def _observed_velocities(
        velocities: tuple[float, float], axis: CubeAxis
) -> tuple[float, float]:
    """
    Return the velocities (km/s) of the spectral axis of the cube of a
    galaxy where the light of the given velocities of its maps lands (see
    CubeAxis): of the Doppler factor (1 + z) D(v), in the convention of
    the axis.
    """
    c = astropy.constants.c.to_value('km/s')
    factor = axis.shift * doppler(np.asarray(velocities, dtype=float))
    if axis.convention == 'relativistic':
        observed = c * (factor ** 2 - 1) / (factor ** 2 + 1)
    elif axis.convention == 'optical':
        observed = c * (factor - 1)
    else:
        observed = c * (1 - 1 / factor)
    return float(observed[0]), float(observed[1])


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
        group: ObservationGroup | Galaxy, properties: Mapping[str, Any]
) -> dict[str, Any]:
    """
    Return the values of the parameters of a group (or of a galaxy alone):
    those of the given properties (numbers or lists of them), and the
    defaults of the others.

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
        galaxy: Galaxy, values: Mapping[str, Any], radius: float | None
) -> tuple[dict[str, list[float]], list[str]]:
    """
    Return the curves of a galaxy from just off its centre to its largest
    radius (arcsec; None: _MASS_REACH kpc, of a galaxy of mass alone, or
    _FIELD arcsec without a distance): 'radius', and each of its
    functions of radii at them
    (e.g. the rotation of a disk; those of its mass, of radii in kpc, at
    the same radii on the sky), with the problems of those that failed: of
    a component, its speeds along each direction it has traits along (e.g.
    'speed_gas:rad'), and its dispersion along each direction if it is not
    isotropic (e.g. 'dispersion_gas:tan'), and its position angle and
    inclination ('orientation_gas:posa', 'orientation_gas:incl').
    """
    problems = []
    try:
        bound = galaxy.plan_functions().bind(values)
    except (ValueError, TypeError, RuntimeError) as e:
        bound, problems = {}, [error_message(e)]
    kpc = bound.get('kpc_per_arcsec')
    if radius is None:
        radius = _MASS_REACH / kpc if kpc else _FIELD
    radii = np.linspace(radius / _NCURVE, radius, _NCURVE)
    curves = dict(radius=radii.tolist())
    for name, function in bound.items():
        units = GalaxyFunctions.radial_units(name)
        if units is None:
            continue
        in_kpc = units[1].is_equivalent('kpc')
        if in_kpc and kpc is None:
            continue
        at = radii * kpc if in_kpc else radii
        family = GalaxyFunctions.family(name)
        if family == 'orientation':
            for which in ('posa', 'incl'):
                curves[f'{name}:{which}'] = _listed(
                    np.asarray(function(at, which), dtype=float))
            continue
        if family == 'speed':
            for direction in _SPEED_DIRECTIONS:
                try:
                    result = np.asarray(function(at, direction), dtype=float)
                except (ValueError, TypeError, RuntimeError) as e:
                    problems.append(f"{name}: {error_message(e)}")
                    break
                # (of the directions it moves along, where it is defined)
                if np.any((result != 0) & np.isfinite(result)):
                    curves[f'{name}:{direction}'] = _listed(result)
            continue
        try:
            result = np.asarray(function(at), dtype=float)
        except ValueError as e:
            # (the dispersion of a disk that is not isotropic, along each
            # direction)
            if family != 'dispersion':
                problems.append(f"{name}: {error_message(e)}")
                continue
            for direction in _DISPERSION_DIRECTIONS:
                curves[f'{name}:{direction}'] = _listed(
                    np.asarray(function(at, direction), dtype=float))
            continue
        except (TypeError, RuntimeError) as e:
            problems.append(f"{name}: {error_message(e)}")
            continue
        curves[name] = _listed(result)
    return curves, problems


def _listed(values: np.ndarray) -> list[float | None]:
    """Return values as a list, None where they are not finite."""
    return [float(v) if math.isfinite(v) else None for v in values]


def _extra_values(value: Any) -> np.ndarray | None:
    """
    Return the array of an extra output (of data on a grid, or an array),
    or None if it is not one (e.g. a table).
    """
    data = getattr(value, 'data', value)
    return data if isinstance(data, np.ndarray) and data.ndim >= 1 else None


def _extra_unit(value: Any) -> str:
    """Return the unit of an extra output as text, '' if none is known."""
    unit = getattr(value, 'unit', None)
    return '' if unit is None else gbkfit.units.format_unit(unit)
