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
observation, through its instrument; and the data it is paired with, if
any).
"""

import dataclasses
import functools
import logging
import math
import os
import threading
from collections.abc import Callable, Mapping
from typing import Any, Literal

import astropy.constants
import astropy.units
import numpy as np
import numpy.typing as npt
import scipy.ndimage

import gbkfit.model
import gbkfit.project
import gbkfit.setup
import gbkfit.units
from gbkfit.config import ConfigError, error_message, paths_relative_to
from gbkfit.dataset import dataset_parser
from gbkfit.driver import Driver
from gbkfit.driver.drivers.host import DriverHost
from gbkfit.grid import doppler, sky_to_pixel
from gbkfit.model import (
    Component, Extras, Galaxy, GalaxyFunctions, galaxy_parser)
from gbkfit.observation import ObservationGroup, observation_parser
from gbkfit.params import ParamSpace

from .errors import Invalid
from .observations import channels_of, field_of
from .setups import setup_text
from .views import Cube, CubeAxis
from .workspace import Workspace


_log = logging.getLogger(__name__)

__all__ = [
    'GalaxyExtra',
    'GalaxyRender',
    'GalaxyVolume',
    'Observed',
    'ObservedArray',
    'Placement',
    'Renders',
    'View'
]


# How a galaxy is seen: at the orientation of its parameters, or turned as
# a whole so that its reference plane is face-on or edge-on (the posa and
# incl of that plane then; the major axis of an edge-on one along x; see
# _oriented)
View = Literal['sky', 'face-on', 'edge-on']
_ORIENTATIONS = {'face-on': (0, 0), 'edge-on': (90, 90)}

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
    (nz, ny, nx; the axis z along the axis of its reference plane, from
    below it to above it), the size of its voxels across (step) and along
    z (depth_step; arcsec), and the voxel of that plane along z (zero,
    0-based; fractional).
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
class Placement:
    """
    Where a galaxy is for an observation: its centre (x, y; arcsec, as the
    model measures positions) and its systemic velocity where its lines
    land on the channels (km/s; see _observed_velocities), those of its
    first component with an orientation (its components share one), or of
    its first component (None for what it has not, e.g. no lines); and
    whether they are in the field and in the channels of the observation
    (None for what it or the galaxy has not).
    """
    centre: tuple[float, float] | None
    velocity: float | None
    in_field: bool | None
    in_channels: bool | None


@dataclasses.dataclass
class Observed:
    """
    A galaxy as an observation sees it (see Renders.observed): the type of
    the observation (e.g. 'pixel_spectra'), its arrays (the model of each item,
    and the data and errors of the data, if it has them), the velocities of
    its channels (km/s; None without a spectral axis), and the warnings of
    the library as it set them up (e.g. a PSF narrower than the pixels),
    and its pixels on the sky: their size (x, y; arcsec) and the pixel (x,
    y; 0-based) that the model measures positions from; and where the
    galaxy is for it (see Placement; None of a setup, of many galaxies).
    """
    type: str
    arrays: list[ObservedArray]
    velocities: np.ndarray | None
    notes: list[str]
    step: tuple[float, float]
    rpix: tuple[float, float]
    placement: Placement | None = None


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
            field: float | None = None,
            centre: tuple[float, float] | None = None,
            progress: Callable[[str], None] | None = None
    ) -> GalaxyRender:
        """
        Render a galaxy of the options of a piece of a project, at the
        values of its parameters of the given properties (numbers, or
        lists of them for vectors; the defaults for the others), telling
        progress the stage it reaches, if given (e.g. 'evaluating'); over
        a field of the given width and centre (arcsec, x and y as the
        model measures them), if given (by default, about the origin, as
        wide as the galaxy). A galaxy that emits no light (e.g. of mass
        alone) has its curves but no maps (see _unlit).

        Raises
        ------
        Invalid
            If the galaxy or the values are not valid, or it cannot be
            rendered.
        """
        tell = progress or _quiet
        try:
            galaxy, values, extras, radius, step = self._evaluate(
                project, options, properties, view, size, tell, field,
                centre)
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
        Render the light of a galaxy in 3d: the brightness of each of its
        components in each voxel of the ideal observation that renders its
        maps (see galaxy), turned as a whole so that its reference plane is
        face-on, z along its axis (see _turn); those without a depth axis
        (e.g. thin disks seen only in bands) face-on in the voxels of that
        plane. Telling progress the stage it reaches, if given.

        Raises
        ------
        Invalid
            If the galaxy or the values are not valid, or it cannot be
            rendered.
        """
        tell = progress or _quiet
        galaxy, values, extras, _, step = self._evaluate(
            project, options, properties, 'sky', size, tell)
        tell('gathering the volume')
        cubes = [
            extra.value for extra in extras
            if extra.name == 'bdata' and 'component' in extra.ids
            and np.ndim(extra.value.data) == 3]
        planes = [
            extra.value for extra in extras
            if extra.name == 'bdata' and 'component' in extra.ids
            and np.ndim(extra.value.data) == 2]
        if cubes:
            coords = cubes[0].coords
            depth_step, zero = float(coords.step[2]), float(coords.rpix[2])
            light = _turned(
                sum(np.asarray(cube.data, dtype=float) for cube in cubes),
                coords, _turn(galaxy, values, 'face-on'),
                _centre(galaxy, values))
        else:
            depth_step, zero = step, 0.0
            light = np.zeros((1, size, size))
        if planes:
            # (without a depth axis, e.g. thin disks seen only in bands:
            # their light face-on, in the plane)
            _, _, flat, _, _ = self._evaluate(
                project, options, properties, 'face-on', size, tell)
            plane = int(round(min(max(zero, 0), len(light) - 1)))
            for extra in flat:
                if extra.name == 'bdata' and 'component' in extra.ids \
                        and np.ndim(extra.value.data) == 2:
                    light[plane] += np.asarray(extra.value.data)
        return GalaxyVolume(step, depth_step, zero, light.astype(np.float32))

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
            tell: Callable[[str], None],
            field: float | None = None,
            centre: tuple[float, float] | None = None
    ) -> tuple[Galaxy, dict[str, Any], Extras, float, float]:
        """
        Evaluate a galaxy of the options of a piece of a project through
        its ideal observations (see _observations), seen as given, over a
        field of a width and centre, if given (see galaxy): return the
        galaxy, the values of its parameters, the extra outputs of the
        evaluation, its largest radius and the size of its pixels (arcsec).

        Raises
        ------
        Invalid
            If the galaxy or the values are not valid, or it cannot be
            evaluated (e.g. it has no component that emits light).
        """
        tell('setting up')
        galaxy, radius, step = self._load(project, options, view, size)
        if field is not None:
            if not field > 0:
                raise Invalid(f"the field must be wider than 0; it is {field}")
            step = field / size
        rpix = None if centre is None else [
            (size - 1) / 2 - c / step for c in centre]
        try:
            observations = _observations(galaxy, size, step, rpix=rpix)
            if not observations:
                raise _NoLight("the galaxy has no component that emits light")
            group = ObservationGroup([galaxy], observations)
            values = _values(group, properties)
            plan = group.plan(_driver())
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
            seen = _seen(galaxy, gbkfit.model.View(
                True, gbkfit.model.LinesTracer()))
            if not seen:
                raise Invalid(
                    "the galaxy has no component that emits lines: a cube "
                    "sees lines")
            observations = _observations(galaxy, size, step)
            group = ObservationGroup([galaxy], observations)
            values = _oriented(galaxy, _values(group, properties), view)
            extras = Extras()
            tell('evaluating the maps')
            group.plan(_driver()).model_h(values, extras)
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
            data = group.plan(_driver()).model_h(values)[0]
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
            observation: str | Mapping[str, Any],
            dataset: str | Mapping[str, Any] | None = None,
            shift: tuple[float, float] | None = None,
            progress: Callable[[str], None] | None = None
    ) -> Observed:
        """
        Evaluate a galaxy of the options of a piece of a project, at the
        values of its parameters of the given properties (the defaults for
        the others, also of the observation), through an observation of
        the project (by name, or its options, e.g. as it is edited: the
        pieces it names, its PSF, LSF and noise, in place), paired with a
        dataset if given (a piece of the project by name, or its options;
        the data win, as in a setup: their sampling replaces that of the
        observation): its model on the grid of the observation, through its
        instrument and line of sight, the data, if any, and where the
        galaxy is for it (see Placement); the galaxy moved on the sky by
        the given shift, if any (x, y; arcsec; see _shifted); telling
        progress the stage it reaches, if given.

        Raises
        ------
        Invalid
            If the galaxy, the observation, the dataset or the values are
            not valid, or the observation cannot see the galaxy (e.g. a
            thick disk along a flat line of sight).
        """
        tell = progress or _quiet
        tell('setting up')
        root = self._workspace.path(project)
        notes: list[str] = []
        # (the dataset at the place of the observation, as a setup pairs
        # them: the pieces resolved in place)
        config: dict[str, Any] = dict(observations=[
            observation if isinstance(observation, str)
            else dict(observation)])
        if dataset is not None:
            config['datasets'] = [dataset]
        try:
            with _noted(notes):
                resolved = gbkfit.project.resolve(
                    config, os.path.join(root, 'observe.yaml'),
                    gbkfit.project.open(root))
                with paths_relative_to(root):
                    galaxy = galaxy_parser.load(dict(options))
                    data = None if dataset is None else dataset_parser.load(
                        resolved.config['datasets'][0])
                    seen = observation_parser.load(
                        resolved.config['observations'], dataset=[data])
                group = ObservationGroup([galaxy], seen)
                values = _shifted(galaxy, _values(group, properties), shift)
                plan = group.plan(_driver())
                tell('evaluating')
                outputs = gbkfit.setup.evaluation_outputs(
                    plan, None, values, True, False, False) \
                    | gbkfit.setup.data_outputs(plan)
        except (ConfigError, ValueError, TypeError, RuntimeError) as e:
            raise Invalid(error_message(e)) from e
        observed = _observed(seen[0], outputs, None, notes)
        observed.placement = _placement(galaxy, values, seen[0])
        return observed

    def setup_observed(
            self,
            path: str,
            setup: Mapping[str, Any],
            observation: int,
            progress: Callable[[str], None] | None = None
    ) -> Observed:
        """
        Evaluate a setup of the given options as the file of a path (of a
        project, relative to the workspace; e.g. as it is edited), at the
        starts of its parameters (see ParamSpace.reference), through its
        observation of the given index: its model on the grid of the
        observation, and its data, if it has them; telling progress the
        stage it reaches, if given.

        Raises
        ------
        Invalid
            If the setup has an error, or the observation is not one of it.
        """
        tell = progress or _quiet
        tell('setting up')
        filename = self._workspace.path(path)
        notes: list[str] = []
        try:
            with _noted(notes):
                description = gbkfit.setup.describe(
                    filename, {filename: setup_text(setup)})
                errors = [
                    p for p in description.problems if p.severity == 'error']
                if errors:
                    raise Invalid(errors[0].message)
                group, space = description.group, description.params
                if group is None or space is None:
                    raise Invalid("the setup has no galaxies and observations")
                if not 0 <= observation < group.nobservations():
                    raise Invalid(
                        f"the setup has no observation {observation}")
                values = space.reference()
                with paths_relative_to(description.root or os.getcwd()):
                    plan = group.plan(_driver())
                tell('evaluating')
                outputs = gbkfit.setup.evaluation_outputs(
                    plan, None, values, True, False, False) \
                    | gbkfit.setup.data_outputs(plan)
        except (ConfigError, ValueError, TypeError, RuntimeError) as e:
            raise Invalid(error_message(e)) from e
        return _observed(
            group.observations()[observation], outputs,
            group.observation_label(observation), notes)

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


@functools.cache
def _driver() -> Driver:
    """
    Return the driver of the renders of this process: the first GPU, if
    there is one that CUDA can use (renders of thick components, cubes as
    deep as they are wide, are many times faster there), or else the host.
    """
    try:
        from gbkfit.driver.drivers.cuda import DriverCuda
        return DriverCuda()
    except Exception as e:  # (no CuPy, no GPU, or no CUDA module built)
        _log.info(f"renders on the host: no GPU ({e})")
        return DriverHost()


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


def _seen(galaxy: Galaxy, view: gbkfit.model.View) -> list[Component]:
    """
    Return the components of a galaxy whose light an observation of the
    given view sees (see View.sees).
    """
    return [
        c for c in galaxy.components() if c.light() is not None
        and view.sees(c.light(), c.moves())]


def _observations(
        galaxy: Galaxy, size: int, step: float,
        spectral: dict[str, Any] | None = None,
        rpix: list[float] | None = None
) -> list[Any]:
    """
    Return the ideal observations of a galaxy, of size pixels of the given
    step (their reference pixel, where the model measures positions from,
    that given, else the centre): a cube of the components that emit
    lines (of the channels of the options of a spectral axis given, or
    else of two, whose velocities its maps hold whatever its channels; at
    the rest of the first line named, if any), and an image of those that
    emit in bands; each evaluated along the line of sight as what it sees
    needs (see Scene.evaluation of gbkfit.model).
    """
    infos = []
    for view, info in (
            (gbkfit.model.View(True, gbkfit.model.LinesTracer()), dict(
                type='pixel_spectra',
                spectral=(spectral or dict(size=2, step=100))
                | dict(convention='relativistic'))),
            (gbkfit.model.View(False, gbkfit.model.ContinuumTracer()),
             dict(type='pixel_brightness'))):
        seen = _seen(galaxy, view)
        if not seen:
            continue
        # (the convention then that of the rest)
        if view.spectral and (rest := _rest(seen)) is not None:
            info = info | dict(spectral={
                key: value for key, value in info['spectral'].items()
                if key != 'convention'} | dict(rest=rest))
        infos.append(info | dict(size=[size, size], step=[step, step]))
        if rpix is not None:
            infos[-1]['rpix'] = list(rpix)
    return observation_parser.load(infos)


def _rest(components: list[Any]) -> str | None:
    """
    Return the rest of the first line named by components that emit lines,
    as an option of a spectral axis (its text, in full); None if they name
    none.
    """
    for component in components:
        if lines := component.light().lines().lines():
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
        if (lines := component.light().lines().lines())]
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
        for line in component.light().lines().lines() or ()]
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


def _observed(
        observation: Any,
        outputs: Mapping[str, Any],
        label: str | None,
        notes: list[str]
) -> Observed:
    """
    Return what an observation sees of an evaluation (see Renders.observed):
    the model, data and errors of its items among the outputs (of the
    observation of the label, if given; else all of them), the velocities
    of its channels, the notes (with one more if the model has no light),
    and its pixels.
    """
    light = observation.data_unit()
    arrays = []
    for output in outputs.values():
        if label is not None and output.ids.get('observation') != label:
            continue
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


def _placement(
        galaxy: Galaxy, values: Mapping[str, Any], observation: Any
) -> Placement:
    """
    Return where a galaxy is for an observation, at the values of its
    parameters (see Placement).
    """
    frames = [
        galaxy.component_frame_params(index)
        for index in range(len(galaxy.components()))]
    oriented = [frame for frame in frames if frame.posa is not None]
    frame = (oriented or frames or [None])[0]

    def first(name: str | None) -> float | None:
        # (the value of a parameter, the first of a vector)
        if name is None:
            return None
        return float(np.ravel(np.asarray(values[name], dtype=float))[0])
    x, y = (None, None) if frame is None else (first(frame.x), first(frame.y))
    centre = None if x is None or y is None else (x, y)
    vsys = None if frame is None else first(frame.vsys)
    lines = _seen(galaxy, gbkfit.model.View(True, gbkfit.model.LinesTracer()))
    velocity = None if vsys is None or not lines else _observed_velocities(
        (vsys, vsys), _cube_axis(galaxy.redshift(), lines))[0]
    in_field = None
    if centre is not None and field_of(observation) is not None:
        grid = observation.grid()
        matrix, offset = sky_to_pixel(grid)
        pixel = matrix @ np.asarray(centre) + offset
        in_field = all(
            -0.5 <= float(p) <= n - 0.5 for p, n in zip(pixel, grid.size))
    in_channels = None
    channels = channels_of(observation)
    if velocity is not None and channels is not None:
        low, high = sorted((channels.first, channels.last))
        half = abs(channels.step) / 2
        in_channels = low - half <= velocity <= high + half
    return Placement(centre, velocity, in_field, in_channels)


def _values(
        group: ObservationGroup | Galaxy, properties: Mapping[str, Any]
) -> dict[str, Any]:
    """
    Return the values of the parameters of a group (or of a galaxy alone):
    those of the given properties (numbers or lists of them, of keys that
    can select elements, e.g. disk_posa[0:4], and ties), and the defaults
    of the parameters that no key names.

    Raises
    ------
    ConfigError
        If a property is not of a parameter of the group, or the
        properties are not valid (see ParamSpace).
    """
    pdescs = group.pdescs()
    # (the parameter of each key: its name, without the elements it
    # selects)
    named = {key.split('[', 1)[0].strip() for key in properties}
    if unknown := sorted(named - set(pdescs)):
        raise ConfigError(f"unknown parameters: {unknown}")
    filled = dict(properties) | {
        name: _default(desc) for name, desc in pdescs.items()
        if name not in named}
    space = ParamSpace.from_properties(pdescs, filled)
    return dict(space.evaluate())


def _shifted(
        galaxy: Galaxy, values: Mapping[str, Any],
        shift: tuple[float, float] | None
) -> dict[str, Any]:
    """
    Return the values of the parameters of a galaxy moved on the sky by a
    shift (x, y; arcsec), if given: the centre of each of its components
    (of every ring, of a vector) moved by it, once for a centre that
    several share.
    """
    moved = dict(values)
    if shift is None:
        return moved
    names = set()
    for index in range(len(galaxy.components())):
        frame = galaxy.component_frame_params(index)
        for name, delta in ((frame.x, shift[0]), (frame.y, shift[1])):
            if name is None or name in names:
                continue
            names.add(name)
            value = np.asarray(moved[name], dtype=float) + delta
            moved[name] = float(value) if value.ndim == 0 else value
    return moved


def _default(desc: Any) -> Any:
    """Return the default value of a parameter, a list for a vector."""
    default = desc.default()
    return [default] * desc.size() if desc.size() > 1 else default


def _oriented(galaxy: Galaxy, values: dict[str, Any], view: View) -> dict:
    """
    Return the values of a galaxy seen in a view: turned as a whole (see
    _turn), every ring of every component with an orientation (a posa and
    an incl) turned with it, so that warps and the angles between
    components keep. The centres stay: its components share one (see
    GalaxyStage).
    """
    turn = _turn(galaxy, values, view)
    if turn is None:
        return values
    values = dict(values)
    for names in dict.fromkeys(_orientations(galaxy)):
        posa, incl = np.broadcast_arrays(*(
            np.asarray(values[name], dtype=float) for name in names))
        turned = _plane_angles(_plane_frames(posa, incl) @ turn.T)
        values.update(zip(names, turned))
    return values


def _orientations(galaxy: Galaxy) -> list[tuple[str, str]]:
    """
    Return the names of the posa and the incl of each component of a
    galaxy that has an orientation, in their order.
    """
    return [
        (frame.posa, frame.incl) for frame in (
            galaxy.component_frame_params(index)
            for index in range(len(galaxy.components())))
        if frame.posa is not None and frame.incl is not None]


def _turn(
        galaxy: Galaxy, values: Mapping[str, Any], view: View
) -> np.ndarray | None:
    """
    Return the turn of the sky (a rotation, 3 by 3, of x, y and the line
    of sight) that shows a galaxy in a view, at the values of its
    parameters: that which takes its reference plane (the first ring of
    its first component with an orientation) face-on or edge-on; None in
    the view of the sky, or without an orientation.
    """
    oriented = _orientations(galaxy)
    if view == 'sky' or not oriented:
        return None
    reference = _plane_frames(*(
        np.ravel(np.asarray(values[name], dtype=float))[0]
        for name in oriented[0]))
    # (sky to plane: frame @ sky, so the frame of a plane turned is frame
    # @ turn.T)
    return _plane_frames(*_ORIENTATIONS[view]).T @ reference


def _plane_frames(posa: npt.ArrayLike, incl: npt.ArrayLike) -> np.ndarray:
    """
    Return the rotations from the sky to the planes of the given position
    angles and inclinations (degrees), as the kernels turn the sky
    (gmodels.hpp, transform_posa and transform_incl): arrays of 3 by 3,
    whose rows are the major axis of each plane (its line of nodes), its
    minor axis and its normal, on the sky (x, y and the line of sight).
    """
    p, i = np.radians(posa), np.radians(incl)
    zero = np.zeros_like(p)
    return np.stack([
        np.stack([-np.sin(p), np.cos(p), zero], axis=-1),
        np.stack([-np.cos(i) * np.cos(p), -np.cos(i) * np.sin(p),
                  np.sin(i)], axis=-1),
        np.stack([np.sin(i) * np.cos(p), np.sin(i) * np.sin(p),
                  np.cos(i)], axis=-1)], axis=-2)


def _plane_angles(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the position angles and inclinations (degrees) of planes of
    the given rotations from the sky (see _plane_frames): those of their
    normals, and the position angle of the major axis of a face-on one.
    A rotation that also turns a plane within itself (a ring turned with
    a galaxy, not its reference plane) keeps its plane, not that turn.
    """
    normal = frames[..., 2, :]
    incl = np.degrees(np.arccos(np.clip(normal[..., 2], -1.0, 1.0)))
    tilted = np.hypot(normal[..., 0], normal[..., 1]) > 1e-12
    posa = np.degrees(np.where(
        tilted, np.arctan2(normal[..., 1], normal[..., 0]),
        np.arctan2(-frames[..., 0, 0], frames[..., 0, 1])))
    return posa, incl


def _centre(galaxy: Galaxy, values: Mapping[str, Any]) -> np.ndarray:
    """
    Return the centre of a galaxy on the sky (x, y, and 0 along the line of
    sight; arcsec), about which it turns (see _turn): that of its first
    component with an orientation (its components share one), or of its
    first component.
    """
    frames = [
        galaxy.component_frame_params(index)
        for index in range(len(galaxy.components()))]
    oriented = [frame for frame in frames if frame.posa is not None]
    frame = (oriented or frames)[0]
    return np.array([
        float(np.ravel(np.asarray(values[name], dtype=float))[0])
        for name in (frame.x, frame.y)] + [0.0])


def _turned(
        cube: np.ndarray,
        coords: Any,
        turn: np.ndarray | None,
        centre: np.ndarray
) -> np.ndarray:
    """
    Return a cube of the sky (z, y, x, of the coordinates of its grid;
    the model measures each axis from its reference pixel) turned about a
    centre by a turn of the sky (see _turn), on the same grid: each voxel
    the trilinear interpolation of the cube where the turn takes it from;
    0 beyond it. The cube itself without a turn.
    """
    if turn is None:
        return cube
    step = np.asarray(coords.step[:3], dtype=float)
    rpix = np.asarray(coords.rpix[:3], dtype=float)
    shape = cube.shape[::-1]
    # (the positions of the voxels, x, y and z, from the centre)
    axes = [(np.arange(n) - rpix[a]) * step[a] - centre[a]
            for a, n in enumerate(shape)]
    x, y, z = np.meshgrid(*axes, indexing='ij')
    turned = np.stack((x, y, z), axis=-1) @ turn
    source = [
        (turned[..., a] + centre[a]) / step[a] + rpix[a] for a in range(3)]
    values = scipy.ndimage.map_coordinates(
        cube, [source[2].T, source[1].T, source[0].T], order=1,
        mode='constant', cval=0.0, prefilter=False)
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
