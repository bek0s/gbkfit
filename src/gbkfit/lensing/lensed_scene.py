"""
Scenes of galaxies in planes behind lenses (see LensedScene): each
plane drawn on a grid of its own, and their light composited, back to
front, on the grid of the image.
"""

import logging
from collections.abc import Sequence
from typing import NamedTuple, Protocol

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.grid import Coords, Grid, GridData
from gbkfit.model import DepthAxis, Extras, Scene, SceneState
from gbkfit.params import ParamValues
from .trace_plan import TracePlan


__all__ = [
    'EDGE_FLUX_LIMIT',
    'MEMORY_WARNING',
    'LensedPlane',
    'LensedScene',
    'LensedScenePlan',
    'SceneDrawing',
    'ScenePlanner'
]


_log = logging.getLogger(__name__)


# The share of the light of a plane on the border of its grid above which
# it is warned of: the rays that cross the plane beyond the grid see
# none of the light there
EDGE_FLUX_LIMIT = 1e-3

# The memory (bytes) of the drawing of a plane above which it is warned
# of (see LensedScenePlan): a grid that could be smaller (a mask of the
# data, its extent or its step, see SourceGrid)
MEMORY_WARNING = 2 ** 30

class SceneDrawing(Protocol):
    """
    The drawing of the light of galaxies on a grid, which an observation
    evaluates (e.g. in its cube): a ScenePlan (see gbkfit.model), or a
    lensed scene (see LensedScenePlan).
    """

    def evaluate(
            self,
            params: ParamValues,
            state: SceneState,
            data: DeviceArray,
            out_extra: Extras | None
    ) -> None:
        """
        Add the light of the galaxies to the data (the image or spectral
        cube), with the parameters of the evaluations and the state of
        the galaxies (see ScenePlan.evaluate).
        """

    def transmission(self) -> DeviceArray | None:
        """
        Return the transmission of its galaxies for the light from behind
        them at its last evaluation, or None if none enters or they have
        no opacity (see ScenePlan.transmission).
        """


class ScenePlanner(Protocol):
    """
    How an observation plans the scene of a plane on a grid (e.g. along
    its line of sight, see gbkfit.observation): the depth axes it is
    drawn on, and its plan on them.
    """

    def depth_axes(self, scene: Scene, grid: Grid) -> tuple[DepthAxis, ...]:
        """
        Return the depth axes on which a scene is drawn on a grid (of an
        image or a spectral cube): none on the plane of the sky, one for
        all its galaxies, or by radiative transfer one for each (see
        Scene.evaluation).
        """

    def plan(
            self,
            scene: Scene,
            grid: Grid,
            axes: tuple[DepthAxis, ...],
            subchannels: int | None
    ) -> SceneDrawing:
        """
        Plan a scene on a grid, on its depth axes there (see depth_axes),
        with the number of sub-channels of each channel that radiative
        transfer with line opacity marches it at, or None for those of
        the observation (see Depth: subchannels); e.g. 1 for a plane
        behind one with line opacity, which is drawn on its sub-channels
        already.
        """


class LensedPlane(NamedTuple):
    """
    A plane of a lensed scene (see LensedScene): the scene of its
    galaxies, their indices among the galaxies of the evaluations (whose
    states its plan takes; see LensedScenePlan.evaluate), and its index
    among the planes of the tracer (see Tracer), or None for the plane of
    the image: the galaxies in front of every lens, and those of the
    plane of the first lens, which it does not bend.
    """
    scene: Scene
    galaxies: tuple[int, ...]
    plane: int | None


class LensedScene:
    """
    Galaxies in planes of the sky, some behind lenses (see Tracer): the
    scene of each plane (see Scene), drawn on a grid of its own (see
    SourceGrid) and lensed onto the grid of the image, back to front,
    through the transmission of the planes in front (see
    LensedScenePlan).

    Every plane but the farthest has light from behind it (see
    Scene.behind), which its opacity dims.

    Parameters
    ----------
    planes : Sequence of LensedPlane
        The planes, near to far: the plane of the image first, if it has
        galaxies, then those behind lenses.

    Raises
    ------
    ValueError
        If no plane is behind a lens, the plane of the image is not the
        first, the planes are not near to far, or a scene has light from
        behind and is the farthest, or has none and is not.
    """

    def __init__(self, planes: Sequence[LensedPlane]):
        planes = tuple(planes)
        lensed = [plane.plane for plane in planes if plane.plane is not None]
        if not lensed:
            raise ValueError(
                "a lensed scene needs a plane behind a lens; draw a scene "
                "without one as it is")
        if any(plane.plane is None for plane in planes[1:]):
            raise ValueError("the plane of the image must be the first")
        if any(a >= b for a, b in zip(lensed, lensed[1:])):
            raise ValueError(
                f"the planes must be near to far; they are {lensed}")
        for k, plane in enumerate(planes):
            farthest = k == len(planes) - 1
            if plane.scene.behind() == farthest:
                name = 'the image' if plane.plane is None \
                    else str(plane.plane)
                raise ValueError(
                    f"the scene of plane {name} must "
                    f"{'not ' if farthest else ''}have light from behind it "
                    f"(see Scene: behind)")
        self._planes = planes

    def planes(self) -> tuple[LensedPlane, ...]:
        """Return its planes, near to far."""
        return self._planes

    def lensed(self) -> tuple[int, ...]:
        """
        Return the indices among the planes of the tracer of its planes
        behind lenses, near to far.
        """
        return tuple(
            plane.plane for plane in self._planes if plane.plane is not None)

    def plan(
            self,
            planner: ScenePlanner,
            trace: TracePlan,
            driver: Driver,
            grid: Grid,
            dtype: np.dtype
    ) -> 'LensedScenePlan':
        """
        Plan its evaluation on the grid of an image or a spectral cube (see
        LensedScenePlan).

        Parameters
        ----------
        planner : callable
            How the observation plans the scene of a plane on a grid (see
            ScenePlanner).
        trace : TracePlan
            The rays of the observation through the lenses, onto the grid
            of each plane behind them.
        driver : Driver
            The driver of the evaluation.
        grid : Grid
            The grid of the image (no spectral axis) or of the spectral
            cube.
        dtype : np.dtype
            The floating type of the evaluation.

        Returns
        -------
        LensedScenePlan
            The plan.

        Raises
        ------
        ValueError
            If the rays are not those of its planes behind lenses, or a
            plane of an image is behind one with line opacity.
        """
        return LensedScenePlan(self, planner, trace, driver, grid, dtype)


class _PlanePlan(NamedTuple):
    """
    The evaluation of a plane of a lensed scene (see LensedScenePlan): the
    plane, its grid, the plan of its scene on it, the array its light is
    drawn into, its transmission (None if it has none), its channels and
    the sub-channels of the light from behind it in each, the array of
    the light that leaves it toward the viewer, on the grid of the image
    (None until laid out), its label in the extra outputs, and the
    weights of the sums of its light (see _EdgeCheck; None for the plane
    of the image).
    """
    plane: LensedPlane
    grid: Grid
    plan: SceneDrawing
    emission: DeviceArray
    transmission: DeviceArray | None
    channels: int
    subchannels: int
    out: DeviceArray | None
    label: str
    edges: '_EdgeCheck | None'


class LensedScenePlan:
    """
    The evaluation of a lensed scene (see LensedScene) on the grid of an
    image or of a spectral cube: as a ScenePlan (see SceneDrawing), the
    light of its galaxies added to the image or cube (see evaluate), with
    their extra outputs.

    Each plane is drawn on its grid: the plane of the image on the grid
    of the image, and each plane behind lenses on its own (see
    TracePlan.grid), each by the plan of the observation (see
    ScenePlanner). Their light is then composited on the grid of the
    image, back to front: the light that leaves a plane toward the viewer
    is its own light at each ray, plus that from behind it through its
    transmission at the ray (see lens_composite of the native module
    DModel; the planes behind lenses at the positions of the rays on them,
    see TracePlan.positions).

    Behind a plane with line opacity, whose transmission is that of the
    sub-channels of each channel (see ScenePlan.transmission), the planes
    are drawn on their sub-channels (as channels, their own march at one
    sub-channel of each: see ScenePlanner), and their light is averaged
    over the sub-channels as it passes the plane: exact for lines behind
    lines narrower than a channel.

    With extra outputs, it adds those of the galaxies of each plane behind
    lenses with the id 'plane' ('plane{index}', by index in the tracer),
    on its grid, and for each such plane its light ('source_cube', on its
    grid), the positions of the rays on it ('ray_x' and 'ray_y') and the
    magnification of its image ('magnification'), on the grid of the
    image.

    It estimates the memory of the drawing of each plane before planning
    it (its light and transmission, and the fields on its depth axis:
    that of its opacity, and those of the extra outputs when asked for),
    logs it, and warns of more than MEMORY_WARNING held for sure. At
    each evaluation, until it warns of it once, it sums the light of
    each plane behind lenses on the border of its grid, on the device,
    and warns if it is more than EDGE_FLUX_LIMIT of its light: rays that
    cross the plane beyond the grid, which is fixed (e.g. as free lenses
    move them in a fit), see no light there.

    Parameters
    ----------
    scene : LensedScene
        The scene.
    planner : callable
        How the observation plans a scene of a plane (see ScenePlanner).
    trace : TracePlan
        The rays through the lenses.
    driver : Driver
        The driver of the evaluation.
    grid : Grid
        The grid of the image (no spectral axis) or of the spectral cube.
    dtype : np.dtype
        The floating type of the evaluation.

    Raises
    ------
    ValueError
        If the rays are not those of the planes behind lenses of the scene,
        or a plane of an image is behind one with line opacity.
    """

    def __init__(
            self,
            scene: LensedScene,
            planner: ScenePlanner,
            trace: TracePlan,
            driver: Driver,
            grid: Grid,
            dtype: np.dtype
    ):
        if trace.planes() != scene.lensed():
            raise ValueError(
                f"the rays are of the planes {list(trace.planes())}; those "
                f"of the scene behind lenses are {list(scene.lensed())}")
        self._trace = trace
        self._driver = driver
        self._image = grid if grid.spectral_axis is None else grid.spatial()
        self._dmodel = driver.native_class('DModel', dtype)()
        image_channels = 1 if grid.spectral_axis is None \
            else grid.size[grid.spectral_axis]
        channels = image_channels
        shape = self._image.size[::-1]
        # Near to far: each plane on the channels of the light from behind
        # the plane in front (the sub-channels of its own, with line
        # opacity)
        planes = []
        for plane in scene.planes():
            label = 'image' if plane.plane is None else f'plane{plane.plane}'
            per_channel = channels // image_channels
            plane_grid = grid if plane.plane is None else _plane_grid(
                trace.grid(plane.plane), grid, per_channel)
            axes = planner.depth_axes(plane.scene, plane_grid)
            _log_memory(label, plane.scene, plane_grid, channels, axes, dtype)
            plan = planner.plan(
                plane.scene, plane_grid, axes,
                1 if per_channel > 1 else None)
            plane_shape = (channels,) + plane_grid.size[1::-1]
            transmission = plan.transmission()
            subchannels = 1
            if transmission is not None:
                rows = transmission.shape[0]
                if rows % channels:
                    raise ValueError(
                        f"the transmission of plane {label} has {rows} rows "
                        f"for {channels} channels")
                subchannels = rows // channels
            emission = driver.mem_alloc_d(plane_shape, dtype)
            edges = None if plane.plane is None \
                else _EdgeCheck(driver, emission)
            planes.append(_PlanePlan(
                plane, plane_grid, plan, emission, transmission, channels,
                subchannels, None, label, edges))
            channels *= subchannels
        # Far to near: the light that leaves each plane behind lenses, on
        # the grid of the image (into that from behind it, without
        # sub-channels); and that of all of them
        behind = None
        for k in reversed(range(len(planes))):
            plane = planes[k]
            if behind is None or plane.subchannels > 1:
                out = driver.mem_alloc_d(
                    (plane.channels,) + shape, dtype)
            else:
                out = behind
            planes[k] = plane._replace(out=out)
            behind = out
        self._planes = tuple(planes)

    def transmission(self) -> None:
        """
        Return None: no light enters a lensed scene from behind it (it
        holds all the planes; see ScenePlan.transmission).
        """
        return None

    def evaluate(
            self,
            params: ParamValues,
            state: SceneState,
            data: DeviceArray,
            out_extra: Extras | None
    ) -> None:
        """
        Add the light of the galaxies to the data (the image or spectral
        cube), with their extra outputs and those of the planes (see the
        class).

        Parameters
        ----------
        params : ParamValues
            The values of the parameters of the evaluations: those of the
            galaxies (moved into the frame of the observation), of the
            lenses and of the offsets of the observation.
        state : SceneState
            The state of each of the galaxies (see LensedPlane:
            galaxies), and the mass on the sky of each galaxy whose mass
            the lenses bend light with (see TracePlan.positions).
        data : DeviceArray
            The image or spectral cube, (nchannels, ny, nx).
        out_extra : Extras, optional
            The extra outputs, if wanted.

        Raises
        ------
        ModelEvaluationError
            If the lenses have no deflection at a ray (e.g. off a map).
        """
        driver = self._driver
        positions = self._trace.positions(params, state.masses)
        for plane in self._planes:
            driver.mem_fill(plane.emission, 0)
            extras = out_extra
            if out_extra is not None and plane.plane.plane is not None:
                extras = out_extra.scoped(plane=plane.label)
            plane.plan.evaluate(
                params, state._replace(galaxies=tuple(
                    state.galaxies[g] for g in plane.plane.galaxies)),
                plane.emission, extras)
        behind = None
        for plane in reversed(self._planes):
            source_x, source_y = (None, None) if plane.plane.plane is None \
                else positions[plane.plane.plane]
            self._dmodel.lens_composite(
                source_x=source_x, source_y=source_y,
                emission=plane.emission, transmission=plane.transmission,
                behind=behind, out=plane.out)
            behind = plane.out
        driver.math_add(data, behind, out=data)
        for plane in self._planes:
            if plane.edges is not None:
                plane.edges.check(plane.label, self._trace.off_grid(
                    plane.plane.plane))
        if out_extra is not None:
            self._extras(out_extra)

    def _extras(self, out_extra: Extras) -> None:
        """
        Add the extra outputs of the planes behind lenses (see the class).
        """
        trace = self._trace
        image = self._image
        for plane in self._planes:
            index = plane.plane.plane
            if index is None:
                continue
            extras = out_extra.scoped(plane=plane.label)
            light = self._driver.mem_copy_d2h(plane.emission)
            # (an image is one channel, of a grid without a spectral axis)
            cube = light if plane.grid.spectral_axis is not None \
                else light[0]
            extras.add(
                'source_cube', _grid_data(cube, plane.grid), 'Light of the '
                'plane', "The light of the galaxies of the plane, on its "
                "grid, before the lenses in front bend it")
            ray_x, ray_y = trace.rays(index)
            for name, values, axis in (
                    ('ray_x', ray_x, 'x'), ('ray_y', ray_y, 'y')):
                extras.add(
                    name, _grid_data(values, image),
                    f'Rays on the plane ({axis})',
                    f"Where the ray of each pixel of the image crosses the "
                    f"plane, along {axis} (arcsec, in the frame of the "
                    f"observation); NaN where no light of the pixel reaches "
                    f"the data")
            extras.add(
                'magnification', _grid_data(trace.magnification(index), image),
                'Magnification', "The magnification of the image of the "
                "plane: the area of a pixel of the image over that of its "
                "rays on the plane (negative where the image is mirrored)")


def _plane_grid(spatial: Grid, grid: Grid, per_channel: int) -> Grid:
    """
    Return the grid of a plane behind lenses: its x and y axes (spatial),
    and the spectral axis of the grid of the image (grid), if any, of the
    given number of sub-channels of each of its channels (as channels).

    Raises
    ------
    ValueError
        If an image (no spectral axis) has sub-channels.
    """
    if grid.spectral_axis is None:
        if per_channel > 1:
            raise ValueError(
                "an image has no sub-channels for the planes behind a plane "
                "with line opacity")
        return spatial
    axis = grid.spectral_axis
    coords = grid.coords
    step = coords.step[axis] / per_channel
    # (the centre of sub-channel s of channel c is at c + (s + 1/2) /
    # per_channel - 1/2 pixels of the grid)
    rpix = coords.rpix[axis] * per_channel + (per_channel - 1) / 2
    return Grid(
        spatial.size + (grid.size[axis] * per_channel,),
        Coords(
            spatial.coords.step + (step,), spatial.coords.rpix + (rpix,),
            spatial.coords.rval + (coords.rval[axis],), 0, coords.rest,
            coords.medium, coords.convention, coords.frame),
        2)


def _grid_data(data: np.ndarray, grid: Grid) -> GridData:
    """Return an extra output of data on a grid."""
    return GridData(data, grid.coords, grid.spectral_axis)


class _EdgeCheck:
    """
    The check of the light of a plane on the border of its grid (see
    LensedScenePlan), at each evaluation until it warns once: the sums of
    the light of each channel over the grid and over its border, by one
    product of the light (nchannels, npixels) with their weights
    (npixels, 2), on the device.
    """

    def __init__(self, driver: Driver, light: DeviceArray):
        channels, size_y, size_x = light.shape
        border = np.ones((size_y, size_x))
        border[1:-1, 1:-1] = 0
        weights = np.stack([np.ones(size_y * size_x), border.ravel()], -1)
        self._driver = driver
        self._light = light.reshape(channels, size_y * size_x)
        self._weights = driver.mem_copy_h2d(weights.astype(light.dtype))
        self._sums = driver.mem_alloc_d((channels, 2), light.dtype)
        self._warned = False

    def check(self, label: str, off_grid: int) -> None:
        """
        Log a warning, once, if more than EDGE_FLUX_LIMIT of the light of
        the plane of the given label is on the border of its grid, with
        the number of the rays of the data that crossed the plane off it.
        """
        if self._warned:
            return
        driver = self._driver
        driver.math_matmul(self._light, self._weights, out=self._sums)
        total, border = driver.mem_copy_d2h(self._sums).sum(axis=0)
        if total > 0 and border > EDGE_FLUX_LIMIT * total:
            self._warned = True
            _log.warning(
                f"{border / total:.2g} of the light of {label} is on the "
                f"border of its grid, and {off_grid} rays of the data cross "
                f"the plane off it, where they see no light: give its "
                f"observation a source grid of a larger extent (see "
                f"SourceGrid); warned once")


def _log_memory(
        label: str,
        scene: Scene,
        grid: Grid,
        channels: int,
        axes: tuple[DepthAxis, ...],
        dtype: np.dtype
) -> None:
    """
    Log the memory of the drawing of a plane of a scene on a grid, of the
    given channels and on the given depth axes (see
    ScenePlanner.depth_axes), before it is planned, and warn if what it
    holds for sure is more than MEMORY_WARNING: its light, and its
    transmission if it has light from behind (of a row a channel; more
    with line opacity), and on a depth axis, a field there (of the
    voxels of the longest) for its dust or line opacity. On a depth axis,
    it also logs the fields that its extra outputs hold, when asked for
    (see Scene.extra_fields).
    """
    size_x, size_y = grid.size[:2]
    itemsize = np.dtype(dtype).itemsize
    light = channels * size_x * size_y * itemsize \
        * (2 if scene.behind() else 1)
    planes = max((axis.size for axis in axes), default=0)
    field = planes * size_x * size_y * itemsize
    opacity = scene.has_dust() or scene.line_opacity()
    held = light + (field if opacity else 0)
    message = (
        f"lensed scene, {label}: a grid of {size_x} x {size_y} pixels and "
        f"{channels} channels: about {_mib(light)} of light")
    if planes:
        extras = scene.extra_fields()
        held_field = ' (one for its opacity)' if opacity else ''
        message += (
            f", and {_mib(field)} for each field on its depth axis of "
            f"{planes} planes{held_field}; its extra outputs, when asked "
            f"for, hold up to {extras} fields ({_mib(extras * field)})")
    _log.info(message)
    if held > MEMORY_WARNING:
        _log.warning(
            f"{message}: give its observation a mask of the pixels of its "
            f"data, or a source grid of a smaller extent or a coarser step "
            f"(see SourceGrid)")


def _mib(size: int) -> str:
    """Return a number of bytes in MiB, for humans."""
    return f"{size / 2 ** 20:.0f} MiB"
