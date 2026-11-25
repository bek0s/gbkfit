"""
The depth axis of the observations on the sky, along the line of sight
(Depth), and the evaluation of what they see along it, which the
components they see decide (see Scene.evaluation of gbkfit.model): on the
plane of the sky, on a depth axis, or by radiative transfer along it;
and of the planes behind lenses, each on a grid of its own (see
plan_scenes). Along the depth axis, the viewer is in front of its first
voxel.
"""

from collections.abc import Callable
from typing import Annotated, Any, Literal, TypeAlias

import astropy.units
import numpy as np
import scipy.constants
from annotated_types import Ge, Gt, Le

from gbkfit.config import Choices, Serializable, Unit, check_options, warn
from gbkfit.driver import Driver
from gbkfit.grid import Grid, VelocityConvention
from gbkfit.lensing import (
    LensedPlane, LensedScene, SceneDrawing, SourceGrid, TracePlan)
from gbkfit.model import (
    MAX_SUBCHANNELS, AdditiveScenePlan, Reference, Scene, SceneEvaluation,
    ScenePlan, TransferScenePlan, View)
from gbkfit.model.components.base import DepthAxis
from ._scene import SeenScenes


__all__ = [
    'Depth'
]


# A length on the sky
_Length: TypeAlias = Annotated[float, Gt(0), Unit('arcsec')]

# The extents of a depth axis that are not lengths (see Depth)
_Extent: TypeAlias = Annotated[
    Literal['galaxy', 'legacy'], Choices(
        galaxy='Spans the galaxies', legacy='Longer image axis (legacy)')]


class Depth(Serializable):
    """
    The depth axis of the grid of an observation, along the line of sight:
    centred on the plane of the sky. The observations whose galaxies need
    one have it (thick components, or opacity; see Scene.evaluation), and
    radiative transfer along it takes the options slab and subchannels
    (line opacity, or galaxies that dim one another); the others ignore
    them.

    Parameters
    ----------
    extent : str or float, optional
        Its length (arcsec), or:

        - 'galaxy' (the default): that which holds the galaxies the
          observation evaluates, at any inclination: the diagonal of the
          x and y axes of the grid they are evaluated on (the grid of the
          data, oversampled and padded for the convolutions), the longest
          line across the field, for disks, which do not end at their
          last ring (see Disk), or twice the bounding radius of the
          largest of the components that have one (e.g. a spheroid, see
          Component.bounding_radius), if longer; through radiative
          transfer, where each galaxy has a depth axis of its own, that of
          each galaxy;
        - 'legacy': the longer (in arcsec) of those x and y axes, as the
          3d models of earlier versions took it.

        A shorter axis than that of 'galaxy' (of 'legacy' or a length)
        loses the density beyond its ends: a warning says so for a length
        (see warn); 'legacy' always is (the diagonal is longer), by
        choice.
    step : float, optional
        The size of its voxels (arcsec); by default that of the pixels
        along the longer (in arcsec) of the x and y axes of that grid.
    slab : int, optional
        By radiative transfer, the number of its planes evaluated at once,
        which bounds the memory of the evaluation (the fields of the
        emitters and the cubes of opacity of a slab of planes); all by
        default (fastest). The clouds of Monte Carlo disks are drawn again
        for each slab.
    subchannels : int, optional
        By radiative transfer with line opacity, the number of sub-channels
        of each channel that the march transfers the light at: the light
        of a channel is their mean (from 1, the default, to 64). Line
        opacity averaged over a channel wider than its lines dims their
        light too little; sub-channels resolve the lines, at their number
        times the time of the march (light without line opacity needs
        none).
    """

    config_titles = dict(
        extent='Extent', step='Voxel size', slab='Planes per slab',
        subchannels='Sub-channels')

    def dump(self) -> dict[str, Any]:
        return dict(
            extent=self._extent, step=self._step, slab=self._slab,
            subchannels=self._subchannels)

    def __init__(
            self,
            extent: _Extent | _Length = 'galaxy',
            step: _Length | None = None,
            slab: Annotated[int, Gt(0)] | None = None,
            subchannels: Annotated[int, Ge(1), Le(MAX_SUBCHANNELS)] = 1
    ):
        check_options(Depth.__init__, locals())
        self._extent = extent
        self._step = step
        self._slab = slab
        self._subchannels = subchannels

    def extent(self) -> str | float:
        """Return its extent: a length, 'galaxy' or 'legacy'."""
        return self._extent

    def step(self) -> float | None:
        """Return the size of its voxels, if given."""
        return self._step

    def slab(self) -> int | None:
        """
        Return the number of planes radiative transfer evaluates at once,
        if given.
        """
        return self._slab

    def subchannels(self) -> int:
        """
        Return the number of sub-channels of each channel of radiative
        transfer with line opacity.
        """
        return self._subchannels

    def axis(
            self, grid: Grid, bounding_radius: float
    ) -> DepthAxis:
        """
        Return the axis on the given grid of the x and y axes, for
        galaxies of the given bounding radius (arcsec, of those of their
        components that have one; see Component.bounding_radius): as many
        voxels as cover the extent, centred on 0.

        Parameters
        ----------
        grid : Grid
            The grid of the x and y axes the galaxies are evaluated on.
        bounding_radius : float
            The bounding radius of the galaxies (arcsec).

        Returns
        -------
        DepthAxis
            The axis.
        """
        # (the longer axis in arcsec, which pixels that are not square can
        # make the one of fewer pixels)
        lengths = [n * abs(d) for n, d in zip(grid.size, grid.coords.step)]
        longest = int(lengths[0] < lengths[1])
        step = self._step if self._step is not None \
            else grid.coords.step[longest]
        galaxy = max(float(np.hypot(*lengths)), 2 * bounding_radius)
        if self._extent == 'legacy':
            size = grid.size[longest]
        else:
            extent = galaxy if self._extent == 'galaxy' else self._extent
            size = max(1, int(np.ceil(extent / step - 1e-9)))
            if size * step < galaxy * (1 - 1e-9):
                warn(
                    f"the depth axis ({size * step:g} arcsec) is shorter "
                    f"than the galaxies need ({galaxy:g} arcsec, the "
                    f"diagonal of the grid or twice their bounding radius): "
                    f"their density beyond its ends is lost; its extent "
                    f"'galaxy' holds it")
        return DepthAxis(size, step, -(size / 2 - 0.5) * step)


def check_scene(scene: Scene) -> None:
    """
    Check, without a driver, that a scene can be evaluated as it needs
    (see Scene.evaluation): by radiative transfer, the dust of all its
    galaxies in one set of cubes of opacity (see TransferScenePlan.check);
    otherwise each galaxy dimmed by its own dust (see
    AdditiveScenePlan.check).

    Raises
    ------
    ValueError
        If it cannot.
    """
    if scene.evaluation() is SceneEvaluation.TRANSFER:
        TransferScenePlan.check(scene)
    else:
        AdditiveScenePlan.check(scene)


def depth_axes(
        scene: Scene, depth: Depth, grid: Grid
) -> tuple[DepthAxis, ...]:
    """
    Return the depth axes on which a scene is evaluated on a grid, as it
    needs (see Scene.evaluation): none on the plane of the sky; one on a
    depth axis, for the largest of its galaxies (see Depth.axis); and by
    radiative transfer, one for each galaxy.

    Parameters
    ----------
    scene : Scene
        The scene.
    depth : Depth
        The depth axis of the observation.
    grid : Grid
        The grid of the image or of the spectral cube.

    Returns
    -------
    tuple of DepthAxis
        The axes.
    """
    spatial = grid.spatial()
    radii = scene.bounding_radii()
    match scene.evaluation():
        case SceneEvaluation.PLANE:
            return ()
        case SceneEvaluation.DEPTH:
            return (depth.axis(spatial, max(radii)),)
        case SceneEvaluation.TRANSFER:
            return tuple(depth.axis(spatial, radius) for radius in radii)


def plan_scene(
        scene: Scene,
        depth: Depth,
        axes: tuple[DepthAxis, ...],
        driver: Driver,
        grid: Grid,
        dtype: np.dtype,
        smoothing: float | None
) -> ScenePlan:
    """
    Plan the evaluation of a scene as it needs (see Scene.evaluation), on
    its depth axes (see depth_axes), on a driver and dtype, on the grid of
    an image (no spectral axis) or a spectral cube, for an instrument that
    smooths the spectra of the model by the given dispersion (km/s; None
    for an image), against which the resolution of the light is checked:

    - on the plane of the sky, or on the depth axis of the scene: each
      galaxy adds its light to the image or the spectral cube, dimmed by
      its own opacity (see AdditiveScenePlan);
    - by radiative transfer, each galaxy on its depth axis (see
      TransferScenePlan), with the options slab and subchannels of the
      depth axis of the observation.

    Parameters
    ----------
    scene : Scene
        The scene.
    depth : Depth
        The depth axis of the observation.
    axes : tuple of DepthAxis
        The depth axes of the scene on the grid (see depth_axes).
    driver : Driver
        The driver of the evaluation.
    grid : Grid
        The grid of the image or of the spectral cube.
    dtype : np.dtype
        The floating type of the evaluation.
    smoothing : float or None
        The dispersion (km/s) by which the instrument smooths the spectra
        of the model; None for an image.

    Returns
    -------
    ScenePlan
        The plan.
    """
    match scene.evaluation():
        case SceneEvaluation.PLANE:
            return AdditiveScenePlan(
                scene, driver, grid, None, dtype, smoothing)
        case SceneEvaluation.DEPTH:
            return AdditiveScenePlan(
                scene, driver, grid, axes[0], dtype, smoothing)
        case SceneEvaluation.TRANSFER:
            return TransferScenePlan(
                scene, driver, grid, list(axes), dtype, depth.slab(),
                depth.subchannels(), smoothing)


class _PlanePlanner:
    """
    How an observation plans the scene of each plane of a lensed scene on
    its grid (see ScenePlanner): on the depth axes of the observation
    (see depth_axes and plan_scene), with the sub-channels the plane
    takes, if given (one for a plane drawn on the sub-channels of one with
    line opacity).
    """

    def __init__(
            self,
            depth: Depth,
            driver: Driver,
            dtype: np.dtype,
            smoothing: float | None
    ):
        self._depth = depth
        self._driver = driver
        self._dtype = dtype
        self._smoothing = smoothing

    def depth_axes(self, scene: Scene, grid: Grid) -> tuple[DepthAxis, ...]:
        """Return the depth axes of a scene on a grid (see depth_axes)."""
        return depth_axes(scene, self._depth, grid)

    def plan(
            self,
            scene: Scene,
            grid: Grid,
            axes: tuple[DepthAxis, ...],
            subchannels: int | None
    ) -> ScenePlan:
        """
        Plan a scene on a grid, on its depth axes there, with the given
        sub-channels, if any (see plan_scene).
        """
        depth = self._depth
        if subchannels is not None:
            depth = Depth(
                depth.extent(), depth.step(), depth.slab(), subchannels)
        return plan_scene(
            scene, depth, axes, self._driver, grid, self._dtype,
            self._smoothing)


def plan_scenes(
        scenes: SeenScenes,
        depth: Depth,
        source_grid: SourceGrid,
        driver: Driver,
        grid: Grid,
        dtype: np.dtype,
        smoothing: float | None,
        footprint: Callable[[], np.ndarray],
        reference: Reference | None
) -> SceneDrawing:
    """
    Plan the evaluation of the scenes of an observation (see SeenScenes)
    on the grid of an image (no spectral axis) or a spectral cube: its one
    scene as it needs (see plan_scene), or with planes behind lenses, each
    plane on a grid of its own (see SourceGrid), its light bent onto the
    grid through the lenses in front (see LensedScene and TracePlan).

    Parameters
    ----------
    scenes : SeenScenes
        The scenes.
    depth : Depth
        The depth axis of the observation, for the scenes that need one.
    source_grid : SourceGrid
        How the grids of the planes behind lenses are laid out.
    driver : Driver
        The driver of the evaluation.
    grid : Grid
        The grid of the image or of the spectral cube.
    dtype : np.dtype
        The floating type of the evaluation.
    smoothing : float or None
        The dispersion (km/s) by which the instrument smooths the spectra
        of the model; None for an image.
    footprint : callable
        Return the pixels of the grid whose light can reach the data (a
        bool array of shape (ny, nx)), from which the rays through the
        lenses are traced; called with planes behind lenses only.
    reference : Reference, optional
        The values of the parameters at which the grids of the planes
        behind lenses are laid out (see ParamSpace.reference), with the
        masses of galaxies there; required with them.

    Returns
    -------
    SceneDrawing
        The plan: a ScenePlan, or a LensedScenePlan.

    Raises
    ------
    ValueError
        If there are planes behind lenses and no reference values, the
        lenses have no deflection at their rays there (e.g. off a map), or
        dust in front of planes behind lenses has an extinction law that
        is not of the wavelengths at which the observation sees them (see
        _check_dust_behind).
    """
    if not scenes.lensed():
        scene = scenes.scene()
        return plan_scene(
            scene, depth, depth_axes(scene, depth, grid), driver, grid,
            dtype, smoothing)
    if reference is None:
        raise ValueError(
            "an observation that sees planes behind lenses lays out their "
            "grids at reference values of the parameters (see "
            "ParamSpace.reference): plan its group with them (see "
            "ObservationGroup.plan)")
    _check_dust_behind(scenes, grid)
    seen = scenes.seen
    trace = TracePlan(
        seen.tracer, seen.lens_mappings,
        [plane.traced for plane in scenes.planes if plane.traced is not None],
        grid, footprint(), seen.sky_origin, seen.shift.alignment,
        seen.shift.offsets[:2], reference, source_grid, driver, dtype)
    lensed = LensedScene([
        LensedPlane(scene, plane.galaxies, plane.traced)
        for plane, scene in zip(scenes.planes, scenes.scenes)])
    return lensed.plan(
        _PlanePlanner(depth, driver, dtype, smoothing), trace, driver, grid,
        dtype)


def _check_dust_behind(scenes: SeenScenes, grid: Grid) -> None:
    """
    Raise ValueError if the extinction law of the dust of a galaxy that
    dims light from behind it (see Scene.behind) is not of the wavelengths
    at which an observation of the scenes sees that light, on the grid of
    an image (of the band of its view) or of a spectral cube (of its
    channels, of an axis of a rest): as far as the redshift of the galaxy
    tells, at its rest. Extinction laws are not extrapolated: the error
    says to leave the galaxy out of the observation (e.g. the lens galaxy
    of a joint fit of an observation in the optical and one at mm
    wavelengths, at which its dust and its light are negligible).
    """
    observed = _observed_wavelengths(grid, scenes.view())
    if observed is None:
        return
    seen = scenes.seen
    for plane, scene in zip(scenes.planes, scenes.scenes):
        if not scene.behind():
            continue
        for g in plane.galaxies:
            galaxy = seen.galaxies[g]
            rest = observed / (1 + galaxy.redshift())
            for i in galaxy.absorbers():
                law = galaxy.components()[i].opacity().law()
                if law is None:
                    continue
                low, high = law.range()
                if rest.min() < low or rest.max() > high:
                    raise ValueError(
                        f"the dust of galaxy {seen.labels[g]!r} dims the "
                        f"light from behind it (of the planes behind lenses), "
                        f"which the observation sees from {rest.min():.4g} to "
                        f"{rest.max():.4g} um at the rest of the galaxy; its "
                        f"extinction law ({law.type()!r}) is of {low:.4g} to "
                        f"{high:.4g} um: leave the galaxy out of the galaxies "
                        f"of the observation (e.g. at mm wavelengths, where "
                        f"its dust and its light are negligible), or give "
                        f"its dust a law of those wavelengths")


def _observed_wavelengths(grid: Grid, view: View) -> np.ndarray | None:
    """
    Return the wavelengths (um, in vacuum) at which an observation sees
    light on the grid of an image (that of the band of its view) or of a
    spectral cube (the edges of its channels); None if they are not known
    (a band, or a spectral axis, without them).
    """
    if grid.spectral_axis is None:
        return None if view.band() is None else np.array(
            [view.band().wavelength().to_value(astropy.units.um)])
    axis = grid.spectral_axis
    convention = VelocityConvention.of(grid.coords)
    if convention.rest() is None:
        return None
    step = grid.coords.step[axis]
    edges = grid.zero()[axis] - step / 2 + step * np.arange(
        grid.size[axis] + 1)
    return scipy.constants.c / convention.frequency(edges) * 1e6
