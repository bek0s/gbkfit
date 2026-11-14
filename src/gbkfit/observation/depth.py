"""
The depth axis of the observations on the sky, along the line of sight
(Depth), and the evaluation of what they see along it, which the
components they see decide (see Scene.evaluation of gbkfit.model): on the
plane of the sky, on a depth axis, or by radiative transfer along it.
Along the depth axis, the viewer is in front of its first voxel.
"""

from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
from annotated_types import Ge, Gt, Le

from gbkfit.config import Choices, Serializable, Unit, check_options, warn
from gbkfit.driver import Driver
from gbkfit.grid import Grid
from gbkfit.model import (
    MAX_SUBCHANNELS, AdditiveScenePlan, Scene, SceneEvaluation, ScenePlan,
    TransferScenePlan)
from gbkfit.model.components.base import DepthAxis


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


def plan_scene(
        scene: Scene,
        depth: Depth,
        driver: Driver,
        grid: Grid,
        dtype: np.dtype,
        smoothing: float | None
) -> ScenePlan:
    """
    Plan the evaluation of a scene as it needs (see Scene.evaluation), on
    a driver and dtype, on the grid of an image (no spectral axis) or a
    spectral cube, for an instrument that smooths the spectra of the model
    by the given dispersion (km/s; None for an image), against which the
    resolution of the light is checked:

    - on the plane of the sky, or on the depth axis of the scene, for the
      largest of its galaxies (see Depth.axis): each galaxy adds its light
      to the image or the spectral cube, dimmed by its own opacity (see
      AdditiveScenePlan);
    - by radiative transfer, each galaxy on a depth axis of its own (see
      TransferScenePlan).

    Parameters
    ----------
    scene : Scene
        The scene.
    depth : Depth
        The depth axis of the observation, if the scene needs one.
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
    spatial = grid.spatial()
    radii = scene.bounding_radii()
    match scene.evaluation():
        case SceneEvaluation.PLANE:
            return AdditiveScenePlan(
                scene, driver, grid, None, dtype, smoothing)
        case SceneEvaluation.DEPTH:
            return AdditiveScenePlan(
                scene, driver, grid, depth.axis(spatial, max(radii)), dtype,
                smoothing)
        case SceneEvaluation.TRANSFER:
            return TransferScenePlan(
                scene, driver, grid,
                [depth.axis(spatial, radius) for radius in radii], dtype,
                depth.slab(), depth.subchannels(), smoothing)
