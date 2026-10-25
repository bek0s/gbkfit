"""
How an observation treats the line of sight: it evaluates the galaxies it
sees on the plane of the sky (Flat), on a grid with a depth axis along
the line of sight (Grid), or by radiative transfer along it (Transfer).
It is explicit, and checked against every component it sees: never
inferred from them. Along the depth axis, the viewer is in front of its
first voxel.
"""

import abc
import numbers
from collections.abc import Sequence
from typing import Any, Self

import numpy as np

from gbkfit.driver import Driver
from gbkfit.model import (
    MAX_SUBCHANNELS, AdditiveScenePlan, DepthUnit, Galaxy, LineOpacity,
    Scene, ScenePlan, TransferScenePlan)
from gbkfit.model.components.base import Component, DepthAxis
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Depth',
    'Flat',
    'Grid',
    'LineOfSight',
    'Transfer',
    'line_of_sight_parser'
]


class Depth(parseutils.Serializable):
    """
    The depth axis of a grid, along the line of sight: centred on the
    plane of the sky.

    Parameters
    ----------
    extent : str or float, optional
        Its length (arcsec); 'galaxy' (the default): that which holds all
        the density of the disks the observation evaluates, at any
        inclination: twice the bounding radius of the largest (that of
        its last ring, R, or of a thick disk, sqrt(R^2 + Z^2) of the
        largest height of its density above the plane of the galaxy, Z;
        see Disk.bounding_radius); through radiative
        transfer, where each galaxy has a depth axis of its own, that of
        each galaxy; or 'legacy': that of the longer of the x and y axes
        of the grid the galaxies are evaluated on (the grid of the data,
        oversampled and padded for the convolutions), as the 3d models of
        earlier versions took it. A shorter axis than that of 'galaxy'
        (of 'legacy' or a length) loses the density beyond its ends: a
        warning says so (see parseutils.warn).
    step : float, optional
        The size of its voxels (arcsec); by default that of the pixels
        along the longer of the x and y axes of that grid.

    Raises
    ------
    ConfigError
        If the extent or the step is not valid.
    """

    def dump(self) -> dict[str, Any]:
        return dict(extent=self._extent, step=self._step)

    def __init__(self, extent: str | float = 'galaxy',
                 step: float | None = None):
        if extent not in ('galaxy', 'legacy') and not (
                isinstance(extent, numbers.Real) and extent > 0):
            raise ConfigError(
                f"extent must be 'galaxy', 'legacy' or a positive length; it "
                f"is {extent!r}")
        if step is not None and not step > 0:
            raise ConfigError(f"step must be positive; it is {step}")
        self._extent = extent
        self._step = step

    def extent(self) -> str | float:
        """Return its extent: a length, 'galaxy' or 'legacy'."""
        return self._extent

    def step(self) -> float | None:
        """Return the size of its voxels, if given."""
        return self._step

    def axis(
            self, grid: gridutils.Grid, bounding_radius: float
    ) -> DepthAxis:
        """
        Return the axis on the given grid of the x and y axes, for disks
        of the given bounding radius (arcsec; see Disk.bounding_radius):
        as many voxels as cover the extent, centred on 0.
        """
        longest = int(grid.size[0] < grid.size[1])
        step = self._step if self._step is not None \
            else grid.coords.step[longest]
        if self._extent == 'legacy':
            size = grid.size[longest]
        else:
            extent = 2 * bounding_radius if self._extent == 'galaxy' \
                else self._extent
            size = max(1, int(np.ceil(extent / step - 1e-9)))
        if size * step < 2 * bounding_radius * (1 - 1e-9):
            parseutils.warn(
                f"the depth axis ({size * step:g} arcsec) is shorter than "
                f"the disks need ({2 * bounding_radius:g} arcsec, twice "
                f"their bounding radius): their density beyond its ends is "
                f"lost; its extent 'galaxy' holds it")
        return DepthAxis(size, step, -(size / 2 - 0.5) * step)


depth_parser = parseutils.BasicParser(Depth)


class LineOfSight(parseutils.TypedSerializable, abc.ABC):
    """
    How an observation treats the line of sight: on which grid it
    evaluates the galaxies it sees, and what components it can see.
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    @abc.abstractmethod
    def depth_axis(
            self, grid: gridutils.Grid, bounding_radius: float
    ) -> DepthAxis | None:
        """
        Return the depth axis of the grid of the galaxies, on the given
        grid of the x and y axes, for disks of the given bounding radius
        (arcsec; see Depth.axis); none on the plane of the sky.
        """
        pass

    @abc.abstractmethod
    def check(self, component: Component) -> None:
        """
        Check that a component can be seen along this line of sight.

        Raises
        ------
        ConfigError
            If it cannot.
        """
        pass

    def check_order(
            self,
            galaxies: Sequence[Galaxy],
            order: Sequence[DepthUnit] | None
    ) -> None:
        """
        Check that this line of sight can order the given galaxies, which
        have opacity, along it (the light of one can be dimmed by
        another), in the given order, front to back (in units; see
        ObservationGroup: depth_order): here, it cannot.

        Raises
        ------
        ConfigError
            If it cannot.
        """
        raise ConfigError(
            "an observation of several galaxies with opacity needs a line "
            "of sight of radiative transfer (transfer), which orders them "
            "along it")

    def plan_scene(
            self,
            scene: Scene,
            driver: Driver,
            grid: gridutils.Grid,
            dtype: np.dtype,
            smoothing: float | None
    ) -> ScenePlan:
        """
        Plan the evaluation of a scene along this line of sight, on a driver
        and dtype, on the grid of an image (no spectral axis) or a spectral
        cube, for an instrument that smooths the spectra of the model by
        the given dispersion (km/s; None for an image), against which the
        resolution of the light is checked: here, each galaxy adds its
        light to the image or the spectral cube, dimmed by its own opacity,
        on the depth axis of the largest disk of the scene (see
        AdditiveScenePlan).
        """
        depth = self.depth_axis(grid.spatial(), max(scene.bounding_radii()))
        return AdditiveScenePlan(scene, driver, grid, depth, dtype, smoothing)


def _check_no_line_opacity(component: Component) -> None:
    """
    Raise ConfigError if the component has line opacity, which only a line
    of sight of radiative transfer sees.
    """
    if isinstance(component.opacity(), LineOpacity):
        raise ConfigError(
            "a component with line opacity needs a line of sight of "
            "radiative transfer (transfer)")


class Flat(LineOfSight):
    """
    No depth axis: the galaxies are evaluated on the plane of the sky. It
    sees thin components only (thin disks and points), and no opacity.
    """

    @staticmethod
    def type() -> str:
        return 'flat'

    def depth_axis(
            self, grid: gridutils.Grid, bounding_radius: float
    ) -> None:
        return None

    def check(self, component: Component) -> None:
        _check_no_line_opacity(component)
        if component.opacity() is not None:
            raise ConfigError(
                "a component with opacity needs a line of sight with a "
                "depth axis (grid or transfer)")
        if component.thick():
            raise ConfigError(
                "a thick component needs a line of sight with a depth axis "
                "(grid or transfer)")


class Grid(LineOfSight):
    """
    A depth axis: the galaxies are evaluated on the voxels of a grid,
    where dust dims the light behind it. It sees every component: thick
    disks fill its voxels, and the light of a thin disk or a point is at
    its depth (where the plane of the disk crosses each line of sight),
    dimmed by the dust in front of it.

    Parameters
    ----------
    depth : Depth, optional
        The depth axis; Depth() by default.
    """

    @staticmethod
    def type() -> str:
        return 'grid'

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        parseutils.load_option_and_update_info(depth_parser, info, 'depth')
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type(), depth=self._depth.dump())

    def __init__(self, depth: Depth | None = None):
        self._depth = depth if depth is not None else Depth()

    def depth(self) -> Depth:
        """Return the depth axis."""
        return self._depth

    def depth_axis(
            self, grid: gridutils.Grid, bounding_radius: float
    ) -> DepthAxis:
        return self._depth.axis(grid, bounding_radius)

    def check(self, component: Component) -> None:
        _check_no_line_opacity(component)


class Transfer(LineOfSight):
    """
    A depth axis and radiative transfer along each line of sight: the
    galaxies are evaluated on the voxels of a grid, as by Grid, but their
    emitters write the fields of their light (its density, velocity and
    dispersion in each voxel, or in the layer of a thin disk or a point;
    see ComponentPlan.evaluate_fields), and a march along each line of
    sight, front to back, adds the light of each channel that reaches the
    viewer, dimmed by the opacity in front of it. It sees every component.
    It gives what Grid gives, but for the clouds of a Monte Carlo disk:
    those of each voxel become one line, of their mean velocity, and as
    wide as the sum of their lines. Its components may have line opacity
    (see LineOpacity).

    It sees several galaxies with opacity, which can dim the light of one
    another: the march goes through one after the other, front to back,
    in their order along the line of sight (see ObservationGroup:
    depth_order), each on a depth axis of its own, centred on its plane.

    Parameters
    ----------
    depth : Depth, optional
        The depth axis; Depth() by default.
    slab : int, optional
        The number of planes of the depth axis evaluated at once, which
        bounds the memory of the evaluation (the fields of the emitters
        and the cubes of opacity of a slab of planes); all by default
        (fastest). The clouds of Monte Carlo disks are drawn again for
        each slab.
    subchannels : int, optional
        With line opacity, the number of sub-channels of each channel
        that the march transfers the light at: the light of a channel is
        their mean (from 1, the default, to 64). Line opacity averaged
        over a channel wider than its lines dims their light too little;
        sub-channels resolve the lines, at their number times the time of
        the march (light without line opacity needs none).

    Raises
    ------
    ConfigError
        If slab is not a positive integer, or subchannels is not an
        integer from 1 to 64.
    """

    @staticmethod
    def type() -> str:
        return 'transfer'

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        parseutils.load_option_and_update_info(depth_parser, info, 'depth')
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), depth=self._depth.dump(), slab=self._slab,
            subchannels=self._subchannels)

    def __init__(
            self,
            depth: Depth | None = None,
            slab: int | None = None,
            subchannels: int = 1
    ):
        if slab is not None and not (
                isinstance(slab, numbers.Integral)
                and not isinstance(slab, bool) and slab > 0):
            raise ConfigError(
                f"slab must be a positive integer; it is {slab!r}")
        if not (isinstance(subchannels, numbers.Integral)
                and not isinstance(subchannels, bool)
                and 1 <= subchannels <= MAX_SUBCHANNELS):
            raise ConfigError(
                f"subchannels must be an integer from 1 to {MAX_SUBCHANNELS}; "
                f"it is {subchannels!r}")
        self._depth = depth if depth is not None else Depth()
        self._slab = slab
        self._subchannels = subchannels

    def depth(self) -> Depth:
        """Return the depth axis."""
        return self._depth

    def slab(self) -> int | None:
        """Return the number of planes evaluated at once, if given."""
        return self._slab

    def subchannels(self) -> int:
        """Return the number of sub-channels of each channel."""
        return self._subchannels

    def depth_axis(
            self, grid: gridutils.Grid, bounding_radius: float
    ) -> DepthAxis:
        return self._depth.axis(grid, bounding_radius)

    def check(self, component: Component) -> None:
        pass

    def check_order(
            self,
            galaxies: Sequence[Galaxy],
            order: Sequence[DepthUnit] | None
    ) -> None:
        ordered = {g for unit in order or () for g in unit.galaxies}
        if missing := [
                galaxy.name() for g, galaxy in enumerate(galaxies)
                if g not in ordered]:
            raise ConfigError(
                f"radiative transfer marches several galaxies with opacity "
                f"one after the other, front to back: the depth_order of the "
                f"observation group must give the place of the galaxies "
                f"{missing}")

    def plan_scene(
            self,
            scene: Scene,
            driver: Driver,
            grid: gridutils.Grid,
            dtype: np.dtype,
            smoothing: float | None
    ) -> TransferScenePlan:
        # (each galaxy on the depth axis of its largest disk)
        depths = [
            self.depth_axis(grid.spatial(), radius)
            for radius in scene.bounding_radii()]
        return TransferScenePlan(
            scene, driver, grid, depths, dtype, self._slab, self._subchannels,
            smoothing)


line_of_sight_parser = parseutils.TypedParser(LineOfSight, [
    Flat,
    Grid,
    Transfer])
