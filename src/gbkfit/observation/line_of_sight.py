"""
How an observation treats the line of sight: it evaluates the galaxies it
sees on the plane of the sky (Flat), or on a grid with a depth axis along
the line of sight (Grid). It is explicit, and checked against every
component it sees: never inferred from them. Along the depth axis, the
viewer is in front of its first voxel.
"""

import abc
import numbers
from typing import Any, Self

import numpy as np

from gbkfit.model.components.base import Component, DepthAxis
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Depth',
    'Flat',
    'Grid',
    'LineOfSight',
    'line_of_sight_parser'
]


class Depth(parseutils.Serializable):
    """
    The depth axis of a grid, along the line of sight: centred on the
    plane of the sky.

    Parameters
    ----------
    extent : str or float, optional
        Its length (arcsec), or 'legacy' (the default): that of the
        longer of the x and y axes of the grid the galaxies are evaluated
        on (the grid of the data, oversampled and padded for the
        convolutions), as the 3d models of earlier versions took it.
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

    def __init__(self, extent: str | float = 'legacy',
                 step: float | None = None):
        if extent != 'legacy' and not (
                isinstance(extent, numbers.Real) and extent > 0):
            raise ConfigError(
                f"extent must be 'legacy' or a positive length; it is "
                f"{extent!r}")
        if step is not None and not step > 0:
            raise ConfigError(f"step must be positive; it is {step}")
        self._extent = extent
        self._step = step

    def extent(self) -> str | float:
        """Return its extent: a length, or 'legacy'."""
        return self._extent

    def step(self) -> float | None:
        """Return the size of its voxels, if given."""
        return self._step

    def axis(self, grid: gridutils.Grid) -> DepthAxis:
        """
        Return the axis on the given grid of the x and y axes: as many
        voxels as cover the extent, centred on 0.
        """
        longest = int(grid.size[0] < grid.size[1])
        step = self._step if self._step is not None \
            else grid.coords.step[longest]
        if self._extent == 'legacy':
            size = grid.size[longest]
        else:
            size = max(1, int(np.ceil(self._extent / step - 1e-9)))
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
    def depth_axis(self, grid: gridutils.Grid) -> DepthAxis | None:
        """
        Return the depth axis of the grid of the galaxies, on the given
        grid of the x and y axes; none on the plane of the sky.
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


class Flat(LineOfSight):
    """
    No depth axis: the galaxies are evaluated on the plane of the sky. It
    sees thin components only (thin disks and points), and no opacity.
    """

    @staticmethod
    def type() -> str:
        return 'flat'

    def depth_axis(self, grid: gridutils.Grid) -> None:
        return None

    def check(self, component: Component) -> None:
        if component.opacity() is not None:
            raise ConfigError(
                "a component with opacity needs a line of sight with a "
                "depth axis (grid)")
        if component.thick():
            raise ConfigError(
                "a thick component needs a line of sight with a depth axis "
                "(grid)")


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

    def depth_axis(self, grid: gridutils.Grid) -> DepthAxis:
        return self._depth.axis(grid)

    def check(self, component: Component) -> None:
        pass


line_of_sight_parser = parseutils.TypedParser(LineOfSight, [
    Flat,
    Grid])
