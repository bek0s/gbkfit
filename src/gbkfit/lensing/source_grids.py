"""
The grids that an observation draws the planes behind lenses on (see
SourceGrid): one for each plane whose light is bent, laid out from the
rays of the pixels of its data that cross it (see TracePlan).
"""

import math
from collections.abc import Sequence
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
from annotated_types import Gt

from gbkfit.config import Choices, Serializable, Unit, check_options
from gbkfit.grid import Grid, make_grid


__all__ = [
    'MARGIN',
    'MAX_STRETCH',
    'MIN_MARGIN_STEPS',
    'STRETCH_QUANTILE',
    'SourceGrid',
    'source_step'
]


# The most times finer than the pixels of the image that the pixels of a
# grid of a plane behind lenses are (see source_step): the stretch of the
# image is infinite on its critical curves, and this bounds the grid
# there. Measured on a thick exponential disk at z = 2.26, in rotation of
# 180 km/s, behind the deflection maps of the group lens J0901 (60 x 60
# pixels of 0.1 arcsec, 81 channels of 10 km/s, no PSF or LSF; a footprint
# where half the rays are stretched more than 4 times), against a step of
# 1/20 of the pixels: the first moment of the image cube is off by an rms
# of 0.57, 0.14 and 0.030 km/s (at most 3.0, 0.69 and 0.15) at 1/2.5, 1/5
# and 1/10 of the pixels, as the square of the step; so about 0.05 (at
# most 0.25) at 1/8, well within the errors of the first moments of data
# (a few km/s).
MAX_STRETCH = 8.0

# The margin of the box of the rays of a plane, on each side, of an
# automatic extent (see SourceGrid): a share of the width of the box, and
# at least a number of pixels; room for the rays to move as the lenses do
# in a fit, away from the reference values the grid is laid out at
MARGIN = 0.2
MIN_MARGIN_STEPS = 2

# The quantile of the stretch of the image at the rays of a plane that
# its automatic step resolves (see source_step). The stretch is infinite
# on the critical curves, which cross the footprint of every strong lens:
# its maximum is that of the pixel nearest a curve, and would always give
# the finest step. The curves are lines, of a share of the pixels of the
# order of one over their number across; the arcs, whose light holds the
# kinematics, are strongly magnified over much more of the footprint
# (e.g. |1 - theta_e / r| < 1/4 for r from 0.8 to 1.33 theta_e about an
# isothermal sphere). The 95th percentile leaves out the few pixels
# nearest the curves and resolves the arcs.
STRETCH_QUANTILE = 0.95

# An automatic step or extent
_Auto: TypeAlias = Annotated[Literal['auto'], Choices(
    auto='From the rays of the data')]

# A length on the sky
_Length: TypeAlias = Annotated[float, Gt(0), Unit('arcsec')]


class SourceGrid(Serializable):
    """
    The grids that an observation draws the planes behind lenses on, one
    for each plane whose light is bent (see LensedScene), fixed for all
    its evaluations: laid out at reference values of the parameters (e.g.
    the start of a fit; see TracePlan), aligned with the sky, from the
    rays of the pixels of its data that cross the plane (its footprint
    there).

    Parameters
    ----------
    step : 'auto' or float, optional
        The size of the pixels (arcsec), or 'auto' (the default): that of
        the pixels of the image (of the high-resolution grid), over how
        much the image stretches the plane at the rays (its
        STRETCH_QUANTILE quantile; see source_step), at most MAX_STRETCH
        times finer.
    extent : 'auto' or Sequence of float, optional
        The span of the centres of the pixels, (x0, x1, y0, y1) (arcsec,
        in the frame of the observation, +x to the west and +y to the
        north, as xpos and ypos), the same on every plane; or 'auto' (the
        default): the box of the rays of each plane, with a margin of
        MARGIN of its width on each side, of at least MIN_MARGIN_STEPS
        pixels. Light that rays meet beyond the grid is lost, which the
        plan warns of (see LensedScenePlan).

    Raises
    ------
    ValueError
        If the extent is not four values of increasing spans.
    """

    config_titles = dict(step='Pixel size', extent='Extent')

    def dump(self) -> dict[str, Any]:
        extent = self._extent if self._extent == 'auto' \
            else list(self._extent)
        return dict(step=self._step, extent=extent)

    def __init__(
            self,
            step: _Auto | _Length = 'auto',
            extent: _Auto | Annotated[Sequence[float], Unit('arcsec')] = 'auto'
    ):
        check_options(SourceGrid.__init__, locals())
        if extent != 'auto':
            extent = tuple(float(value) for value in extent)
            if len(extent) != 4 or not (
                    extent[0] < extent[1] and extent[2] < extent[3]):
                raise ValueError(
                    f"the extent of a source grid is (x0, x1, y0, y1), with "
                    f"x0 < x1 and y0 < y1; it is {extent}")
        self._step = step
        self._extent = extent

    def step(self) -> float | Literal['auto']:
        """Return the size of its pixels, or 'auto'."""
        return self._step

    def extent(self) -> tuple[float, float, float, float] | Literal['auto']:
        """Return the span of the centres of its pixels, or 'auto'."""
        return self._extent

    def layout(
            self,
            x: np.ndarray,
            y: np.ndarray,
            step: float,
            rval: Sequence[float]
    ) -> Grid:
        """
        Return the grid of a plane (its x and y axes), of the rays that
        cross it, of the size of its pixels: of its extent, or of the box
        of the rays with its margin, centred there, aligned with the sky.

        Parameters
        ----------
        x, y : np.ndarray
            The positions of the rays on the plane (arcsec, in the frame of
            the observation).
        step : float
            The size of the pixels (arcsec; see step and source_step).
        rval : Sequence of float
            The RA and Dec (degrees) of the origin of the frame of the
            observation (its reference pixel).

        Returns
        -------
        Grid
            The grid.
        """
        if self._extent == 'auto':
            spans = []
            for values in (x, y):
                low, high = float(np.min(values)), float(np.max(values))
                margin = max(MARGIN * (high - low), MIN_MARGIN_STEPS * step)
                spans += [low - margin, high + margin]
        else:
            spans = list(self._extent)
        sizes, firsts = [], []
        for low, high in (spans[:2], spans[2:]):
            size = math.ceil((high - low) / step - 1e-9) + 1
            sizes.append(size)
            firsts.append((low + high) / 2 - (size - 1) / 2 * step)
        rpix = tuple(-first / step for first in firsts)
        return make_grid(tuple(sizes), step, rpix, tuple(rval[:2]), 0)


def source_step(substep: float, stretch: np.ndarray) -> float:
    """
    Return the automatic size of the pixels of the grid of a plane (see
    SourceGrid): that of the pixels of the image (substep) over how much
    the image stretches the plane where its rays cross it (see stretch,
    of Tracer.jacobian; its STRETCH_QUANTILE quantile), from 1 (no finer
    than the image) to MAX_STRETCH; so that the stretched light is as
    well sampled on the plane as the image is.

    Parameters
    ----------
    substep : float
        The size of the pixels of the image (arcsec; of the
        high-resolution grid, the smaller of x and y).
    stretch : np.ndarray
        The stretch at the rays (infinite on critical curves).

    Returns
    -------
    float
        The size of the pixels (arcsec).
    """
    # (a stretch of the rays, not between two: infinite ones interpolate
    # to NaN)
    high = float(np.quantile(stretch, STRETCH_QUANTILE, method='higher'))
    return substep / min(max(high, 1.0), MAX_STRETCH)
