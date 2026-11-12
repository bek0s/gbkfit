"""
Pixel weights of the data of an observation (its option pixel_weights):
functions of the position of each pixel in the plane of a disk of a
galaxy it sees (e.g. less weight near the minor axis), evaluated with
the current parameters of the disk, so that free angles are followed,
and normalised by the comparison of the data (see Weights).
"""

import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias

import numpy as np
from annotated_types import Ge, Gt, Le

from gbkfit.config import (
    BasicParser, Choices, Serializable, TypedParser, TypedSerializable,
    Unit, check_options)
from gbkfit.driver import DeviceArray, Driver
from gbkfit.grid import Grid, sky_positions
from gbkfit.params import ParamValues, StageCache
from gbkfit.utils import iterutils

if TYPE_CHECKING:
    from gbkfit.model import Scene


__all__ = [
    'WeightAzimuthPower',
    'WeightAzimuthRange',
    'WeightTrait',
    'Weights',
    'weight_trait_parser',
    'weights_parser'
]


# The axes of a disk, and the azimuth of each (radians, in its plane)
_AXES = dict(major=0.0, minor=np.pi / 2)
_Axis: TypeAlias = Annotated[
    Literal[tuple(_AXES)], Choices(major='Major axis', minor='Minor axis')]


class WeightTrait(TypedSerializable, abc.ABC):
    """
    A weight function of the azimuth of a point in the plane of a disk,
    measured from its major axis (as that of the polar traits).
    """

    @abc.abstractmethod
    def weight(self, theta: np.ndarray) -> np.ndarray:
        """Return the weight at each azimuth (radians)."""


class WeightAzimuthPower(WeightTrait):
    """
    Less weight near an axis of the disk: |cos(theta)|^power near the
    minor axis, |sin(theta)|^power near the major axis.

    Parameters
    ----------
    power : float
        The power (positive).
    axis : str, optional
        The axis near which the weight is less: 'minor' (the default) or
        'major'.
    """

    @staticmethod
    def type() -> str:
        return 'azimuth_power'

    @staticmethod
    def type_title() -> str:
        return 'Azimuth power'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type(), power=self._power, axis=self._axis)

    def __init__(self, power: Annotated[float, Gt(0)], axis: _Axis = 'minor'):
        check_options(WeightAzimuthPower.__init__, locals())
        self._power = power
        self._axis = axis

    def weight(self, theta: np.ndarray) -> np.ndarray:
        # (the cosine is 0 on the minor axis, the sine on the major axis)
        along = np.cos(theta) if self._axis == 'minor' else np.sin(theta)
        return np.abs(along) ** self._power


class WeightAzimuthRange(WeightTrait):
    """
    A weight within an angle of an axis of the disk (on both sides of the
    centre), and 1 elsewhere: e.g. none near the minor axis.

    Parameters
    ----------
    half_width : float
        The angle from the axis (degrees, in the plane of the disk).
    axis : str, optional
        The axis: 'minor' (the default) or 'major'.
    weight : float, optional
        The weight within the angle; 0 by default.
    """

    @staticmethod
    def type() -> str:
        return 'azimuth_range'

    @staticmethod
    def type_title() -> str:
        return 'Azimuth range'

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), half_width=self._half_width, axis=self._axis,
            weight=self._weight)

    def __init__(
            self,
            half_width: Annotated[float, Gt(0), Le(90), Unit('deg')],
            axis: _Axis = 'minor',
            weight: Annotated[float, Ge(0)] = 0
    ):
        check_options(WeightAzimuthRange.__init__, locals())
        self._half_width = half_width
        self._axis = axis
        self._weight = weight

    def weight(self, theta: np.ndarray) -> np.ndarray:
        # The angle from the axis, on either side of the centre
        angle = np.abs(np.arcsin(np.sin(theta - _AXES[self._axis])))
        return np.where(
            angle <= np.radians(self._half_width), self._weight, 1.0)


weight_trait_parser = TypedParser(WeightTrait, [
    WeightAzimuthPower,
    WeightAzimuthRange], family='weight_trait')


class Weights(Serializable):
    """
    Spatial weights of the data of an observation: the product of weight
    functions of the azimuth of each pixel in the plane of a disk of a
    galaxy that the observation sees. The pixels are deprojected through
    the geometry of the disk with its current parameters, so free angles
    are followed; a warped disk is deprojected at its mean plane (the mean
    of its node-wise parameters). The comparison of the data normalises
    them to a mean square of 1 over the values it compares, so that only
    their shape counts: a fit cannot lower its chi-square by turning the
    disk so that the data fall where the weights are low (see
    ComparisonGauss).

    Parameters
    ----------
    component : str
        The name of the disk ('galaxy.component' when the observation sees
        several galaxies).
    traits : WeightTrait or Sequence of WeightTrait
        The weight functions.

    Raises
    ------
    ValueError
        If there are no weight functions.
    """

    def dump(self) -> dict[str, Any]:
        return dict(
            component=self._component,
            traits=weight_trait_parser.dump(list(self._traits)))

    def __init__(
            self,
            component: str,
            traits: WeightTrait | Sequence[WeightTrait]
    ):
        check_options(Weights.__init__, locals())
        traits = iterutils.tuplify(traits)
        if not traits:
            raise ValueError(
                "pixel weights need at least one weight function")
        self._component = component
        self._traits = traits

    def component(self) -> str:
        """Return the name of the disk whose plane the pixels are in."""
        return self._component

    def traits(self) -> tuple[WeightTrait, ...]:
        """Return the weight functions."""
        return self._traits

    def weight(
            self,
            geometry: dict[str, float],
            x: np.ndarray,
            y: np.ndarray
    ) -> np.ndarray:
        """
        Return the weight of points on the sky (arcsec, in the frame of the
        model; see sky_positions), in the plane of a disk of the given
        geometry (xpos, ypos, posa and incl), as the disk kernels place
        them.

        Parameters
        ----------
        geometry : dict
            The geometry of the disk: xpos, ypos (arcsec), posa and incl
            (degrees).
        x, y : np.ndarray
            The positions of the points (arcsec).

        Returns
        -------
        np.ndarray
            The weight of each point, of their shape.
        """
        posa = np.radians(geometry['posa'])
        incl = np.radians(geometry['incl'])
        x = x - geometry['xpos']
        y = y - geometry['ypos']
        xd = -x * np.sin(posa) + y * np.cos(posa)
        yd = (-x * np.cos(posa) - y * np.sin(posa)) / np.cos(incl)
        theta = np.arctan2(yd, xd)
        weight = np.ones_like(theta)
        for trait in self._traits:
            weight *= trait.weight(theta)
        return weight


weights_parser = BasicParser(Weights)


class WeightsPlan:
    """
    The evaluation of the weights of an observation on the spatial grid of
    its model: into a weight cube, the same in every channel, with the
    parameters of the geometry of the disk (their names in the parameters
    of the observation group, by their names in the disk). The weights are
    worked out again only for evaluations at which the values of those
    parameters change (e.g. not in a fit of a rotation curve alone).
    """

    def __init__(
            self,
            weights: Weights,
            geometry: dict[str, str],
            grid: Grid,
            driver: Driver,
            dtype: np.dtype
    ):
        self._weights = weights
        self._geometry = geometry
        self._x, self._y = sky_positions(grid)
        self._driver = driver
        self._dtype = dtype
        self._weight_d = driver.mem_alloc_d(self._x.shape, dtype)
        self._weight_cache = StageCache(geometry.values())

    def evaluate(self, params: ParamValues, wcube: DeviceArray) -> None:
        """Write the weights of the pixels to every channel of wcube."""
        self._weight_cache.evaluate(params, self._weigh)
        # (wcube is written each time: an evaluation can smooth it in
        # place, see ModelCube: smooth_weights)
        self._driver.mem_fill(wcube, 1)
        self._driver.math_mul(wcube, self._weight_d[None], out=wcube)

    def _weigh(self, params: ParamValues) -> None:
        """
        Work out the weights of the pixels with the values of the
        parameters of the geometry, and copy them to the device.
        """
        geometry = {
            name: float(np.mean(params[full]))
            for name, full in self._geometry.items()}
        weight = self._weights.weight(geometry, self._x, self._y)
        self._driver.mem_copy_h2d(weight.astype(self._dtype), self._weight_d)


def plan_weights(
        weights: Weights,
        scene: 'Scene',
        grid: Grid,
        driver: Driver,
        dtype: np.dtype
) -> WeightsPlan:
    """
    Plan the evaluation of the weights of an observation of a scene on the
    spatial grid of its model (see weights_geometry).
    """
    return WeightsPlan(
        weights, weights_geometry(weights, scene), grid, driver, dtype)


def weights_geometry(weights: Weights, scene: 'Scene') -> dict[str, str]:
    """
    Return the parameters of the plane of the disk of pixel weights in a
    scene, by role (xpos, ypos, posa, incl). Raise ValueError unless
    their component is a disk of the scene.
    """
    frame = scene.frame_params(weights.component())
    names = dict(xpos=frame.x, ypos=frame.y, posa=frame.posa, incl=frame.incl)
    if missing := [role for role, name in names.items() if name is None]:
        raise ValueError(
            f"pixel weights need the plane of a disk; the component "
            f"{weights.component()!r} has no {missing}")
    return names
