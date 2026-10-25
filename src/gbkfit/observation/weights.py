"""
Pixel weights of the data of an observation (its option pixel_weights):
functions of the position of each pixel in the plane of a disk of a
galaxy it sees (e.g. less weight near the minor axis), evaluated with
the current parameters of the disk, so that free angles are followed.
"""

import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamValues
from gbkfit.utils import gridutils, iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError

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


def _check_axis(axis: str) -> None:
    """Raise ConfigError unless axis is one of the axes of a disk."""
    if axis not in _AXES:
        raise ConfigError(
            f"axis must be one of {list(_AXES)}; it is {axis!r}")


class WeightTrait(parseutils.TypedSerializable, abc.ABC):
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

    Raises
    ------
    ConfigError
        If the power is not positive, or the axis is not one of those.
    """

    @staticmethod
    def type() -> str:
        return 'azimuth_power'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type(), power=self._power, axis=self._axis)

    def __init__(self, power: float, axis: str = 'minor'):
        if not power > 0:
            raise ConfigError(f"power must be positive; it is {power}")
        _check_axis(axis)
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

    Raises
    ------
    ConfigError
        If the angle is not between 0 and 90, the weight is negative, or
        the axis is not one of those.
    """

    @staticmethod
    def type() -> str:
        return 'azimuth_range'

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), half_width=self._half_width, axis=self._axis,
            weight=self._weight)

    def __init__(
            self, half_width: float, axis: str = 'minor', weight: float = 0):
        if not 0 < half_width <= 90:
            raise ConfigError(
                f"half_width must be above 0 and at most 90; it is "
                f"{half_width}")
        if not weight >= 0:
            raise ConfigError(f"weight must not be negative; it is {weight}")
        _check_axis(axis)
        self._half_width = half_width
        self._axis = axis
        self._weight = weight

    def weight(self, theta: np.ndarray) -> np.ndarray:
        # The angle from the axis, on either side of the centre
        angle = np.abs(np.arcsin(np.sin(theta - _AXES[self._axis])))
        return np.where(
            angle <= np.radians(self._half_width), self._weight, 1.0)


weight_trait_parser = parseutils.TypedParser(WeightTrait, [
    WeightAzimuthPower,
    WeightAzimuthRange])


class Weights(parseutils.Serializable):
    """
    Spatial weights of the data of an observation: the product of weight
    functions of the azimuth of each pixel in the plane of a disk of a
    galaxy that the observation sees. The pixels are deprojected through
    the geometry of the disk with its current parameters, so free angles
    are followed; a warped disk is deprojected at its mean plane (the mean
    of its node-wise parameters).

    Parameters
    ----------
    component : str
        The name of the disk ('galaxy.component' when the observation sees
        several galaxies).
    traits : WeightTrait or Sequence of WeightTrait
        The weight functions.

    Raises
    ------
    ConfigError
        If there are no weight functions.
    """

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        parseutils.load_option_and_update_info(
            weight_trait_parser, info, 'traits', required=True)
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        return dict(
            component=self._component,
            traits=weight_trait_parser.dump(list(self._traits)))

    def __init__(
            self,
            component: str,
            traits: WeightTrait | Sequence[WeightTrait]
    ):
        traits = iterutils.tuplify(traits)
        if not traits:
            raise ConfigError(
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
        model; see gridutils.sky_positions), in the plane of a disk of the
        given geometry (xpos, ypos, posa and incl), as the disk kernels
        place them.
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


weights_parser = parseutils.BasicParser(Weights)


class WeightsPlan:
    """
    The evaluation of the weights of an observation on the spatial grid of
    its model: into a weight cube, the same in every channel, with the
    parameters of the geometry of the disk (their names in the parameters
    of the observation group, by their names in the disk).
    """

    def __init__(
            self,
            weights: Weights,
            geometry: dict[str, str],
            grid: gridutils.Grid,
            driver: Driver,
            dtype: np.dtype
    ):
        self._weights = weights
        self._geometry = geometry
        self._x, self._y = gridutils.sky_positions(grid)
        self._driver = driver
        self._dtype = dtype
        self._weight_d = driver.mem_alloc_d(self._x.shape, dtype)

    def evaluate(self, params: ParamValues, wcube: DeviceArray) -> None:
        """Write the weights of the pixels to every channel of wcube."""
        geometry = {
            name: float(np.mean(params[full]))
            for name, full in self._geometry.items()}
        weight = self._weights.weight(geometry, self._x, self._y)
        self._driver.mem_copy_h2d(weight.astype(self._dtype), self._weight_d)
        self._driver.mem_fill(wcube, 1)
        self._driver.math_mul(wcube, self._weight_d[None], out=wcube)


def plan_weights(
        weights: Weights,
        scene: 'Scene',
        grid: gridutils.Grid,
        driver: Driver,
        dtype: np.dtype
) -> WeightsPlan:
    """
    Plan the evaluation of the weights of an observation of a scene on the
    spatial grid of its model. Raise ConfigError unless their component is
    a disk of the scene.
    """
    frame = scene.frame_params(weights.component())
    names = dict(xpos=frame.x, ypos=frame.y, posa=frame.posa, incl=frame.incl)
    if missing := [role for role, name in names.items() if name is None]:
        raise ConfigError(
            f"pixel weights need the plane of a disk; the component "
            f"{weights.component()!r} has no {missing}")
    return WeightsPlan(weights, names, grid, driver, dtype)
