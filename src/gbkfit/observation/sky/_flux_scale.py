"""
The plan of an observation on the sky whose flux scale has several
values (see ObservationSky): its items of light times the values.
"""

from typing import TYPE_CHECKING

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.model import Extras, SceneState
from gbkfit.params import ParamValues
from ..plan import FluxScale, HostModelData, ModelData, ObservationPlan

if TYPE_CHECKING:
    from ..noise import Noise


class FluxScaledPlan(ObservationPlan):
    """
    The plan of an observation times a flux scale of several values (see
    ObservationSky): its items of light are times the values, each along
    the axis of the scale (one value at each point of the other axes).
    """

    def __init__(
            self,
            plan: ObservationPlan,
            driver: Driver,
            param: str,
            scale: FluxScale,
            dtype: np.dtype
    ):
        self._plan = plan
        self._driver = driver
        self._param = param
        self._scale = scale
        self._dtype = dtype
        # The values, of the shape of the items but 1 along the axis, on
        # the host (h) and the device (d), allocated on first use
        self._h_values: np.ndarray | None = None
        self._d_values: DeviceArray | None = None

    def evaluate(
            self,
            params: ParamValues,
            state: SceneState,
            out_extra: Extras | None
    ) -> ModelData:
        model = self._plan.evaluate(params, state, out_extra)
        driver = self._driver
        if self._h_values is None:
            shape = model[self._scale.items[0]].data.shape
            (self._h_values, self._d_values) = driver.mem_alloc_s(
                tuple(1 if k == self._scale.axis else n
                      for k, n in enumerate(shape)), self._dtype)
        self._h_values[...] = np.reshape(
            params[self._param], self._h_values.shape)
        driver.mem_copy_h2d(self._h_values, self._d_values)
        for key in self._scale.items:
            values = model[key].data
            driver.math_mul(values, self._d_values, out=values)
        return model

    def mock(
            self,
            params: ParamValues,
            state: SceneState,
            noise: 'Noise',
            rng: np.random.Generator,
            ndraws: int
    ) -> tuple[list[HostModelData], dict[str, np.ndarray] | None]:
        """
        Draw mock data (see ObservationPlan.mock), of scales of 1 only:
        the noise is drawn where the observation records the light, which
        a scale of each spectrum does not reach.

        Raises
        ------
        ValueError
            If a value of the scale is not 1.
        """
        if np.any(params[self._param] != 1):
            raise ValueError(
                f"mock data take a flux scale per spectrum or per row of 1 "
                f"only; '{self._param}' is not")
        return self._plan.mock(params, state, noise, rng, ndraws)

    def has_exact_errors(self) -> bool:
        return self._plan.has_exact_errors()
