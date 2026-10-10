import abc
import typing
from typing import Any

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamDesc
from gbkfit.utils import gridutils, parseutils


__all__ = [
    'Model',
    'ModelImage',
    'ModelPlan',
    'ModelSCube',
    'Selection',
    'model_parser'
]


class Selection(typing.NamedTuple):
    """
    What of a model an observation sees: its components and the emission
    lines of its components of the given names (all if None). Opacity
    components always absorb; the components without lines have one,
    which is always seen.
    """
    components: tuple[str, ...] | None = None
    lines: tuple[str, ...] | None = None


class Model(parseutils.TypedSerializable, abc.ABC):
    """
    A model of a galaxy: what is on the sky. It can have a name, which
    then prefixes its parameters instead of its position when there are
    several models (see ObservationGroup).

    Parameters
    ----------
    name : str, optional
        Its name.

    Raises
    ------
    ConfigError
        If the name is not valid (see parseutils.check_name).
    """

    def __init__(self, name: str | None):
        parseutils.check_name(name)
        self._name = name

    def name(self) -> str | None:
        """Return its name, if any."""
        return self._name

    @abc.abstractmethod
    def pdescs(self) -> dict[str, ParamDesc]:
        """Return its parameters, by name."""
        pass

    @abc.abstractmethod
    def has_weights(self) -> bool:
        """Check whether it has spatial weights (weight traits)."""
        pass

    def constants(self) -> dict[str, Any]:
        """
        Return the values that parameter expressions can use (e.g. the
        radial nodes of a disk), by name; none here.
        """
        return {}

    @abc.abstractmethod
    def plan(
            self,
            driver: Driver,
            grid: gridutils.Grid,
            has_weights: bool,
            dtype: np.dtype,
            selection: Selection = Selection()
    ) -> 'ModelPlan':
        """
        Plan the evaluation of the model.

        Parameters
        ----------
        driver : Driver
            The driver it is evaluated on.
        grid : Grid
            The grid of its data: x and y, and the spectral axis of
            spectral cubes.
        has_weights : bool
            Whether to evaluate its spatial weights.
        dtype : np.dtype
            The floating type of the evaluation.
        selection : Selection, optional
            What of the model is evaluated.

        Returns
        -------
        ModelPlan
            The plan.
        """
        pass


class ModelPlan(abc.ABC):
    """The evaluation of a model on a driver, grid and dtype."""

    @abc.abstractmethod
    def evaluate(
            self,
            params: dict[str, float | np.ndarray],
            data: DeviceArray,
            weights: DeviceArray | None,
            out_extra: dict[str, Any] | None
    ) -> None:
        """
        Evaluate the model.

        Parameters
        ----------
        params : dict
            The values of its parameters, by name.
        data : DeviceArray
            The image or spectral cube on the grid of the plan, which the
            model is added to.
        weights : DeviceArray, optional
            The weights of the data, which its spatial weights multiply.
        out_extra : dict, optional
            Where its extra outputs go, if wanted.
        """
        pass


class ModelImage(Model, abc.ABC):
    """A model evaluated into images."""


class ModelSCube(Model, abc.ABC):
    """A model evaluated into spectral cubes."""


model_parser = parseutils.TypedParser(Model)
