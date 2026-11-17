"""
The plan of an observation: the evaluation of its model data (see
ObservationPlan), the model of each of its data items (see ModelItem),
and how a flux scale scales them (see FluxScale).
"""

import abc
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import numpy as np

from gbkfit.driver import DeviceArray
from gbkfit.model import Extras, SceneState
from gbkfit.params import ParamValues

if TYPE_CHECKING:
    from .noise import Noise


__all__ = [
    'FluxScale',
    'HostModelData',
    'ModelData',
    'ModelItem',
    'ObservationPlan'
]


class ModelItem[A](NamedTuple):
    """
    The model of a data item of an observation (see ObservationPlan):
    its arrays, of the shape of the data item, on the driver (see
    ModelData) or on the host (see HostModelData).

    Attributes
    ----------
    data : array
        The values of the model; NaN where it has no value.
    mask : array or None
        Where the model has a value (1) or not (0), if the observation
        measures it (e.g. moments, of the spectra where they have flux):
        an output, which the objective does not use.
    weights : array or None
        The weights of the values in the comparison with the data (e.g.
        the pixel weights, see Weights), if any.
    """
    data: A
    mask: A | None = None
    weights: A | None = None


# The model data of an observation on the driver: the model of each of
# its data items (e.g. 'spectra'), by key
ModelData: TypeAlias = dict[str, ModelItem[DeviceArray]]

# Model data copied to the host
HostModelData: TypeAlias = dict[str, ModelItem[np.ndarray]]


class FluxScale(NamedTuple):
    """
    How the values of a flux scale of an observation scale its model (see
    ObservationSky): the data items of light that they multiply, the axis
    of their arrays along which each value is one (one value at each point
    of the other axes, in their order, e.g. one for each spectrum), or None
    for one value for all, and the number of values.
    """
    items: tuple[str, ...]
    axis: int | None
    size: int


class ObservationPlan(abc.ABC):
    """The evaluation of the model data of an observation."""

    @abc.abstractmethod
    def evaluate(
            self,
            params: ParamValues,
            state: SceneState,
            out_extra: Extras | None
    ) -> ModelData:
        """
        Evaluate the model data.

        Parameters
        ----------
        params : dict
            The parameters of the model.
        state : SceneState
            The galaxies it sees in the evaluation: the state of each (its
            mass, and what it gives its components, see
            GalaxyStage.evaluate), in the order of the galaxies it sees,
            and the mass on the sky of each galaxy whose mass bends their
            light (see GalaxyState.sky).
        out_extra : Extras, optional
            Where the extra outputs go, if wanted: those of the instrument
            (e.g. the model cube), and those of the model (with the ids of
            their galaxies, and of their components).

        Returns
        -------
        ModelData
            The model data.
        """

    @abc.abstractmethod
    def mock(
            self,
            params: ParamValues,
            state: SceneState,
            noise: 'Noise',
            rng: np.random.Generator,
            ndraws: int
    ) -> tuple[list[HostModelData], dict[str, np.ndarray] | None]:
        """
        Draw mock data: the model data measured from the data that the
        instrument records of the model plus noise, ndraws times, each
        with its own noise, and their exact errors. The model is evaluated
        once: the noise of every draw and the errors are of its data
        without noise (the signal; see Noise).

        Parameters
        ----------
        params, state
            As for evaluate.
        noise : Noise
            The noise of the recorded data.
        rng : Generator
            The random numbers of the noise.
        ndraws : int
            The number of draws.

        Returns
        -------
        list of HostModelData
            The model data of each draw, on the host: of each data item,
            the data and their mask as evaluate gives them; no weights.
        dict or None
            The error of each value of each data item, by name, if known
            exactly (see has_exact_errors): the dispersion of the noise of
            the signal.

        Raises
        ------
        ValueError
            If the noise does not suit the recorded data (see
            Noise.draw).
        """

    @abc.abstractmethod
    def has_exact_errors(self) -> bool:
        """
        Check whether the errors of mock data are known exactly (see
        mock): if the data items are the recorded data plus the noise
        (e.g. spectra), the dispersion of the noise; not if they are
        measured from them (e.g. moments), when their errors are those of
        the scatter of draws.
        """
