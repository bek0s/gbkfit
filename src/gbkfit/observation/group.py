from collections.abc import Sequence
from typing import Any

import numpy as np

from gbkfit.model.base import Model
from gbkfit.params import ParamDesc
from gbkfit.utils import iterutils, parseutils, timeutils
from gbkfit.utils.parseutils import ConfigError
from .observables import ModelData
from .observation import Observation


__all__ = [
    'ObservationGroup'
]


class ObservationGroup:
    """
    Models and the observations of them, evaluated together.

    Each observation refers to its model by name, which it may leave out
    when there is one model. The parameters and constants of the models
    are prefixed by their names, or their positions (e.g. 'model1_').
    The plans of the observations are made here, once.

    Parameters
    ----------
    models : Model or Sequence of Model
        The models.
    observations : Observation or Sequence of Observation
        The observations.

    Raises
    ------
    ConfigError
        If there are no models or no observations, their names repeat, or
        an observation does not name a model of the group (when there are
        several).
    """

    def __init__(
            self,
            models: Model | Sequence[Model],
            observations: Observation | Sequence[Observation]
    ):
        self._models = iterutils.tuplify(models)
        self._observations = iterutils.tuplify(observations)
        if not self._models:
            raise ConfigError("at least one model is required")
        if not self._observations:
            raise ConfigError("at least one observation is required")
        model_names = [model.name() for model in self._models]
        observation_names = [obs.name() for obs in self._observations]
        self._prefixes = parseutils.item_prefixes(
            model_names, 'models', 'model', False)
        parseutils.item_prefixes(
            observation_names, 'observations', 'observation', False)
        repeated = sorted(
            set(model_names) & set(observation_names) - {None})
        if repeated:
            raise ConfigError(
                f"the models and the observations must have different "
                f"names; repeated: {repeated}")
        self._model_index = [
            self._resolve(i, obs) for i, obs in enumerate(self._observations)]
        self._pdescs, self._mappings = iterutils.merge_with_prefixes(
            [model.pdescs() for model in self._models], self._prefixes)
        self._constants, _ = iterutils.merge_with_prefixes(
            [model.constants() for model in self._models], self._prefixes)
        # The extra outputs of the observations are named after their names,
        # or their index if they have none (e.g. 'observation0_')
        self._extra_prefixes = [
            f'{name}_' if name is not None else f'observation{i}_'
            for i, name in enumerate(observation_names)]
        self._plans = tuple(
            obs.plan(self._models[j])
            for obs, j in zip(self._observations, self._model_index))
        self._h_model_data = [
            {key: dict(d=None, m=None, w=None)
             for key in obs.observable().keys()}
            for obs in self._observations]
        self._d_model_data = [dict() for _ in self._observations]
        # The times of the steps of its evaluations (and of those of the
        # objectives of its data)
        self._timers = timeutils.Timers()

    def _resolve(self, i: int, observation: Observation) -> int:
        """Return the index of the model of observation i."""
        name = observation.model()
        if name is None:
            if len(self._models) > 1:
                raise ConfigError(
                    f"observation {i} must name its model; there are "
                    f"{len(self._models)} models")
            return 0
        names = [model.name() for model in self._models]
        if name not in names:
            raise ConfigError(
                f"observation {i} refers to an unknown model '{name}'; "
                f"the models are {names}")
        return names.index(name)

    def models(self) -> tuple[Model, ...]:
        """Return the models."""
        return self._models

    def observations(self) -> tuple[Observation, ...]:
        """Return the observations."""
        return self._observations

    def nobservations(self) -> int:
        """Return the number of observations."""
        return len(self._observations)

    def model_of(self, i: int) -> Model:
        """Return the model of observation i."""
        return self._models[self._model_index[i]]

    def pdescs(self) -> dict[str, ParamDesc]:
        """Return the parameters of the models, prefixed."""
        return self._pdescs

    def constants(self) -> dict[str, Any]:
        """Return the constants of the models, prefixed."""
        return self._constants

    def timers(self) -> timeutils.Timers:
        """Return the times of the steps of the evaluations of the group."""
        return self._timers

    def model_d(
            self,
            params: dict[str, float | np.ndarray],
            out_extra: dict[str, Any] | None = None
    ) -> list[ModelData]:
        """
        Evaluate the model data of the observations, on their drivers.

        Parameters
        ----------
        params : dict
            The parameters of the models, prefixed (see pdescs).
        out_extra : dict, optional
            Where the extra outputs go, if wanted: those of each
            observation prefixed by its name, or 'observation{i}_'.

        Returns
        -------
        list of ModelData
            The model data of each observation, on its driver; they are
            overwritten by the next evaluation.
        """
        with self._timers.measure('model_eval'):
            for i, plan in enumerate(self._plans):
                mapping = self._mappings[self._model_index[i]]
                out_extra_i = {} if out_extra is not None else None
                self._d_model_data[i] = plan.evaluate(
                    {param: params[mapping[param]] for param in mapping},
                    out_extra_i)
                if out_extra is not None:
                    for key, val in out_extra_i.items():
                        out_extra[f'{self._extra_prefixes[i]}{key}'] = val
        return self._d_model_data

    def model_h(
            self,
            params: dict[str, float | np.ndarray],
            out_extra: dict[str, Any] | None = None
    ) -> list[dict[str, dict[str, np.ndarray | None]]]:
        """
        Evaluate the model data of the observations, on the host.

        Parameters
        ----------
        params : dict
            The parameters of the models, prefixed (see pdescs).
        out_extra : dict, optional
            Where the extra outputs go, if wanted (see model_d).

        Returns
        -------
        list of dict
            The model data of each observation (see ModelData), copied to
            the host; they are overwritten by the next evaluation.
        """
        self.model_d(params, out_extra)
        with self._timers.measure('model_d2h'):
            for i, obs in enumerate(self._observations):
                driver = obs.driver()
                for key, d_item in self._d_model_data[i].items():
                    h_item = self._h_model_data[i][key]
                    for k, d_array in d_item.items():
                        if d_array is not None:
                            h_item[k] = driver.mem_copy_d2h(
                                d_array, h_item[k])
        return self._h_model_data
