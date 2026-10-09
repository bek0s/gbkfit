from typing import Any

import numpy as np

from gbkfit.params import ParamDesc
from gbkfit.utils import iterutils, miscutils, parseutils, timeutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'ObservationGroup'
]


class ObservationGroup:
    """
    Gmodels and the observations of them. Each observation refers to its
    gmodel by name, which it may leave out when there is one gmodel. The
    parameters and constants of the gmodels are prefixed by their names, or
    their positions (e.g. 'gmodel1_'). The plans of the observations are
    made here, once.
    """

    def __init__(self, gmodels, observations):
        self._gmodels = iterutils.tuplify(gmodels, False)
        self._observations = iterutils.tuplify(observations, False)
        if not self._gmodels:
            raise ConfigError("at least one gmodel is required")
        if not self._observations:
            raise ConfigError("at least one observation is required")
        gmodel_names = [gmodel.name() for gmodel in self._gmodels]
        observation_names = [obs.name() for obs in self._observations]
        self._prefixes = parseutils.item_prefixes(
            gmodel_names, 'gmodels', 'gmodel', False)
        parseutils.item_prefixes(
            observation_names, 'observations', 'observation', False)
        repeated = sorted(
            set(gmodel_names) & set(observation_names) - {None})
        if repeated:
            raise ConfigError(
                f"the gmodels and the observations must have different "
                f"names; repeated: {repeated}")
        self._gmodel_index = [
            self._resolve(i, obs) for i, obs in enumerate(self._observations)]
        self._pdescs, self._mappings = miscutils.merge_with_prefixes(
            [gmodel.pdescs() for gmodel in self._gmodels], self._prefixes)
        self._constants, _ = miscutils.merge_with_prefixes(
            [gmodel.constants() for gmodel in self._gmodels], self._prefixes)
        # The extra outputs of the observations are named after their names,
        # or their index if they have none (e.g. 'observation0_')
        self._extra_prefixes = [
            f'{name}_' if name is not None else f'observation{i}_'
            for i, name in enumerate(observation_names)]
        self._plans = tuple(
            obs.plan(self._gmodels[j])
            for obs, j in zip(self._observations, self._gmodel_index))
        self._h_model_data = [
            {key: dict(d=None, m=None, w=None)
             for key in obs.observable().keys()}
            for obs in self._observations]
        self._d_model_data = [dict() for _ in self._observations]

    def _resolve(self, i, observation):
        """The index of the gmodel of the observation."""
        name = observation.gmodel()
        if name is None:
            if len(self._gmodels) > 1:
                raise ConfigError(
                    f"observation {i} must name its gmodel; there are "
                    f"{len(self._gmodels)} gmodels")
            return 0
        names = [gmodel.name() for gmodel in self._gmodels]
        if name not in names:
            raise ConfigError(
                f"observation {i} refers to an unknown gmodel '{name}'; "
                f"the gmodels are {names}")
        return names.index(name)

    def gmodels(self):
        return self._gmodels

    def observations(self):
        return self._observations

    def nobservations(self) -> int:
        return len(self._observations)

    def gmodel_of(self, i):
        """The gmodel of observation i."""
        return self._gmodels[self._gmodel_index[i]]

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._pdescs

    def constants(self) -> dict[str, Any]:
        return self._constants

    def model_d(
            self,
            params: dict[str, float | np.ndarray],
            out_extra: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        t = timeutils.SimpleTimer('model_eval').start()
        for i, plan in enumerate(self._plans):
            mapping = self._mappings[self._gmodel_index[i]]
            out_extra_i = {} if out_extra is not None else None
            self._d_model_data[i] = plan.evaluate(
                {param: params[mapping[param]] for param in mapping},
                out_extra_i)
            if out_extra is not None:
                for key, val in out_extra_i.items():
                    out_extra[f'{self._extra_prefixes[i]}{key}'] = val
        t.stop()
        return self._d_model_data

    def model_h(
            self,
            params: dict[str, float | np.ndarray],
            out_extra: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.model_d(params, out_extra)
        t = timeutils.SimpleTimer('model_d2h').start()
        for i, obs in enumerate(self._observations):
            driver = obs.driver()
            for key, d_item in self._d_model_data[i].items():
                h_item = self._h_model_data[i][key]
                for k, d_array in d_item.items():
                    if d_array is not None:
                        h_item[k] = driver.mem_copy_d2h(d_array, h_item[k])
        t.stop()
        return self._h_model_data
