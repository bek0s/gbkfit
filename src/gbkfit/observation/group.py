from collections.abc import Sequence
from typing import Any

import numpy as np

from gbkfit.model import Galaxy
from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import iterutils, parseutils, threadutils, timeutils
from gbkfit.utils.parseutils import ConfigError
from ._scene import Scene, make_scene
from .observables import ModelData
from .observation import Observation


__all__ = [
    'ObservationGroup'
]


class ObservationGroup:
    """
    Galaxies and the observations of them, evaluated together.

    Each observation names the galaxies it sees, which it may leave out
    when there is one galaxy; the light of several galaxies adds up. The
    parameters and constants of the galaxies are prefixed by their names
    when they have names (which they need when there are several; e.g.
    'north_'). The plans of the observations are made here, once. The
    gravity of each galaxy is evaluated once for all the observations of
    it (see Galaxy.plan_gravity).

    Parameters
    ----------
    galaxies : Galaxy or Sequence of Galaxy
        The galaxies.
    observations : Observation or Sequence of Observation
        The observations.

    Raises
    ------
    ConfigError
        If there are no galaxies or no observations, several galaxies
        without names, names repeat, an observation does not name the
        galaxies it sees (when there are several) or names unknown ones,
        or what an observation sees does not go with it (see make_scene).
    """

    def __init__(
            self,
            galaxies: Galaxy | Sequence[Galaxy],
            observations: Observation | Sequence[Observation]
    ):
        self._galaxies = iterutils.tuplify(galaxies)
        self._observations = iterutils.tuplify(observations)
        if not self._galaxies:
            raise ConfigError("at least one galaxy is required")
        if not self._observations:
            raise ConfigError("at least one observation is required")
        galaxy_names = [galaxy.name() for galaxy in self._galaxies]
        if len(galaxy_names) > 1 and None in galaxy_names:
            raise ConfigError(
                "the galaxies need names when there are several")
        observation_names = [obs.name() for obs in self._observations]
        self._prefixes = parseutils.item_prefixes(
            galaxy_names, 'galaxies', 'galaxy', False)
        parseutils.item_prefixes(
            observation_names, 'observations', 'observation', False)
        repeated = sorted(
            set(galaxy_names) & set(observation_names) - {None})
        if repeated:
            raise ConfigError(
                f"the galaxies and the observations must have different "
                f"names; repeated: {repeated}")
        self._pdescs, self._mappings = iterutils.merge_with_prefixes(
            [galaxy.pdescs() for galaxy in self._galaxies], self._prefixes)
        # The parameters of the observations (their calibrations)
        for observation in self._observations:
            pdescs = observation.pdescs()
            if repeated := sorted(pdescs.keys() & self._pdescs.keys()):
                raise ConfigError(
                    f"the parameters of the observations have the names of "
                    f"parameters of the galaxies: {repeated}; rename the "
                    f"observations")
            self._pdescs |= pdescs
        self._constants, _ = iterutils.merge_with_prefixes(
            [galaxy.constants() for galaxy in self._galaxies], self._prefixes)
        # The extra outputs of the observations are named after their names,
        # or their index if they have none (e.g. 'observation0_')
        self._extra_prefixes = [
            f'{name}_' if name is not None else f'observation{i}_'
            for i, name in enumerate(observation_names)]
        plans = []
        seen = []
        for i, observation in enumerate(self._observations):
            with parseutils.config_path('observations', i):
                seen.append(self._seen(i, observation))
                plans.append(observation.plan(
                    self._scene(seen[-1], observation)))
        self._plans = tuple(plans)
        # The indices of the galaxies that each observation sees
        self._seen_galaxies = tuple(seen)
        self._gravity_plans = tuple(
            galaxy.plan_gravity() for galaxy in self._galaxies)
        self._h_model_data = [
            {key: dict(d=None, m=None, w=None)
             for key in obs.observable().keys()}
            for obs in self._observations]
        self._d_model_data = [dict() for _ in self._observations]
        # The times of the steps of its evaluations (and of those of the
        # objectives of its data)
        self._timers = timeutils.Timers()

    def _seen(self, i: int, observation: Observation) -> list[int]:
        """Return the indices of the galaxies that observation i sees."""
        names = observation.galaxies()
        if names is None:
            if len(self._galaxies) > 1:
                raise ConfigError(
                    f"observation {i} must name the galaxies it sees; there "
                    f"are {len(self._galaxies)} galaxies")
            return [0]
        known = [galaxy.name() for galaxy in self._galaxies]
        if unknown := [name for name in names if name not in known]:
            raise ConfigError(
                f"observation {i} refers to unknown galaxies {unknown}; the "
                f"galaxies are {known}")
        return sorted(known.index(name) for name in names)

    def _scene(self, seen: list[int], observation: Observation) -> Scene:
        """
        Return what an observation sees (see make_scene): the galaxies of
        the given indices.
        """
        return make_scene(
            [self._galaxies[j] for j in seen],
            [self._mappings[j] for j in seen],
            [self._prefixes[j] for j in seen],
            observation.components(), observation.lines(),
            observation.band(), observation.line_of_sight(),
            observation.observable().emission)

    def galaxies(self) -> tuple[Galaxy, ...]:
        """Return the galaxies."""
        return self._galaxies

    def observations(self) -> tuple[Observation, ...]:
        """Return the observations."""
        return self._observations

    def nobservations(self) -> int:
        """Return the number of observations."""
        return len(self._observations)

    def pdescs(self) -> dict[str, ParamDesc]:
        """
        Return the parameters of the galaxies, prefixed, and those of the
        observations (see Observation.pdescs).
        """
        return self._pdescs

    def constants(self) -> dict[str, Any]:
        """Return the constants of the galaxies, prefixed."""
        return self._constants

    def timers(self) -> timeutils.Timers:
        """Return the times of the steps of the evaluations of the group."""
        return self._timers

    def model_d(
            self,
            params: ParamValues,
            out_extra: dict[str, Any] | None = None
    ) -> list[ModelData]:
        """
        Evaluate the model data of the observations, on their drivers,
        with BLAS on one thread (see threadutils.single_threaded_blas).

        Parameters
        ----------
        params : dict
            The parameters of the galaxies, prefixed (see pdescs).
        out_extra : dict, optional
            Where the extra outputs go, if wanted: those of each
            observation prefixed by its name, or 'observation{i}_'.

        Returns
        -------
        list of ModelData
            The model data of each observation, on its driver; they are
            overwritten by the next evaluation.
        """
        with (self._timers.measure('model_eval'),
              threadutils.single_threaded_blas()):
            with self._timers.measure('model_gravity'):
                gravity = [
                    plan.evaluate(
                        {name: params[full] for name, full in mapping.items()})
                    for plan, mapping in zip(
                        self._gravity_plans, self._mappings)]
            for i, plan in enumerate(self._plans):
                out_extra_i = {} if out_extra is not None else None
                self._d_model_data[i] = plan.evaluate(
                    params, [gravity[g] for g in self._seen_galaxies[i]],
                    out_extra_i)
                if out_extra is not None:
                    for key, val in out_extra_i.items():
                        out_extra[f'{self._extra_prefixes[i]}{key}'] = val
        return self._d_model_data

    def model_h(
            self,
            params: ParamValues,
            out_extra: dict[str, Any] | None = None
    ) -> list[dict[str, dict[str, np.ndarray | None]]]:
        """
        Evaluate the model data of the observations, on the host.

        Parameters
        ----------
        params : dict
            The parameters of the galaxies, prefixed (see pdescs).
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
