"""
A group of galaxies and observations planned on devices (GroupPlan): the
stage of each galaxy and the plan of each observation, each on its
driver, with the memory of their models.
"""

import concurrent.futures
import contextvars
import copy
import threading
from collections.abc import Collection, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from gbkfit.dataset import Data, Dataset
from gbkfit.driver import Driver
from gbkfit.model import DataUnit, Extras, GalaxyFunctions, GalaxyState
from gbkfit.model.mass.units import G
from gbkfit.params import ParamFunctions, ParamValues
from gbkfit.utils import parseutils, threadutils, timeutils
from gbkfit.utils.parseutils import ConfigError
from .noise import Noise
from .observables import HostModelData, ModelData
from .observation import Observation

if TYPE_CHECKING:
    from .group import ObservationGroup


__all__ = [
    'GroupPlan'
]


class GroupPlan:
    """
    A group of galaxies and observations planned on devices (see
    ObservationGroup.plan): the stage of each galaxy and the plan of each
    observation, each on its driver, which own the memory of the models.
    It evaluates the models of the observations (see model_d) and their
    mock data (see mock).

    Its evaluations write the memory of its plans, under its lock (see
    lock): threads that evaluate it at the same time wait for each other.
    Plans of the group on other drivers have their own memory and lock
    (see ObservationGroup.plan); plans of it with other data share them
    (see with_data).

    An evaluation skips the stage of a galaxy, or the model of an
    observation, when the parameters it reads have exactly the values of
    its last evaluation (see model_d): e.g. a fitter that moves the noise
    of an observation, or the offset of its frame, evaluates no other
    model again.

    The observations on one driver evaluate their models one after the
    other; those on different drivers (e.g. one on each GPU, for models
    too big for one) at the same time, each driver in a thread of its own,
    after the stages of the galaxies.

    Parameters
    ----------
    group : ObservationGroup
        The group.
    galaxy_drivers : Sequence of Driver
        The driver of the stage of each galaxy (see Galaxy.plan_stage).
    observation_drivers : Sequence of Driver
        The driver of each observation (see Observation.plan).

    Raises
    ------
    ValueError
        If there is not one driver per galaxy and one per observation.
    ConfigError
        If what an observation sees does not go with it (see
        Observation.plan), or a galaxy cannot plan its stage (see
        Galaxy.plan_stage).
    """

    def __init__(
            self,
            group: 'ObservationGroup',
            galaxy_drivers: Sequence[Driver],
            observation_drivers: Sequence[Driver]
    ):
        galaxies, observations = group.galaxies(), group.observations()
        if len(galaxy_drivers) != len(galaxies) \
                or len(observation_drivers) != len(observations):
            raise ValueError(
                f"a group of {len(galaxies)} galaxies and "
                f"{len(observations)} observations needs a driver for each; "
                f"there are {len(galaxy_drivers)} and "
                f"{len(observation_drivers)}")
        self._group = group
        self._galaxy_drivers = tuple(galaxy_drivers)
        self._observation_drivers = tuple(observation_drivers)
        plans = []
        for i, (observation, driver) in enumerate(
                zip(observations, self._observation_drivers)):
            with parseutils.config_path('observations', i):
                plans.append(observation.plan(driver, group.seen_by(i)))
        self._plans = tuple(plans)
        stages = []
        for g, (galaxy, driver) in enumerate(
                zip(galaxies, self._galaxy_drivers)):
            with parseutils.config_path('galaxies', g):
                stages.append(galaxy.plan_stage(group.sky_origin(), driver))
        self._stages = tuple(stages)
        self._lock = threading.RLock()
        # The last evaluation of the stage of each galaxy, and of the model
        # of each observation (see model_d)
        self._stage_lasts = tuple(
            _LastValues(mapping.values()) for mapping in group.mappings())
        self._model_lasts = tuple(
            _LastValues(group.model_params(i))
            for i in range(len(observations)))
        # The observations on each driver, and the threads of the drivers
        # when there are several (see model_d)
        on_driver: dict[int, list[int]] = {}
        for i, driver in enumerate(self._observation_drivers):
            on_driver.setdefault(id(driver), []).append(i)
        self._on_drivers = tuple(tuple(on) for on in on_driver.values())
        self._executor = None if len(self._on_drivers) == 1 \
            else concurrent.futures.ThreadPoolExecutor(
                len(self._on_drivers), thread_name_prefix='gbkfit-driver')
        self._init_evaluations()

    def _init_evaluations(self) -> None:
        """
        Make what its evaluations write: the model data of each observation
        on the host and the device, and the times of the steps.
        """
        observations = self._group.observations()
        self._h_model_data = [
            {key: dict(d=None, m=None, w=None) for key in obs.keys()}
            for obs in observations]
        self._d_model_data = [dict() for _ in observations]
        # The times of the steps of its evaluations (and of those of the
        # objectives of it)
        self._timers = timeutils.Timers()

    def group(self) -> 'ObservationGroup':
        """Return the group."""
        return self._group

    def galaxy_drivers(self) -> tuple[Driver, ...]:
        """Return the driver of the stage of each galaxy."""
        return self._galaxy_drivers

    def observation_drivers(self) -> tuple[Driver, ...]:
        """Return the driver of each observation."""
        return self._observation_drivers

    def timers(self) -> timeutils.Timers:
        """Return the times of the steps of its evaluations."""
        return self._timers

    def lock(self) -> threading.RLock:
        """
        Return the lock of the memory of its plans, which its plans with
        other data share (see with_data). Its evaluations hold it; who
        reads the model data on the devices (see model_d) holds it until
        done with them (e.g. an objective, from the evaluation of the
        models to their residuals), as the next evaluation of a plan that
        shares them overwrites them.
        """
        return self._lock

    def with_data(self, datasets: Sequence[Dataset]) -> Self:
        """
        Return a plan of the group with other data (see
        ObservationGroup.with_data), on the same drivers.

        The plans of the observations do not depend on the values of their
        data, only on their unit (see Observation.data_unit): the copy
        shares the plans of those whose data keep their unit (e.g. all of
        them, in a bootstrap), and with them the memory of their models,
        on their devices, and its lock (see lock): evaluated at the same time
        (e.g. in two threads), they wait for each other (plans on other
        drivers do not, see ObservationGroup.plan). They share their last
        evaluations too (see model_d). A copy whose observations have other
        parameters (e.g. a likelihood of data that they had none of) has
        plans of its own.

        Raises
        ------
        ConfigError
            If there is not one dataset per observation, or a dataset does
            not suit its observation.
        """
        group = self._group.with_data(datasets)
        olds, news = self._group.observations(), group.observations()
        if any(new.pdescs().keys() != old.pdescs().keys()
               for new, old in zip(news, olds)):
            return type(self)(
                group, self._galaxy_drivers, self._observation_drivers)
        plans = []
        lasts = []
        for i, (plan, last, new, old, driver) in enumerate(zip(
                self._plans, self._model_lasts, news, olds,
                self._observation_drivers)):
            if not _same_unit(new.data_unit(), old.data_unit()):
                with parseutils.config_path('observations', i):
                    plan = new.plan(driver, group.seen_by(i))
                last = _LastValues(group.model_params(i))
            plans.append(plan)
            lasts.append(last)
        copied = copy.copy(self)
        copied._group = group
        copied._plans = tuple(plans)
        copied._model_lasts = tuple(lasts)
        copied._init_evaluations()
        return copied

    def plan_functions(self) -> ParamFunctions:
        """
        Plan the functions of the galaxies, for the derived values of the
        parameters (see ParamSpace): those of each galaxy (see
        Galaxy.plan_functions), on the driver of its stage, prefixed as its
        parameters (e.g. 'north_vcirc'), and the gravitational constant G
        ((km/s)^2 kpc / Msun). They have their own stages and plans, apart
        from those of its evaluations (which may run at the same time).

        Returns
        -------
        ParamFunctions
            The functions.
        """
        group = self._group
        galaxies = zip(group.galaxies(), self._galaxy_drivers)
        return _GroupFunctions(
            [galaxy.plan_functions(group.sky_origin(), driver)
             for galaxy, driver in galaxies],
            group.mappings(), group.prefixes())

    def _galaxy_states(
            self,
            params: ParamValues,
            out_extra: Extras | None
    ) -> list[GalaxyState]:
        """
        Evaluate the stage of each galaxy: its mass, and what it gives its
        components (see GalaxyStage.evaluate); its extra outputs go to
        out_extra, if given, with the label of the galaxy as the id
        'galaxy'. Without them, the state of a galaxy whose parameters have
        the values of its last evaluation is that of its last evaluation.
        """
        group = self._group
        states = []
        for g, (stage, last, mapping) in enumerate(zip(
                self._stages, self._stage_lasts, group.mappings())):
            state = None if out_extra is not None else last.get(params)
            if state is None:
                state = stage.evaluate(
                    {name: params[full] for name, full in mapping.items()},
                    None if out_extra is None
                    else out_extra.scoped(galaxy=group.galaxy_label(g)))
                last.keep(params, state)
            states.append(state)
        return states

    def galaxy_states(self, params: ParamValues) -> list[GalaxyState]:
        """
        Return the state of each galaxy in an evaluation (see
        GalaxyStage.evaluate), as model_d evaluates it: that of its last
        evaluation if its parameters have the same values.
        """
        with self._lock:
            return self._galaxy_states(params, None)

    def model_d(
            self,
            params: ParamValues,
            out_extra: Extras | None = None,
            skip: Collection[int] = ()
    ) -> list[ModelData]:
        """
        Evaluate the model data of the observations, on their drivers,
        with BLAS on one thread (see threadutils.single_threaded_blas),
        but those of the observations of the indices skip, whose models
        are made elsewhere (e.g. by an objective, see
        Component.orbit_weights): their model data are those of their last
        evaluation, if any.

        Without extra outputs, an observation whose parameters (see
        ObservationGroup.model_params) have exactly the values of its last
        evaluation (by this plan, or by one with other data that shares
        its plans, see with_data) keeps the model data of that evaluation,
        which nothing else writes; so does the stage of a galaxy (the state
        of its mass, and what it gives its components), whose parameters
        are its own. NaN is never the value of the last evaluation.

        After the stages of the galaxies, the observations on each driver
        are evaluated one after the other, and those on different drivers
        at the same time, each driver in a thread of its own (the extra
        outputs of each observation go to out_extra after, in their
        order); an error of one is raised once all are done.

        Parameters
        ----------
        params : dict
            The parameters of the group (see ObservationGroup.pdescs).
        out_extra : Extras, optional
            Where the extra outputs go, if wanted: those of each
            observation with its label as the id 'observation' (see
            ObservationGroup.observation_label), and those of the stage of
            each galaxy (e.g. its rotation tables, see GalaxyStage.evaluate)
            with its label as the id 'galaxy'.

        Returns
        -------
        list of ModelData
            The model data of each observation, on its driver; they are
            overwritten by the next evaluation (of it, or of a plan that
            shares its memory: hold its lock while reading them, see
            lock).
        """
        with (self._lock, self._timers.measure('model_eval'),
              threadutils.single_threaded_blas()):
            with self._timers.measure('model_gravity'):
                states = self._galaxy_states(params, out_extra)
            evaluated = [i for i in range(len(self._plans)) if i not in skip]
            if self._executor is None:
                for i in evaluated:
                    self._evaluate_model(i, params, states, out_extra)
            else:
                self._evaluate_models_at_once(
                    evaluated, params, states, out_extra)
        return self._d_model_data

    def _evaluate_model(
            self,
            index: int,
            params: ParamValues,
            states: Sequence[GalaxyState],
            out_extra: Extras | None
    ) -> None:
        """
        Evaluate the model of the observation of the given index, in the
        states of the galaxies (see model_d).
        """
        group = self._group
        last = self._model_lasts[index]
        model = None if out_extra is not None else last.get(params)
        if model is None:
            model = self._plans[index].evaluate(
                group.moved(index, params),
                [states[g] for g in group.seen_galaxies(index)],
                None if out_extra is None else out_extra.scoped(
                    observation=group.observation_label(index)))
            last.keep(params, model)
        self._d_model_data[index] = model

    def _evaluate_models_at_once(
            self,
            evaluated: Collection[int],
            params: ParamValues,
            states: Sequence[GalaxyState],
            out_extra: Extras | None
    ) -> None:
        """
        Evaluate the models of the observations of the given indices on
        each driver one after the other, in a thread of the driver, with
        the context of the caller (e.g. the paths of configuration
        errors); the extra outputs of each observation go to out_extra
        after all are done, in their order (see model_d).
        """
        extras = [
            None if out_extra is None else Extras() for _ in self._plans]

        def evaluate(indices: tuple[int, ...]) -> None:
            for i in indices:
                if i in evaluated:
                    self._evaluate_model(i, params, states, extras[i])
        futures = [
            self._executor.submit(
                contextvars.copy_context().run, evaluate, indices)
            for indices in self._on_drivers]
        concurrent.futures.wait(futures)
        for future in futures:
            future.result()
        if out_extra is not None:
            for extras_i in extras:
                for x in extras_i:
                    out_extra.add(
                        x.name, x.value, x.title, x.description, **x.ids)

    def model_h(
            self,
            params: ParamValues,
            out_extra: Extras | None = None
    ) -> list[dict[str, dict[str, np.ndarray | None]]]:
        """
        Evaluate the model data of the observations, on the host.

        Parameters
        ----------
        params : dict
            The parameters of the group (see ObservationGroup.pdescs).
        out_extra : Extras, optional
            Where the extra outputs go, if wanted (see model_d).

        Returns
        -------
        list of dict
            The model data of each observation (see ModelData), copied to
            the host; they are overwritten by the next evaluation.
        """
        with self._lock:
            self.model_d(params, out_extra)
            self._copy_to_host()
        return self._h_model_data

    def _copy_to_host(self) -> None:
        """Copy the model data of the last evaluation to the host."""
        with self._timers.measure('model_d2h'):
            for i, driver in enumerate(self._observation_drivers):
                for key, d_item in self._d_model_data[i].items():
                    h_item = self._h_model_data[i][key]
                    for k, d_array in d_item.items():
                        if d_array is not None:
                            h_item[k] = driver.mem_copy_d2h(
                                d_array, h_item[k])

    def mock(
            self,
            params: ParamValues,
            noises: Noise | Sequence[Noise],
            nmocks: int = 1,
            nsamples: int = 100,
            seed: int | None = None
    ) -> list[list[Dataset]]:
        """
        Draw mock data of the observations: their model data measured
        from what their instruments record of the model plus noise (see
        ObservablePlan.mock), with their errors.

        The data that are what the instrument records plus the noise (e.g.
        spectra) have its dispersion as their errors, and declare its
        correlation; those measured from them (e.g. moments) the scatter
        of their values over nsamples other draws (the standard deviation,
        of the values that are finite in at least two draws; the others
        are missing). The mock data have the unit and the beam of the
        data of their observation, if it has data.

        Parameters
        ----------
        params : dict
            The parameters of the group (see ObservationGroup.pdescs).
        noises : Noise or Sequence of Noise
            The noise of the data of every observation, or of each.
        nmocks : int, optional
            The number of mock datasets of each observation.
        nsamples : int, optional
            The number of draws of the errors of measured data.
        seed : int, optional
            The seed of the random numbers; the same seed gives the same
            mock data. By default, the mock data differ from run to run.

        Returns
        -------
        list of list of Dataset
            The mock datasets of each observation, in the units of its
            model.

        Raises
        ------
        ConfigError
            If there is not one noise per observation, or one, a noise does
            not suit the data of its observation (see Noise.draw), or
            nmocks or nsamples (needed) is below 1 or 2.
        """
        observations = self._group.observations()
        noises = list(noises) if isinstance(noises, Sequence) \
            else [noises] * len(observations)
        if len(noises) != len(observations):
            raise ConfigError(
                f"there are {len(observations)} observations, but "
                f"{len(noises)} noises")
        if nmocks < 1:
            raise ConfigError(f"nmocks must be at least 1; it is {nmocks}")
        streams = np.random.SeedSequence(seed).spawn(len(observations))
        with self._lock:
            states = self._galaxy_states(params, None)
            try:
                return [
                    self._mock(
                        i, params, states, noise, stream, nmocks, nsamples)
                    for i, (noise, stream) in enumerate(zip(noises, streams))]
            finally:
                # (the draws overwrite the model data of the observations)
                for last in self._model_lasts:
                    last.forget()

    def _mock(
            self,
            index: int,
            params: ParamValues,
            states: Sequence[GalaxyState],
            noise: Noise,
            stream: np.random.SeedSequence,
            nmocks: int,
            nsamples: int
    ) -> list[Dataset]:
        """
        Draw the mock datasets of the observation of the given index (see
        mock), in the states of the galaxies, with the random numbers of
        the stream.
        """
        group = self._group
        observation, plan = group.observations()[index], self._plans[index]
        with parseutils.config_path('observations', index):
            exact = plan.has_exact_errors()
            if not exact and nsamples < 2:
                raise ConfigError(
                    f"the errors of measured data are the scatter of "
                    f"nsamples draws, at least 2; nsamples is {nsamples}")
            draws, errors = plan.mock(
                group.moved(index, params),
                [states[g] for g in group.seen_galaxies(index)],
                noise, np.random.default_rng(stream),
                nmocks + (0 if exact else nsamples))
        if errors is None:
            errors = _scatter(draws[nmocks:])
        options = _mock_options(observation, noise, exact)
        return [
            observation.dataset({
                key: Data(item['d'], error=errors[key], **options[key])
                for key, item in draw.items()})
            for draw in draws[:nmocks]]


class _LastValues:
    """
    The last evaluation of a stage of the evaluations of a group plan
    (the stage of a galaxy, or the model of an observation; see
    GroupPlan.model_d): the values of the parameters of the given names,
    which it reads, and what it gave.
    """

    def __init__(self, names: Iterable[str]):
        self._names = tuple(names)
        self._values: dict[str, Any] | None = None
        self._result: Any = None

    def get(self, params: ParamValues) -> Any:
        """
        Return what the stage gave at its last evaluation if the values of
        its parameters are the same (see _same); else None.
        """
        if self._values is None or not all(
                _same(params[name], value)
                for name, value in self._values.items()):
            return None
        return self._result

    def keep(self, params: ParamValues, result: Any) -> None:
        """
        Keep what the stage gave at the values of its parameters (copies
        of them).
        """
        self._values = {
            name: params[name] if np.isscalar(params[name])
            else np.array(params[name]) for name in self._names}
        self._result = result

    def forget(self) -> None:
        """Forget the last evaluation (e.g. its memory was overwritten)."""
        self._values = None
        self._result = None


def _same(a: Any, b: Any) -> bool:
    """
    Check whether two values of a parameter are the same: equal, element
    by element for arrays; NaN is not the same as itself.
    """
    if np.isscalar(a) and np.isscalar(b):
        return bool(a == b)
    return np.array_equal(a, b)


class _GroupFunctions(ParamFunctions):
    """
    The functions of the galaxies of a group (see
    GroupPlan.plan_functions): those of each galaxy, the mapping of its
    parameters to those of the group, and its prefix.
    """

    def __init__(
            self,
            functions: Sequence[GalaxyFunctions],
            mappings: Sequence[dict[str, str]],
            prefixes: Sequence[str]
    ):
        self._functions = tuple(functions)
        self._mappings = tuple(mappings)
        self._prefixes = tuple(prefixes)

    def names(self) -> tuple[str, ...]:
        return ('G',) + tuple(
            f'{prefix}{name}'
            for functions, prefix in zip(self._functions, self._prefixes)
            for name in functions.names())

    def bind(self, values: ParamValues) -> dict[str, Any]:
        bound: dict[str, Any] = dict(G=G)
        for functions, mapping, prefix in zip(
                self._functions, self._mappings, self._prefixes):
            own = functions.bind(
                {name: values[full] for name, full in mapping.items()})
            bound |= {f'{prefix}{name}': value for name, value in own.items()}
        return bound


def _same_unit(a: DataUnit | None, b: DataUnit | None) -> bool:
    """
    Return whether two units of the light of data are the same (see
    DataUnit, whose beams can be arrays, one for each spectral point).
    """
    if a is None or b is None:
        return a is b
    return all(np.array_equal(x, y) if isinstance(x, np.ndarray)
               else x == y for x, y in zip(a, b))


def _mock_options(
        observation: Observation, noise: Noise, exact: bool
) -> dict[str, dict[str, Any]]:
    """
    Return the options of the mock data of each data item of an
    observation (see GroupPlan.mock): the unit and the beam of its
    data, if any, and the correlation of the noise of those that are what
    the instrument records plus the noise (exact errors).
    """
    spatial, spectral = noise.correlations() if exact else (None, None)
    if spatial == 'instrument':
        # (only observations on the sky draw noise correlated by their
        # instrument; see Noise.draw)
        spatial = observation.instrument().psf()
    data = observation.data()
    return {
        key: dict(
            spatial_correlation=spatial, spectral_correlation=spectral)
        | ({} if data is None else dict(
            unit=data[key].unit(), beam=data[key].beam()))
        for key in observation.keys()}


def _scatter(draws: Sequence[HostModelData]) -> dict[str, np.ndarray]:
    """
    Return the scatter of the values of each data item over draws of
    them: the standard deviation of the values that are finite in at
    least two draws (NaN for the others).
    """
    scatter = {}
    for key in draws[0]:
        values = np.stack([draw[key]['d'] for draw in draws])
        finite = np.isfinite(values)
        count = finite.sum(axis=0)
        values = np.where(finite, values, 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean = values.sum(axis=0) / count
            variance = np.where(
                finite, (values - mean) ** 2, 0).sum(axis=0) / (count - 1)
        scatter[key] = np.where(count >= 2, np.sqrt(variance), np.nan)
    return scatter
