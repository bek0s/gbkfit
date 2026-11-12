"""
A group of galaxies and observations planned on devices (GroupPlan): the
stage of each galaxy and the plan of each observation, each on its
driver, with the memory of their models, and the models of observations
linear in amplitudes of their light (LinearPieces).
"""

import concurrent.futures
import contextlib
import contextvars
import copy
import functools
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, Self

from gbkfit.config import config_path
from gbkfit.dataset import Dataset
from gbkfit.driver import DeviceArray, Driver
from gbkfit.model import Extras, GalaxyFunctions, GalaxyState
from gbkfit.model.mass.units import G
from gbkfit.params import (
    ParamFunctions, ParamValues, ParamView, StageCache, with_elements)
from gbkfit.utils import threadutils, timeutils
from .orbits import OrbitRecording, OrbitSeries, orbit_recordings
from .observables import HostModelData, ModelData, ModelItem, ObservablePlan

if TYPE_CHECKING:
    from .group import ObservationGroup


__all__ = [
    'GroupPlan',
    'LinearPieces'
]


# The host arrays of an item of model data before its first copy to the
# host (see GroupPlan.model_h): none, so the copy makes them
_NO_ITEM = ModelItem(None)


class LinearPieces(NamedTuple):
    """
    The model of an observation as linear in elements of the parameters
    (see GroupPlan.amplitude_pieces): base + sum_k x_k pieces[k] at the
    values x_k of the elements, on the driver of the observation.

    Attributes
    ----------
    base : ModelData
        The model with the elements at 0.
    pieces : dict
        The pieces of each data item, by key: an array of the piece of
        each element along its first axis, each of the shape of the item.
    """
    base: ModelData
    pieces: dict[str, DeviceArray]


class GroupPlan:
    """
    A group of galaxies and observations planned on devices (see
    ObservationGroup.plan): the stage of each galaxy and the plan of each
    observation, each on its driver, which own the memory of the models.
    It evaluates the models of the observations (see model_d), and lends
    the plans of the observations to draw their mock data (see
    borrow_plans and draw_mocks).

    Its evaluations write the memory of its plans, under its lock (see
    lock): threads that evaluate it at the same time wait for each other.
    Plans of the group on other drivers have their own memory and lock
    (see ObservationGroup.plan); plans of it with other data share them
    (see with_data).

    The orbits of the stars of a spheroid whose weights are solved (see
    Component.orbit_weights) that observations with data see are recorded
    once for all of them, in the series of the bins of their data (see
    OrbitRecording and orbit_series): their models are linear in the
    weights there, as an objective fits them (see objective.Objective).
    Observations of the orbits without data have the models of their own
    plans (see OrbitMomentsPlan).

    The models of observations that see amplitudes of the light of
    components (see Component.light_amplitudes) are linear in them: it
    gives them as pieces (see amplitude_pieces), from which an objective
    solves them (see objective.Objective).

    An evaluation skips each part of the stage of a galaxy, a recording
    of orbits, or the model of an observation, when the parameters it
    reads have exactly the values of its last evaluation, and the parts
    it takes are those of it (see model_d and stage_caches): e.g. a
    fitter that moves the noise of an observation, or the offset of its
    frame, evaluates no other model again, and one that moves the
    position angle of a spheroid of Jeans kinematics only the models of
    its observations.

    The observations on one driver evaluate their models one after the
    other; those on different drivers (e.g. one on each GPU, for models
    too big for one) at the same time, each driver in a thread of its own,
    after the stages of the galaxies and the recordings of orbits.

    A copy of it (e.g. pickled for another process) is a plan of its group
    on drivers like its own, with memory of its own.

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
    ValueError
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
        self._functions = _GroupFunctions(group, self._galaxy_drivers)
        plans = []
        for i, (observation, driver) in enumerate(
                zip(observations, self._observation_drivers)):
            with config_path('observations', i):
                plans.append(observation.plan(driver, group.seen_by(i)))
        self._plans = tuple(plans)
        stages = []
        for g, (galaxy, driver) in enumerate(
                zip(galaxies, self._galaxy_drivers)):
            with config_path('galaxies', g):
                stages.append(galaxy.plan_stage(
                    group.sky_origin(), driver, group.cosmology()))
        self._stages = tuple(stages)
        self._lock = threading.RLock()
        self._init_recordings()
        # The cache of the model of each observation (see model_d)
        self._model_caches = tuple(
            self._model_cache(i) for i in range(len(observations)))
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
        self._h_model_data: list[HostModelData] = [
            dict() for _ in observations]
        self._d_model_data: list[ModelData] = [dict() for _ in observations]
        # The pieces of the models linear in amplitudes of the light, of
        # each observation that has some (see amplitude_pieces)
        self._pieces: dict[int, LinearPieces] = {}
        # The times of the steps of its evaluations (and of those of the
        # objectives of it)
        self._timers = timeutils.Timers()

    def _init_recordings(self) -> None:
        """
        Make the recordings of orbits of the observations with data (see
        OrbitRecording), and their caches (see model_d): each takes the
        state of its spheroid (its orbits).
        """
        self._recordings = orbit_recordings(
            self._group, self._galaxy_drivers, self._observation_drivers)
        caches = []
        for recording in self._recordings:
            g, c = recording.component()
            caches.append(StageCache(
                recording.reads(), (self._stages[g].state_cache(c),)))
        self._recording_caches: tuple[StageCache[OrbitSeries], ...] = \
            tuple(caches)
        # The recording that models each observation of one, by index
        self._recording_of = {
            i: k for k, recording in enumerate(self._recordings)
            for i in recording.observations()}

    def _model_cache(self, index: int) -> StageCache[ModelData]:
        """
        Return a cache of the model of the observation of the given index
        (see model_d): of those of its parameters that its model reads
        (see ObservationGroup.model_params), or of the weights of the
        orbits of the recording that models it (see orbit_series), which
        it takes.
        """
        k = self._recording_of.get(index)
        if k is None:
            return StageCache(self._group.model_params(index))
        recording = self._recordings[k]
        return StageCache(
            (recording.weights(),), (self._recording_caches[k],))

    def __reduce__(self) -> tuple[type, tuple]:
        # (pickled copies make their memory, lock and threads again)
        return GroupPlan, (
            self._group, self._galaxy_drivers, self._observation_drivers)

    def __copy__(self) -> Self:
        # (shallow copies share them, see with_data)
        copied = object.__new__(type(self))
        copied.__dict__.update(self.__dict__)
        return copied

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

    def with_data(self, datasets: Mapping[int, Dataset]) -> Self:
        """
        Return a plan of the group with other data of some of its
        observations (see ObservationGroup.with_data), on the same
        drivers.

        The plans of the observations take little from their data (their
        unit and the floating type of their models, which the data can
        choose, or e.g. the radii of a profile; see Observation.same_plan):
        the copy shares the plans that are also those of its observations
        (e.g. all of them, in a bootstrap), and with them
        the memory of their models, on their devices, and its lock (see
        lock): evaluated at the same time (e.g. in two threads), they wait
        for each other (plans on other drivers do not, see
        ObservationGroup.plan). They share their last evaluations too (see
        model_d), but those of the observations that its recordings of
        orbits model (see orbit_series): the copy has recordings of its
        own, in the series of its data.

        Parameters
        ----------
        datasets : Mapping of int to Dataset
            The data of each observation that takes some, by index; the
            others keep theirs, or have none.

        Returns
        -------
        GroupPlan
            The plan.

        Raises
        ------
        ValueError
            If an index is not that of an observation, or a dataset does
            not suit its observation.
        """
        group = self._group.with_data(datasets)
        olds, news = self._group.observations(), group.observations()
        copied = copy.copy(self)
        copied._group = group
        copied._init_recordings()
        plans = []
        caches = []
        for i, (plan, cache, new, old, driver) in enumerate(zip(
                self._plans, self._model_caches, news, olds,
                self._observation_drivers)):
            if not old.same_plan(new):
                with config_path('observations', i):
                    plan = new.plan(driver, group.seen_by(i))
                cache = copied._model_cache(i)
            elif i in copied._recording_of:
                cache = copied._model_cache(i)
            plans.append(plan)
            caches.append(cache)
        copied._plans = tuple(plans)
        copied._model_caches = tuple(caches)
        copied._init_evaluations()
        return copied

    def functions(self) -> ParamFunctions:
        """
        Return the functions of the galaxies, for the derived values of
        the parameters (see ParamSpace.derived_values): those of each
        galaxy (see Galaxy.plan_functions), on the driver of its stage,
        prefixed as its parameters (e.g. 'north_vcirc'), and the
        gravitational constant G ((km/s)^2 kpc / Msun). They are planned
        at their first use, with their own stages and plans, apart from
        those of its evaluations (which may run at the same time).
        """
        return self._functions

    def _galaxy_states(
            self,
            params: ParamValues,
            out_extra: Extras | None
    ) -> list[GalaxyState]:
        """
        Evaluate the stage of each galaxy: its mass, and what it gives its
        components (see GalaxyStage.evaluate), each part of it that of its
        last evaluation if what it reads is the same; its extra outputs go
        to out_extra, if given, with the label of the galaxy as the id
        'galaxy'.
        """
        group = self._group
        return [
            stage.evaluate(
                {name: params[full] for name, full in mapping.items()},
                None if out_extra is None
                else out_extra.scoped(galaxy=group.galaxy_label(g)))
            for g, (stage, mapping) in enumerate(
                zip(self._stages, group.mappings()))]

    def galaxy_states(self, params: ParamValues) -> list[GalaxyState]:
        """
        Return the state of each galaxy in an evaluation (see
        GalaxyStage.evaluate), as model_d evaluates it: each part of it
        that of its last evaluation if what it reads is the same.
        """
        with self._lock:
            return self._galaxy_states(params, None)

    def stage_caches(self) -> dict[str, StageCache[Any]]:
        """
        Return the caches of the stages of its evaluations (see model_d),
        by label: those of the parts of the stage of each galaxy (see
        GalaxyStage.caches), as 'galaxy.{galaxy}.{part}', which read the
        parameters of the galaxy by its own names (see
        ObservationGroup.mappings); and those of its recordings of orbits,
        as 'recording.{k}', and of the model of each observation, as
        'observation.{observation}', which read those of the group. Each
        has the names of the parameters it reads, the caches it takes and
        its version (see StageCache).
        """
        group = self._group
        caches: dict[str, StageCache[Any]] = {}
        for g, stage in enumerate(self._stages):
            caches |= {
                f'galaxy.{group.galaxy_label(g)}.{name}': cache
                for name, cache in stage.caches().items()}
        caches |= {
            f'recording.{k}': cache
            for k, cache in enumerate(self._recording_caches)}
        caches |= {
            f'observation.{group.observation_label(i)}': cache
            for i, cache in enumerate(self._model_caches)}
        return caches

    def orbit_recordings(self) -> tuple[OrbitRecording, ...]:
        """
        Return its recordings of orbits (see OrbitRecording): one for each
        spheroid whose orbits observations with data see.
        """
        return self._recordings

    def orbit_caches(self) -> tuple[StageCache[OrbitSeries], ...]:
        """
        Return the cache of each of its recordings of orbits (see
        orbit_series), in their order: of the parameters it reads (see
        OrbitRecording.reads), taking the orbits of its spheroid.
        """
        return self._recording_caches

    def orbit_series(self, params: ParamValues) -> list[OrbitSeries]:
        """
        Return the orbits of each of its recordings of orbits in the series
        of the bins of its observations (see OrbitRecording.record), as
        model_d records them (with BLAS on one thread): those of its last
        recording if the parameters it reads (see OrbitRecording.reads),
        and the orbits of its spheroid, are the same.
        The models of the observations are linear in the weights of the
        orbits there (see OrbitSeries): e.g. an objective solves the
        weights (see objective.Objective), and model_d gives the models at
        their values.
        """
        with self._lock, threadutils.single_threaded_blas():
            return self._orbit_series(
                params, self._galaxy_states(params, None))

    def _orbit_series(
            self,
            params: ParamValues,
            states: Sequence[GalaxyState]
    ) -> list[OrbitSeries]:
        """
        Return the orbits of each recording of orbits in the series of the
        bins of its observations, in the states of the galaxies: those of
        its last recording if what it reads is the same (see model_d), else
        recorded again.
        """
        with self._timers.measure('model_orbits'):
            return [
                cache.evaluate(params, functools.partial(
                    recording.record, states=states))
                for recording, cache in zip(
                    self._recordings, self._recording_caches)]

    @contextlib.contextmanager
    def borrow_plans(self) -> Iterator[tuple[ObservablePlan, ...]]:
        """
        Lend the plans of the observations (see Observation.plan), under
        its lock (see lock), to evaluate them apart from its evaluations
        (e.g. to draw their mock data, see draw_mocks): they write over
        the memory of the models, so it forgets their last evaluations
        (see model_d) when they are given back.

        Yields
        ------
        tuple of ObservablePlan
            The plan of each observation, on its driver.
        """
        with self._lock:
            try:
                yield self._plans
            finally:
                # (their evaluations overwrite the model data of the
                # observations)
                for cache in self._model_caches:
                    cache.forget()

    def close(self) -> None:
        """
        Release the memory of its recordings of orbits now, rather than when
        Python collects it: the samples of the orbits on the device (up to
        a gigabyte, see OrbitIntegrator) and the orbits of the last
        recording on the host; and that of the pieces of models (see
        amplitude_pieces). The next evaluation allocates it again.
        """
        with self._lock:
            for recording, cache in zip(
                    self._recordings, self._recording_caches):
                recording.close()
                cache.forget()
            self._pieces = {}

    def model_d(
            self,
            params: ParamValues,
            out_extra: Extras | None = None
    ) -> list[ModelData]:
        """
        Evaluate the model data of the observations, on their drivers,
        with BLAS on one thread (see threadutils.single_threaded_blas).
        Those of the observations that a recording of orbits models (see
        orbit_series) are of its orbits in the series of their data, at
        the weights of the orbits in the parameters; their extra outputs
        are those of their own plans (see OrbitMomentsPlan).

        Without extra outputs, an observation whose parameters (see
        ObservationGroup.model_params) have exactly the values of its last
        evaluation (by this plan, or by one with other data that shares
        its plans, see with_data) keeps the model data of that evaluation,
        which nothing else writes; so does each part of the stage of a
        galaxy (see GalaxyStage) whose parameters, and the parts it takes,
        are those of its last evaluation (with extra outputs too, but the
        states of the components). A recording of orbits keeps the orbits
        of its last recording while the parameters it reads (see
        OrbitRecording.reads) and the orbits of its spheroid do (with extra
        outputs too: it has none), and the model of an observation it
        models while those and the weights of the orbits do. NaN is never
        the value of the last evaluation (see StageCache).

        After the stages of the galaxies and the recordings of orbits (on
        the drivers of the galaxies), the observations on each driver are
        evaluated one after the other, and those on different drivers at
        the same time, each driver in a thread of its own (the extra
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
            series = self._orbit_series(params, states)
            self._evaluate_models(
                range(len(self._plans)), params, states, series, out_extra)
        return self._d_model_data

    def amplitude_pieces(
            self,
            params: ParamValues,
            elements: Sequence[tuple[str, int | None]],
            observations: Sequence[int]
    ) -> dict[int, LinearPieces]:
        """
        Evaluate the models of the given observations as linear in elements
        of the parameters that are amplitudes of the light they see (see
        ObservationGroup.light_amplitudes): the base, with the elements at
        0, and the piece of each, the model with it at 1 alone less the
        base (see LinearPieces). With K elements, the models of these
        observations (not those of the others) are evaluated K + 1 times,
        as model_d evaluates them, with BLAS on one thread.

        Parameters
        ----------
        params : dict
            The parameters of the group (see ObservationGroup.pdescs).
        elements : Sequence of tuple
            The elements: each the name of its parameter and its index
            (None for a scalar).
        observations : Sequence of int
            The observations, by index.

        Returns
        -------
        dict
            The pieces of the model of each observation, by index, on its
            driver; they are overwritten by the next call (hold the lock
            while reading them, see lock).

        Raises
        ------
        ValueError
            If a recording of orbits models one of the observations (see
            orbit_series): its model is not linear in the light.
        """
        if recorded := sorted(set(observations) & set(self._recording_of)):
            raise ValueError(
                f"the models of observations {recorded} are of orbits, not "
                f"linear in the light")
        base = with_elements(params, elements, 0.0)
        with (self._lock, self._timers.measure('model_pieces'),
              threadutils.single_threaded_blas()):
            for k in range(len(elements) + 1):
                values = base if k == 0 \
                    else with_elements(base, [elements[k - 1]], 1.0)
                states = self._galaxy_states(values, None)
                self._evaluate_models(observations, values, states, (), None)
                for i in observations:
                    if k == 0:
                        self._keep_base(i, len(elements))
                    else:
                        self._keep_piece(i, k - 1)
        return {i: self._pieces[i] for i in observations}

    def _keep_base(self, index: int, size: int) -> None:
        """
        Copy the model of the observation of the given index to the base of
        its pieces (see amplitude_pieces), for the given number of pieces.
        """
        driver = self._observation_drivers[index]
        model = self._d_model_data[index]
        pieces = self._pieces.get(index)
        if pieces is None or _npieces(pieces) != size:
            pieces = _alloc_pieces(driver, model, size)
            self._pieces[index] = pieces
        for key, item in model.items():
            # (each array of the item: the values, the mask and the weights)
            for array, base in zip(item, pieces.base[key]):
                if array is not None:
                    driver.mem_copy_d2d(array, base)

    def _keep_piece(self, index: int, k: int) -> None:
        """
        Make the model of the observation of the given index less the base
        its piece k (see amplitude_pieces).
        """
        driver = self._observation_drivers[index]
        pieces = self._pieces[index]
        for key, item in self._d_model_data[index].items():
            driver.math_sub(
                item.data, pieces.base[key].data, out=pieces.pieces[key][k])

    def _evaluate_models(
            self,
            indices: Iterable[int],
            params: ParamValues,
            states: Sequence[GalaxyState],
            series: Sequence[OrbitSeries],
            out_extra: Extras | None
    ) -> None:
        """
        Evaluate the models of the observations of the given indices (see
        model_d): those on each driver one after the other, and those on
        different drivers at the same time, if any (see
        _evaluate_models_at_once).
        """
        if self._executor is None:
            for i in indices:
                self._evaluate_model(i, params, states, series, out_extra)
        else:
            self._evaluate_models_at_once(
                indices, params, states, series, out_extra)

    def _evaluate_model(
            self,
            index: int,
            params: ParamValues,
            states: Sequence[GalaxyState],
            series: Sequence[OrbitSeries],
            out_extra: Extras | None
    ) -> None:
        """
        Evaluate the model of the observation of the given index, in the
        states of the galaxies and of the orbits of the recordings (see
        model_d).
        """
        group = self._group
        plan = self._plans[index]
        cache = self._model_caches[index]
        seen = [states[g] for g in group.seen_galaxies(index)]
        extra = None if out_extra is None else out_extra.scoped(
            observation=group.observation_label(index))
        k = self._recording_of.get(index)
        if k is not None and extra is not None:
            # (its extra outputs, of the orbits as they are; its model is
            # that of the recording)
            plan.evaluate(group.moved(index, params), seen, extra)

        def model(view: ParamView) -> ModelData:
            # (of the values of the parameters it reads)
            if k is None:
                return plan.evaluate(group.moved(index, view), seen, extra)
            return self._recordings[k].model_d(series[k], index, view)
        evaluate = cache.evaluate if extra is None else cache.refresh
        self._d_model_data[index] = evaluate(params, model)

    def _evaluate_models_at_once(
            self,
            indices: Iterable[int],
            params: ParamValues,
            states: Sequence[GalaxyState],
            series: Sequence[OrbitSeries],
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
        wanted = set(indices)

        def evaluate(on_driver: tuple[int, ...]) -> None:
            for i in on_driver:
                if i in wanted:
                    self._evaluate_model(
                        i, params, states, series, extras[i])
        futures = [
            self._executor.submit(
                contextvars.copy_context().run, evaluate, on_driver)
            for on_driver in self._on_drivers]
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
    ) -> list[HostModelData]:
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
        list of HostModelData
            The model data of each observation, copied to the host; they
            are overwritten by the next evaluation.
        """
        with self._lock:
            self.model_d(params, out_extra)
            self._copy_to_host()
        return self._h_model_data

    def _copy_to_host(self) -> None:
        """Copy the model data of the last evaluation to the host."""
        with self._timers.measure('model_d2h'):
            for i, driver in enumerate(self._observation_drivers):
                h_model = self._h_model_data[i]
                for key, d_item in self._d_model_data[i].items():
                    # (into the arrays of the last copy, once there is one)
                    h_item = h_model.get(key, _NO_ITEM)
                    h_model[key] = ModelItem._make(
                        None if d_array is None
                        else driver.mem_copy_d2h(d_array, h_array)
                        for d_array, h_array in zip(d_item, h_item))


def _alloc_pieces(
        driver: Driver, model: ModelData, size: int
) -> LinearPieces:
    """
    Allocate the pieces of a model of the given number of pieces (see
    GroupPlan.amplitude_pieces), on its driver: the base of each array of
    the model (its values, its mask and its weights), and the pieces of
    its values.
    """
    return LinearPieces(
        {key: ModelItem._make(
            None if array is None
            else driver.mem_alloc_d(array.shape, array.dtype)
            for array in item)
         for key, item in model.items()},
        {key: driver.mem_alloc_d((size, *item.data.shape), item.data.dtype)
         for key, item in model.items()})


def _npieces(pieces: LinearPieces) -> int:
    """Return the number of the pieces of a model."""
    return len(next(iter(pieces.pieces.values())))


class _GroupFunctions(ParamFunctions):
    """
    The functions of the galaxies of a group (see GroupPlan.functions):
    those of each galaxy, planned on the driver of its stage at their
    first binding.
    """

    def __init__(self, group: 'ObservationGroup', drivers: Sequence[Driver]):
        self._group = group
        self._drivers = tuple(drivers)
        self._functions: tuple[GalaxyFunctions, ...] | None = None
        self._lock = threading.Lock()

    def names(self) -> tuple[str, ...]:
        return self._group.function_names()

    def bind(self, values: ParamValues) -> dict[str, Any]:
        group = self._group
        with self._lock:
            if self._functions is None:
                self._functions = tuple(
                    galaxy.plan_functions(
                        group.sky_origin(), driver, group.cosmology())
                    for galaxy, driver in zip(
                        group.galaxies(), self._drivers))
        bound: dict[str, Any] = dict(G=G)
        for functions, mapping, prefix in zip(
                self._functions, group.mappings(), group.prefixes()):
            own = functions.bind(
                {name: values[full] for name, full in mapping.items()})
            bound |= {f'{prefix}{name}': value for name, value in own.items()}
        return bound
