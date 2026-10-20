import copy
from collections.abc import Sequence
from typing import Any, NamedTuple, Self

import numpy as np

from gbkfit.dataset import Data, Dataset
from gbkfit.driver import Driver
from gbkfit.model import Galaxy
from gbkfit.model.components._functions import GalaxyFunctions
from gbkfit.model.components._units import DataUnit
from gbkfit.model.mass.units import G
from gbkfit.params import (
    ParamDesc, ParamFunctions, ParamScalarDesc, ParamValues)
from gbkfit.utils import iterutils, parseutils, threadutils, timeutils
from gbkfit.utils.parseutils import ConfigError
from ._scene import DepthUnit, Scene, make_scene
from .noise import Noise
from .observables import HostModelData, ModelData
from .observation import Observation


__all__ = [
    'DepthGroup',
    'ObservationGroup'
]


class DepthGroup(NamedTuple):
    """
    Galaxies that can pass through each other along the line of sight
    (e.g. a merger), by name, which radiative transfer evaluates together
    (see TransferPlan), and the name of the group (which prefixes its
    extra outputs). Each galaxy after the first has a parameter of its
    position along the line of sight from the first, '{galaxy}_separation'
    (arcsec; positive behind it, away from the viewer).
    """
    name: str
    galaxies: tuple[str, ...]


def _same_unit(a: DataUnit | None, b: DataUnit | None) -> bool:
    """
    Return whether two units of the light of data are the same (see
    DataUnit, whose beams can be arrays, one for each spectral point).
    """
    if a is None or b is None:
        return a is b
    return all(np.array_equal(x, y) if isinstance(x, np.ndarray)
               else x == y for x, y in zip(a, b))


class ObservationGroup:
    """
    Galaxies and the observations of them, evaluated together.

    Each observation names the galaxies it sees, which it may leave out
    when there is one galaxy; the light of several galaxies adds up, or
    through radiative transfer, is dimmed by the opacity of the galaxies
    in front (see depth_order). The parameters and constants of the
    galaxies are prefixed by their names when they have names (which they
    need when there are several; e.g. 'north_'). The plans of the
    observations are made here, once (copies of the group with other data
    share them; see with_data). The gravity of each galaxy is
    evaluated once for all the observations of it (see
    Galaxy.plan_gravity).

    Parameters
    ----------
    galaxies : Galaxy or Sequence of Galaxy
        The galaxies.
    observations : Observation or Sequence of Observation
        The observations.
    depth_order : Sequence of str or DepthGroup, optional
        The galaxies along the line of sight, front to back, by name, or
        in groups of galaxies that can pass through each other (see
        DepthGroup): an observation of several galaxies through radiative
        transfer (see Transfer) marches them in this order, which it needs
        when one of them has opacity (the light of each is dimmed by the
        opacity of those in front of it), and then of all of them. In
        configurations, a group is {name: [galaxies]}.

    Raises
    ------
    ConfigError
        If there are no galaxies or no observations, several galaxies
        without names, names repeat, an observation does not name the
        galaxies it sees (when there are several) or names unknown ones,
        depth_order names unknown galaxies or repeats them, has a group of
        fewer than two galaxies, or of the name of a galaxy or an
        observation, or what an observation sees does not go with it (see
        make_scene).
    """

    def __init__(
            self,
            galaxies: Galaxy | Sequence[Galaxy],
            observations: Observation | Sequence[Observation],
            depth_order: Sequence[str | DepthGroup] | None = None
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
        # (as given, for copies; see with_driver)
        self._given_depth_order = \
            None if depth_order is None else tuple(depth_order)
        self._depth_order = _depth_order(
            depth_order, galaxy_names, observation_names)
        self._pdescs, self._mappings = iterutils.merge_with_prefixes(
            [galaxy.pdescs() for galaxy in self._galaxies], self._prefixes)
        # The separations of the galaxies of groups, by galaxy
        self._separations = {
            g: f'{self._prefixes[g]}separation'
            for name, members in self._depth_order or ()
            if name is not None for g in members[1:]}
        if taken := sorted(
                set(self._separations.values()) & self._pdescs.keys()):
            raise ConfigError(
                f"the parameters of the separations of galaxies have the "
                f"names of parameters of the galaxies: {taken}; rename their "
                f"components")
        self._pdescs |= {
            separation: ParamScalarDesc(
                desc="the position of the galaxy along the line of sight "
                     "from the first of its group (arcsec, away from the "
                     "viewer)")
            for separation in self._separations.values()}
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
        self._init_evaluations()

    def _init_evaluations(self) -> None:
        """
        Make what its evaluations write: the model data of each observation
        on the host and the device, and the times of the steps.
        """
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
        the given indices, in the depth order of those of them in it (a
        group of which it sees one galaxy is that galaxy).
        """
        order = None
        if self._depth_order is not None:
            order = []
            for name, members in self._depth_order:
                visible = [g for g in members if g in seen]
                galaxies = tuple(seen.index(g) for g in visible)
                if len(visible) == 1:
                    order.append(DepthUnit(galaxies))
                elif visible:
                    order.append(DepthUnit(galaxies, name, tuple(
                        self._separations.get(g) for g in visible)))
        return make_scene(
            [self._galaxies[j] for j in seen],
            [self._mappings[j] for j in seen],
            [self._prefixes[j] for j in seen],
            observation.components(), observation.lines(),
            observation.band(), observation.line_of_sight(),
            observation.observable().emission, order,
            observation.data_unit())

    def with_driver(self, driver: Driver) -> Self:
        """
        Return a group of the same galaxies and observations, all on
        another driver (see Galaxy.with_driver and
        Observation.with_driver), with its own plans on it.
        """
        return type(self)(
            [galaxy.with_driver(driver) for galaxy in self._galaxies],
            [obs.with_driver(driver) for obs in self._observations],
            self._given_depth_order)

    def with_data(self, datasets: Sequence[Dataset]) -> Self:
        """
        Return a group of the same galaxies and observations, with other
        data: one dataset for each observation (e.g. those of a bootstrap;
        see Observation.with_data), under its likelihood.

        The plans of the observations do not depend on the values of their
        data, only on their unit (see Observation.data_unit): the copy
        shares the plans of those whose data keep their unit (e.g. all of
        them, in a bootstrap), and with them the memory of their models,
        on their devices. So the two must not be evaluated at the same
        time (e.g. in two threads; copies on other drivers can, see
        with_driver). A copy whose observations have other parameters
        (e.g. a likelihood of data that they had none of) has plans of its
        own.

        Raises
        ------
        ConfigError
            If there is not one dataset per observation, or a dataset does
            not suit its observation.
        """
        if len(datasets) != len(self._observations):
            raise ConfigError(
                f"there are {len(self._observations)} observations, but "
                f"{len(datasets)} datasets")
        observations = [
            obs.with_data(data)
            for obs, data in zip(self._observations, datasets)]
        if any(new.pdescs().keys() != old.pdescs().keys()
               for new, old in zip(observations, self._observations)):
            return type(self)(
                self._galaxies, observations, self._given_depth_order)
        group = copy.copy(self)
        group._observations = tuple(observations)
        group._plans = tuple(
            plan if _same_unit(new.data_unit(), old.data_unit())
            else new.plan(self._scene(seen, new))
            for plan, seen, new, old in zip(
                self._plans, self._seen_galaxies, observations,
                self._observations))
        group._init_evaluations()
        return group

    def galaxies(self) -> tuple[Galaxy, ...]:
        """Return the galaxies."""
        return self._galaxies

    def observations(self) -> tuple[Observation, ...]:
        """Return the observations."""
        return self._observations

    def nobservations(self) -> int:
        """Return the number of observations."""
        return len(self._observations)

    def extra_prefix(self, index: int) -> str:
        """
        Return the prefix of the extra outputs of the observation of the
        given index: its name, or 'observation{index}_'.
        """
        return self._extra_prefixes[index]

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

    def plan_functions(self) -> ParamFunctions:
        """
        Plan the functions of the galaxies, for the derived values of the
        parameters (see ParamSpace): those of each galaxy (see
        Galaxy.plan_functions), prefixed as its parameters (e.g.
        'north_vcirc'), and the gravitational constant G ((km/s)^2 kpc /
        Msun). They have their own plans, apart from those of the
        evaluations of the group.

        Returns
        -------
        ParamFunctions
            The functions.
        """
        return _GroupFunctions(
            [galaxy.plan_functions() for galaxy in self._galaxies],
            self._mappings, self._prefixes)

    def _gravity(
            self,
            params: ParamValues,
            out_extra: dict[str, Any] | None
    ) -> list[dict[int, ParamValues]]:
        """
        Evaluate what the gravity of each galaxy gives its components (see
        GravityPlan.evaluate); its extra outputs go to out_extra, if
        given, prefixed by the prefix of the galaxy.
        """
        gravity = []
        for plan, mapping, prefix in zip(
                self._gravity_plans, self._mappings, self._prefixes):
            extra = {} if out_extra is not None else None
            gravity.append(plan.evaluate(
                {name: params[full] for name, full in mapping.items()},
                extra))
            if out_extra is not None:
                out_extra |= {
                    f'{prefix}{key}': value for key, value in extra.items()}
        return gravity

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
            observation prefixed by its name, or 'observation{i}_', and
            those of the gravity of each galaxy (its rotation tables, see
            GravityPlan.evaluate) by the prefix of the galaxy.

        Returns
        -------
        list of ModelData
            The model data of each observation, on its driver; they are
            overwritten by the next evaluation.
        """
        with (self._timers.measure('model_eval'),
              threadutils.single_threaded_blas()):
            with self._timers.measure('model_gravity'):
                gravity = self._gravity(params, out_extra)
            for i, plan in enumerate(self._plans):
                out_extra_i = {} if out_extra is not None else None
                self._d_model_data[i] = plan.evaluate(
                    params, [gravity[g] for g in self._seen_galaxies[i]],
                    out_extra_i)
                if out_extra is not None:
                    for key, val in out_extra_i.items():
                        out_extra[f'{self.extra_prefix(i)}{key}'] = val
        return self._d_model_data

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
        spectra) have its dispersion as their errors; those measured from
        them (e.g. moments) the scatter of their values over nsamples
        other draws (the standard deviation, of the values that are
        finite in at least two draws; the others are missing).

        Parameters
        ----------
        params : dict
            The parameters of the galaxies, prefixed (see pdescs).
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
        noises = list(noises) if isinstance(noises, Sequence) \
            else [noises] * len(self._observations)
        if len(noises) != len(self._observations):
            raise ConfigError(
                f"there are {len(self._observations)} observations, but "
                f"{len(noises)} noises")
        if nmocks < 1:
            raise ConfigError(f"nmocks must be at least 1; it is {nmocks}")
        gravity = self._gravity(params, None)
        streams = np.random.SeedSequence(seed).spawn(len(self._observations))
        mocks = []
        for i, (obs, plan, noise, stream) in enumerate(zip(
                self._observations, self._plans, noises, streams)):
            with parseutils.config_path('observations', i):
                errors = plan.exact_errors(noise)
                if errors is None and nsamples < 2:
                    raise ConfigError(
                        f"the errors of measured data are the scatter of "
                        f"nsamples draws, at least 2; nsamples is {nsamples}")
                draws = plan.mock(
                    params, [gravity[g] for g in self._seen_galaxies[i]],
                    noise, np.random.default_rng(stream),
                    nmocks + (nsamples if errors is None else 0))
            if errors is None:
                errors = _scatter(draws[nmocks:])
            observable = obs.observable()
            mocks.append([
                observable.dataset({
                    key: Data(item['d'], error=errors[key])
                    for key, item in draw.items()})
                for draw in draws[:nmocks]])
        return mocks

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


class _GroupFunctions(ParamFunctions):
    """
    The functions of the galaxies of a group (see
    ObservationGroup.plan_functions): those of each galaxy, the mapping of
    its parameters to those of the group, and its prefix.
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


def _depth_order(
        order: Sequence[str | DepthGroup] | None,
        galaxies: Sequence[str | None],
        observations: Sequence[str | None]
) -> tuple[tuple[str | None, tuple[int, ...]], ...] | None:
    """
    Return the items of a depth order (see ObservationGroup: depth_order)
    as the name of each group (None for a galaxy) and the indices of its
    galaxies, among the names of the galaxies. Raise ConfigError if a name
    is unknown or repeats, or a group has fewer than two galaxies or the
    name of a galaxy or an observation.
    """
    if order is None:
        return None
    items = []
    for item in order:
        if isinstance(item, DepthGroup):
            parseutils.check_name(item.name)
            if len(item.galaxies) < 2:
                raise ConfigError(
                    f"the group {item.name!r} of depth_order needs at least "
                    f"two galaxies; it has {list(item.galaxies)}")
            if item.name in [*galaxies, *observations]:
                raise ConfigError(
                    f"the group {item.name!r} of depth_order has the name of "
                    f"a galaxy or an observation")
            items.append((item.name, tuple(item.galaxies)))
        else:
            items.append((None, (item,)))
    names = [name for _, members in items for name in members]
    if unknown := [name for name in names if name not in galaxies]:
        raise ConfigError(
            f"depth_order names unknown galaxies {unknown}; the galaxies are "
            f"{list(galaxies)}")
    if len(set(names)) != len(names):
        raise ConfigError(f"depth_order repeats galaxies: {names}")
    return tuple(
        (name, tuple(galaxies.index(galaxy) for galaxy in members))
        for name, members in items)
