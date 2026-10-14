from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from gbkfit.model import Galaxy
from gbkfit.params import ParamDesc, ParamScalarDesc, ParamValues
from gbkfit.utils import iterutils, parseutils, threadutils, timeutils
from gbkfit.utils.parseutils import ConfigError
from ._scene import DepthUnit, Scene, make_scene
from .observables import ModelData
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


class ObservationGroup:
    """
    Galaxies and the observations of them, evaluated together.

    Each observation names the galaxies it sees, which it may leave out
    when there is one galaxy; the light of several galaxies adds up, or
    through radiative transfer, is dimmed by the opacity of the galaxies
    in front (see depth_order). The parameters and constants of the
    galaxies are prefixed by their names when they have names (which they
    need when there are several; e.g. 'north_'). The plans of the
    observations are made here, once. The gravity of each galaxy is
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
                gravity = []
                for plan, mapping, prefix in zip(
                        self._gravity_plans, self._mappings, self._prefixes):
                    extra = {} if out_extra is not None else None
                    gravity.append(plan.evaluate(
                        {name: params[full] for name, full in mapping.items()},
                        extra))
                    if out_extra is not None:
                        out_extra |= {
                            f'{prefix}{key}': value
                            for key, value in extra.items()}
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
