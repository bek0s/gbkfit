"""
Groups of observations: galaxies and the observations of them,
evaluated together (see ObservationGroup), and the galaxies that pass
through each other along the line of sight (see DepthGroup).
"""

import copy
import logging
from collections.abc import Collection, Mapping, Sequence
from typing import Any, NamedTuple, Self


from gbkfit.config import (
    config_path, item_prefixes)
from gbkfit.cosmology import DEFAULT, Cosmology
from gbkfit.dataset import Dataset
from gbkfit.driver import Driver
from gbkfit.model import Galaxy
from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import iterutils
from ._depth_order import DepthGroup, DepthOrder
from ._frames import (
    FrameShift, alignments, check_offsets, first_on_sky, frame_shift,
    warn_frames)
from .group_plan import GroupPlan, place_drivers
from .observation import Observation, SeenGalaxies


__all__ = [
    'DepthGroup',
    'ObservationGroup'
]


_log = logging.getLogger(__name__)


class _Member(NamedTuple):
    """
    An observation of a group, with what the group keeps of it, made once
    (see ObservationGroup): its label (the id 'observation' of its extra
    outputs; see ObservationGroup.observation_label), the indices of the
    galaxies it sees, those galaxies as it sees them (see
    ObservationGroup.seen_by), and how it moves them into its frame.
    """
    observation: Observation
    label: str
    seen: tuple[int, ...]
    galaxies: SeenGalaxies
    shift: FrameShift


class ObservationGroup:
    """
    Galaxies and the observations of them, evaluated together.

    Each observation names the galaxies it sees, which it may leave out
    when there is one galaxy; the light of several galaxies adds up, or
    through radiative transfer, is dimmed by the opacity of the galaxies
    in front (see depth_order). The parameters and constants of the
    galaxies are prefixed by their names when they have names (which they
    need when there are several; e.g. 'north_').

    The centres of the components (xpos, ypos) are measured from the
    origin of the frame of the first observation on the sky (see
    Observation.sky_frame): its reference pixel, at the RA and Dec of its
    grid (see ObservationSky.sky_rval). The other observations on the sky
    see them moved by where that origin is in their frames, from their
    RA and Dec: e.g. a cube whose reference pixel is 2 arcsec west of that
    of the first sees them at xpos - 2 in its frame (2 arcsec further east
    of its origin), so that the data of several instruments meet on the
    sky as their coordinates say. (The frames are aligned by that shift
    alone: those of reference pixels far apart near a pole also turn, by
    about their separation times tan(Dec), which is warned of; e.g. 0.04
    deg for 1 arcmin at Dec -66.5 deg.) An observation with offsets of
    its frame (see ObservationSky) sees the galaxies moved by them too,
    for what the coordinates do not know (e.g. the errors of their
    astrometry); each galaxy seen on the sky needs an observation on the
    sky without them. The centres of what an observation adds in the
    frame of the model (e.g. the masses of its lens, see
    SkyFrame.centre_params) are moved with those of the components; the
    shifts of what it adds that is placed by its RA and Dec (e.g. shifted
    deflection maps, see SkyFrame.shift_params), by its offsets alone.
    The observations that are not on the sky see the parameters as they
    are.

    The group describes the galaxies and the observations; its
    evaluation is planned on devices by plan (see GroupPlan), where the
    stage of each galaxy is evaluated once for all the observations of
    it. An objective compares the models of the observations with data
    with them (see Objective); those of the others are evaluated by who
    wants them (e.g. a prediction of a band without data, see the outputs
    of a setup).

    Options of setups that are given by observation (e.g. their
    likelihoods, see Likelihoods, or the noise of their mock data, see
    MockOptions) are one value for all of them, or values by observation
    name (see per_observation).

    Parameters
    ----------
    galaxies : Galaxy or Sequence of Galaxy
        The galaxies.
    observations : Observation or Sequence of Observation
        The observations.
    depth_order : Sequence of str or DepthGroup, optional
        The galaxies along the line of sight, front to back, by name, or
        in groups of galaxies that can pass through each other (see
        DepthGroup): an observation of several galaxies of which one has
        opacity marches them in this order by radiative transfer (the
        light of each is dimmed by the opacity of those in front of it;
        see Scene.evaluation), and needs it then of all of them. In
        configurations, a group is {name: [galaxies]}.
    cosmology : Cosmology, optional
        The cosmology of the galaxies, that of their setup (see
        Galaxy.plan_stage); Planck18 by default (see
        gbkfit.cosmology.DEFAULT).

    Raises
    ------
    ValueError
        If there are no galaxies or no observations, several galaxies
        without names, names repeat, an observation does not name the
        galaxies it sees (when there are several) or names unknown ones,
        depth_order names unknown galaxies or repeats them, has a group of
        fewer than two galaxies, or of the name of a galaxy or an
        observation, every observation of a galaxy has offsets of position
        or of velocity.
    """

    def __init__(
            self,
            galaxies: Galaxy | Sequence[Galaxy],
            observations: Observation | Sequence[Observation],
            depth_order: Sequence[str | DepthGroup] | None = None,
            cosmology: Cosmology | None = None
    ):
        self._galaxies = iterutils.tuplify(galaxies)
        observations = iterutils.tuplify(observations)
        if not self._galaxies:
            raise ValueError("at least one galaxy is required")
        if not observations:
            raise ValueError("at least one observation is required")
        galaxy_names = [galaxy.name() for galaxy in self._galaxies]
        if len(galaxy_names) > 1 and None in galaxy_names:
            raise ValueError(
                "the galaxies need names when there are several")
        observation_names = [obs.name() for obs in observations]
        self._prefixes = item_prefixes(
            galaxy_names, 'galaxies', 'galaxy', False)
        item_prefixes(
            observation_names, 'observations', 'observation', False)
        repeated = sorted(
            set(galaxy_names) & set(observation_names) - {None})
        if repeated:
            raise ValueError(
                f"the galaxies and the observations must have different "
                f"names; repeated: {repeated}")
        self._cosmology = DEFAULT if cosmology is None else cosmology
        self._depth = None if depth_order is None else DepthOrder(
            depth_order, galaxy_names, observation_names, self._prefixes)
        # The parameters of the galaxies (and of their separations), to
        # which those of the observations are added
        galaxy_pdescs, self._mappings = iterutils.merge_with_prefixes(
            [galaxy.pdescs() for galaxy in self._galaxies], self._prefixes)
        # The separations of the galaxies of groups, by galaxy
        separations = {} if self._depth is None else self._depth.pdescs()
        self._separations = {} if self._depth is None \
            else self._depth.separations()
        if taken := sorted(separations.keys() & galaxy_pdescs.keys()):
            raise ValueError(
                f"the parameters of the separations of galaxies have the "
                f"names of parameters of the galaxies: {taken}; rename their "
                f"components")
        galaxy_pdescs |= separations
        self._pdescs = _with_observation_pdescs(
            galaxy_pdescs, observations)
        self._constants, _ = iterutils.merge_with_prefixes(
            [galaxy.constants() for galaxy in self._galaxies], self._prefixes)
        # The ids of the galaxies and of the observations in the extra
        # outputs: their names, or their positions if they have none (e.g.
        # 'galaxy0', 'observation0')
        self._galaxy_labels = [
            f'galaxy{i}' if name is None else name
            for i, name in enumerate(galaxy_names)]
        labels = [
            f'observation{i}' if name is None else name
            for i, name in enumerate(observation_names)]
        # (the origin of the frame of the parameters of the galaxies)
        frames = [observation.sky_frame() for observation in observations]
        first = first_on_sky(frames)
        self._sky_origin = None if first is None else frames[first].origin
        seen = []
        for i, observation in enumerate(observations):
            with config_path('observations', i):
                seen.append(tuple(self._seen(i, observation)))
        self._members = tuple(
            _Member(
                observation, label, indices, self._galaxies_seen(indices),
                frame_shift(
                    self._galaxies, self._mappings, indices, frame,
                    alignment))
            for observation, label, indices, frame, alignment in zip(
                observations, labels, seen, frames, alignments(frames)))
        check_offsets(self._galaxies, frames, seen)
        warn_frames(self._galaxies, frames, seen)
        _warn_unused_masses(self._galaxies, observations, seen)
        self._check_observations()

    def _check_observations(self) -> None:
        """
        Check that each observation can see the galaxies it sees (see
        Observation.check): every rule about what each sees, here, without
        devices, so that the plans fail only for resources.
        """
        for i, member in enumerate(self._members):
            with config_path('observations', i):
                member.observation.check(member.galaxies)

    def _seen(self, i: int, observation: Observation) -> list[int]:
        """Return the indices of the galaxies that observation i sees."""
        names = observation.galaxies()
        if names is None:
            if len(self._galaxies) > 1:
                raise ValueError(
                    f"observation {i} must name the galaxies it sees; there "
                    f"are {len(self._galaxies)} galaxies")
            return [0]
        known = [galaxy.name() for galaxy in self._galaxies]
        if unknown := [name for name in names if name not in known]:
            raise ValueError(
                f"observation {i} refers to unknown galaxies {unknown}; the "
                f"galaxies are {known}")
        return sorted(known.index(name) for name in names)

    def _galaxies_seen(self, seen: tuple[int, ...]) -> SeenGalaxies:
        """
        Return the galaxies of the given indices as an observation that
        sees them sees them (see seen_by): in the depth order of those of
        them in it (a group of which it sees one galaxy is that galaxy).
        """
        order = None if self._depth is None else self._depth.seen(seen)
        return SeenGalaxies(
            tuple(self._galaxies[j] for j in seen),
            tuple(self._mappings[j] for j in seen),
            tuple(self._galaxy_labels[j] for j in seen),
            order, self._sky_origin,
            self._cosmology)

    def observed(self) -> tuple[int, ...]:
        """
        Return the observations with data, by index: those whose models an
        objective compares with their data (see Objective).
        """
        return tuple(
            i for i, member in enumerate(self._members)
            if member.observation.dataset() is not None)

    def per_observation[T](
            self,
            values: T | Mapping[str, T],
            option: str,
            indices: Collection[int] | None = None,
            complete: bool = True
    ) -> dict[int, T]:
        """
        Return the values of an option of setups that is given by
        observation (e.g. the noise of mock data, see MockOptions), by the
        index of the observation: one value for all the observations, or
        values by observation name (a Mapping).

        Values by name can be of any observation (they have names, or none
        has; see the class). Each of the observations of the given indices
        then needs a name (a position, the label of an observation without
        one, would change with the order of the observations), and a value
        unless they need not be complete.

        Parameters
        ----------
        values : Any or Mapping of str to Any
            One value for all, or values by observation name.
        option : str
            The name of the option, for the errors.
        indices : Collection of int, optional
            The observations that need a value (or a name), by index; all
            by default.
        complete : bool, optional
            Whether each of those needs a value by name.

        Returns
        -------
        dict
            The values, by index: of every observation for one value, or
            else of those named.

        Raises
        ------
        ValueError
            If the values by name name unknown observations, or an
            observation of the given indices has no name, or no value
            while they need to be complete.
        """
        observations = self.observations()
        if not isinstance(values, Mapping):
            return dict.fromkeys(range(len(observations)), values)
        indices = range(len(observations)) if indices is None else indices
        if unnamed := [i for i in indices if observations[i].name() is None]:
            raise ValueError(
                f"option '{option}' is given by observation name, but "
                f"observations {unnamed} have no name: name them, or give "
                f"one value for all")
        names = {
            observation.name(): i
            for i, observation in enumerate(observations)
            if observation.name() is not None}
        if unknown := sorted(set(values) - set(names)):
            raise ValueError(
                f"option '{option}' names unknown observations {unknown}; "
                f"those with names are {list(names)}")
        if complete and (missing := [
                observations[i].name() for i in indices
                if observations[i].name() not in values]):
            raise ValueError(
                f"option '{option}' is given by observation name, and needs "
                f"a value for each observation; these have none: {missing}")
        return {names[name]: value for name, value in values.items()}

    def seen_by(self, index: int) -> SeenGalaxies:
        """
        Return the galaxies that the observation of the given index sees,
        as it sees them: in the depth order of those of them in it (a
        group of which it sees one galaxy is that galaxy).
        """
        return self._members[index].galaxies

    def plan(
            self,
            driver: Driver | None = None,
            placement: Mapping[str, Driver] | None = None
    ) -> 'GroupPlan':
        """
        Plan the evaluation of the group on devices (see GroupPlan): the
        stage of each galaxy and the model of each observation, each on
        its driver (see place_drivers).

        Parameters
        ----------
        driver : Driver, optional
            The driver of the galaxies and the observations; the host by
            default.
        placement : Mapping of str to Driver, optional
            Other drivers of some of them, by their labels (see
            galaxy_label and observation_label): e.g. {'halpha': gpu0,
            'hi': gpu1}.

        Returns
        -------
        GroupPlan
            The plan.

        Raises
        ------
        ValueError
            If the placement names unknown galaxies or observations, or
            what an observation sees does not go with it (see
            Observation.plan), or a galaxy cannot plan its stage (see
            Galaxy.plan_stage).
        """
        return GroupPlan(self, *place_drivers(self, driver, placement))

    def follows_observations(self, index: int) -> bool:
        """
        Check whether the galaxy of the given index follows the
        observations that see it onto their driver, unless placed (see
        drivers): whether it has no mass, so that its stage is for them
        alone. (The functions of a galaxy with mass, of the derived
        values, run on its own driver; see GroupPlan.functions.)
        """
        return self._galaxies[index].mass_model() is None

    def with_datasets(self, datasets: Mapping[int, Dataset]) -> Self:
        """
        Return a copy of the group with other data of some of its
        observations (e.g. those of a bootstrap, or mock data; see
        Observation.with_dataset): the others keep theirs, or have none. Its
        parameters are those of this one (the data do not change them).

        Parameters
        ----------
        datasets : Mapping of int to Dataset
            The data of each observation that takes some, by index.

        Returns
        -------
        ObservationGroup
            The group.

        Raises
        ------
        ValueError
            If an index is not that of an observation, a dataset does not
            suit its observation, or an observation with its new data
            cannot see its galaxies (see Observation.check).
        """
        if unknown := sorted(set(datasets) - set(range(len(self._members)))):
            raise ValueError(
                f"there are {len(self._members)} observations; there are no "
                f"observations {unknown}")
        group = copy.copy(self)
        group._members = tuple(
            member if i not in datasets else member._replace(
                observation=member.observation.with_dataset(datasets[i]))
            for i, member in enumerate(self._members))
        # (what an observation can see may depend on its data: e.g. the
        # radii of a profile are those of its data, and radii in angles
        # need a galaxy with a distance)
        group._check_observations()
        return group

    def galaxies(self) -> tuple[Galaxy, ...]:
        """Return the galaxies."""
        return self._galaxies

    def observations(self) -> tuple[Observation, ...]:
        """Return the observations."""
        return tuple(member.observation for member in self._members)

    def nobservations(self) -> int:
        """Return the number of observations."""
        return len(self._members)

    def observation_label(self, index: int) -> str:
        """
        Return the id of the observation of the given index in the extra
        outputs (and in the outputs of the tasks): its name, or
        'observation{index}'.
        """
        return self._members[index].label

    def pdescs(self) -> dict[str, ParamDesc]:
        """
        Return the parameters of the galaxies, prefixed, and those of the
        observations (see Observation.pdescs): those of their models (the
        likelihoods, which compare them with the data, add their own; see
        Likelihoods.pdescs).
        """
        return self._pdescs

    def constants(self) -> dict[str, Any]:
        """
        Return the values that parameter expressions can use: those of
        the galaxies (see Galaxy.constants), prefixed as their parameters,
        and the Hubble constant of their cosmology in units of 100 km/s/Mpc
        ('cosmology_h'; e.g. for the relations of the concentrations of
        halos, see gbkfit.relations).
        """
        return self._constants | dict(cosmology_h=self._cosmology.h())

    def function_names(self) -> tuple[str, ...]:
        """
        Return the names of the functions of the galaxies that the derived
        values of the parameters can use (see GroupPlan.functions),
        without planning them: those of each galaxy (see
        Galaxy.function_names), prefixed as its parameters, and the
        gravitational constant G.
        """
        return ('G',) + tuple(
            f'{prefix}{name}'
            for galaxy, prefix in zip(self._galaxies, self._prefixes)
            for name in galaxy.function_names())

    def galaxy_label(self, index: int) -> str:
        """
        Return the id of the galaxy of the given index in the extra
        outputs: its name, or 'galaxy{index}'.
        """
        return self._galaxy_labels[index]

    def mappings(self) -> tuple[dict[str, str], ...]:
        """
        Return the names of the parameters of each galaxy among those of
        the group, by its own names.
        """
        return tuple(self._mappings)

    def prefixes(self) -> tuple[str, ...]:
        """
        Return the prefix of the parameters of each galaxy (its name and
        '_', or none for one galaxy without a name).
        """
        return tuple(self._prefixes)

    def cosmology(self) -> Cosmology:
        """Return the cosmology of the galaxies."""
        return self._cosmology

    def sky_origin(self) -> tuple[float, float] | None:
        """
        Return the RA and Dec (degrees) of the origin of the frame of the
        parameters of the galaxies: that of the first observation on the
        sky (see the class); None if none is.
        """
        return self._sky_origin

    def seen_galaxies(self, index: int) -> tuple[int, ...]:
        """
        Return the indices of the galaxies that the observation of the
        given index sees.
        """
        return self._members[index].seen

    def moved(self, index: int, params: ParamValues) -> ParamValues:
        """
        Return the parameters as the observation of the given index sees
        them: the centres and systemic velocities of what it sees moved
        into its frame (see the class).
        """
        return self._members[index].shift.moved(params)

    def frame_shifts(
            self, index: int, params: ParamValues
    ) -> tuple[float, float, float]:
        """
        Return how the observation of the given index moves what it sees
        into its frame (see moved): the shifts of the centres, x and y
        (arcsec), and of the systemic velocities (km/s), at the values of
        the offsets of its frame (see SkyFrame).
        """
        return self._members[index].shift.shifts(params)

    def orbit_components(self, index: int) -> tuple[tuple[int, int], ...]:
        """
        Return the components whose stars the observation of the given
        index sees on orbits whose weights are solved (see
        Observation.orbit_components), each as the index of its galaxy (in
        the group) and its own index.
        """
        member = self._members[index]
        components = member.observation.orbit_components(member.galaxies)
        return tuple((member.seen[g], i) for g, i in components)

    def light_amplitudes(self, index: int) -> dict[str, str | None]:
        """
        Return the amplitudes of the light that the model of the
        observation of the given index takes, each with None if its model
        is linear in it, or why not (see Observation.light_amplitudes).
        """
        member = self._members[index]
        return member.observation.light_amplitudes(member.galaxies)

    def model_params(self, index: int) -> tuple[str, ...]:
        """
        Return the names of the parameters that the model of the
        observation of the given index reads: those of the galaxies it
        sees and their separations (see DepthGroup), and its own.
        """
        member = self._members[index]
        return (
            tuple(name for g in member.seen
                  for name in self._mappings[g].values())
            + tuple(self._separations[g] for g in member.seen
                    if g in self._separations)
            + tuple(sorted(member.observation.pdescs())))


def _with_observation_pdescs(
        pdescs: Mapping[str, ParamDesc],
        observations: Sequence[Observation]
) -> dict[str, ParamDesc]:
    """
    Return the parameters of the galaxies of a group (and of their
    separations) with those of its observations (e.g. their flux scales;
    see Observation.pdescs). Raise ValueError if those of an observation
    have the names of others.
    """
    pdescs = dict(pdescs)
    for observation in observations:
        own = observation.pdescs()
        if repeated := sorted(own.keys() & pdescs.keys()):
            raise ValueError(
                f"the parameters of the observations have the names of "
                f"parameters of the galaxies: {repeated}; rename the "
                f"observations")
        pdescs |= own
    return pdescs


def _warn_unused_masses(
        galaxies: Sequence[Galaxy],
        observations: Sequence[Observation],
        seen: Sequence[Sequence[int]]
) -> None:
    """
    Log a warning for each galaxy with mass that neither its components
    nor the observations that see it take (e.g. of components without
    'mass' velocity traits): only derived values can then use it.
    """
    for g, galaxy in enumerate(galaxies):
        if galaxy.mass_model() is None or galaxy.components_use_mass():
            continue
        if any(g in indices and observation.takes_mass()
               for observation, indices in zip(observations, seen)):
            continue
        name = f"galaxy {g}" if galaxy.name() is None \
            else f"galaxy {galaxy.name()!r}"
        _log.warning(
            f"the mass of {name} is used by none of its components ('mass' "
            f"velocity traits or a vertical equilibrium) and none of its "
            f"observations (profiles of its mass): only derived values can "
            f"use it")
