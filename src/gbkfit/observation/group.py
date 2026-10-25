import copy
import logging
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Self

from astropy.coordinates import SkyCoord

from gbkfit.dataset import Dataset
from gbkfit.driver import Driver
from gbkfit.model import Galaxy
from gbkfit.params import ParamDesc, ParamScalarDesc, ParamValues
from gbkfit.utils import iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from ._group_plan import GroupPlan
from ._scene import DepthUnit, SeenGalaxies
from .observation import Observation


__all__ = [
    'DepthGroup',
    'ObservationGroup'
]


_log = logging.getLogger(__name__)


class DepthGroup(NamedTuple):
    """
    Galaxies that can pass through each other along the line of sight
    (e.g. a merger), by name, which radiative transfer evaluates together
    (see TransferPlan), and the name of the group (the id 'group' of its
    extra outputs). Each galaxy after the first has a parameter of its
    position along the line of sight from the first, '{galaxy}_separation'
    (arcsec; positive behind it, away from the viewer).
    """
    name: str
    galaxies: tuple[str, ...]


class _Frame(NamedTuple):
    """
    How an observation moves what it sees into its frame (see
    ObservationGroup): the parameters of the centres (x and y) and of the
    systemic velocities (v) of the components of the galaxies it sees
    (and the centres of what it adds, e.g. the masses of its lens),
    where the origin of the frame of the first observation is in its
    frame (arcsec, x and y), and the names of the parameters of the
    offsets of its frame, of x, y and the velocity (None for those it does
    not fit).
    """
    x: tuple[str, ...]
    y: tuple[str, ...]
    v: tuple[str, ...]
    alignment: tuple[float, float]
    offsets: tuple[str | None, str | None, str | None]

    def moved(self, params: ParamValues) -> ParamValues:
        """
        Return the parameters with the centres and the systemic velocities
        moved by the alignment and the offsets (the parameters themselves
        without either).
        """
        if self.alignment == (0, 0) and self.offsets == (None, None, None):
            return params
        offsets = [
            0.0 if name is None else params[name] for name in self.offsets]
        shifts = [
            self.alignment[0] + offsets[0], self.alignment[1] + offsets[1],
            offsets[2]]
        moved = dict(params)
        for names, shift in zip((self.x, self.y, self.v), shifts):
            for name in names:
                moved[name] = params[name] + shift
        return moved


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
    Observation.sky_origin): its reference pixel, at the RA and Dec of its
    grid (see Observable.sky_rval). The other observations on the sky
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
    Observation.centre_params) are moved with those of the components.
    The observations that are not on the sky see the parameters as they
    are.

    The group describes the galaxies and the observations; its
    evaluation is planned on devices by plan (see GroupPlan), where the
    stage of each galaxy is evaluated once for all the observations of
    it.

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
        observation, every observation of a galaxy has offsets of position
        or of velocity.
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
        # (as given, for copies; see with_data)
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
        # The parameters of the observations (e.g. their flux scales)
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
        # The ids of the galaxies and of the observations in the extra
        # outputs: their names, or their positions if they have none (e.g.
        # 'galaxy0', 'observation0')
        self._galaxy_labels = [
            f'galaxy{i}' if name is None else name
            for i, name in enumerate(galaxy_names)]
        self._observation_labels = [
            f'observation{i}' if name is None else name
            for i, name in enumerate(observation_names)]
        # (the origin of the frame of the parameters of the galaxies)
        first = _first_on_sky(self._observations)
        self._sky_origin = None if first is None \
            else self._observations[first].sky_origin()
        seen = []
        for i, observation in enumerate(self._observations):
            with parseutils.config_path('observations', i):
                seen.append(self._seen(i, observation))
        # The indices of the galaxies that each observation sees
        self._seen_galaxies = tuple(tuple(indices) for indices in seen)
        self._frames = tuple(
            self._frame(seen, observation, alignment)
            for seen, observation, alignment in zip(
                seen, self._observations, _alignments(self._observations)))
        _check_offsets(self._galaxies, self._observations, seen)
        _warn_unused_masses(self._galaxies, self._observations, seen)

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

    def _frame(
            self,
            seen: list[int],
            observation: Observation,
            alignment: tuple[float, float]
    ) -> _Frame:
        """
        Return the frame of an observation that sees the galaxies of the
        given indices, with the given alignment (see _Frame).
        """
        names = dict(x=[], y=[], v=[])
        for g in seen:
            galaxy, mapping = self._galaxies[g], self._mappings[g]
            for i, component in enumerate(galaxy.components()):
                params = galaxy.component_params(i)
                frame = component.frame_params()
                for key, own in zip('xyv', (frame.x, frame.y, frame.vsys)):
                    name = None if own is None else mapping[params[own]]
                    if name is not None and name not in names[key]:
                        names[key].append(name)
        # (and the centres of what the observation adds, e.g. its lens)
        for x, y in observation.centre_params():
            names['x'].append(x)
            names['y'].append(y)
        return _Frame(
            tuple(names['x']), tuple(names['y']), tuple(names['v']),
            alignment, observation.offset_params())

    def seen_by(self, index: int) -> SeenGalaxies:
        """
        Return the galaxies that the observation of the given index sees,
        as it sees them: in the depth order of those of them in it (a
        group of which it sees one galaxy is that galaxy).
        """
        seen = self._seen_galaxies[index]
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
        return SeenGalaxies(
            tuple(self._galaxies[j] for j in seen),
            tuple(self._mappings[j] for j in seen),
            tuple(self._galaxy_labels[j] for j in seen),
            None if order is None else tuple(order), self._sky_origin)

    def plan(
            self,
            driver: Driver | None = None,
            drivers: Mapping[str, Driver] | None = None
    ) -> 'GroupPlan':
        """
        Plan the evaluation of the group on devices (see GroupPlan): the
        stage of each galaxy and the model of each observation, each on
        its driver (see placement).

        Parameters
        ----------
        driver : Driver, optional
            The driver of the galaxies and the observations; the host by
            default.
        drivers : Mapping of str to Driver, optional
            Other drivers of some of them, by their labels (see
            galaxy_label and observation_label): e.g. {'halpha': gpu0,
            'hi': gpu1}.

        Returns
        -------
        GroupPlan
            The plan.

        Raises
        ------
        ConfigError
            If drivers names unknown galaxies or observations, or what an
            observation sees does not go with it (see Observation.plan), or
            a galaxy cannot plan its stage (see Galaxy.plan_stage).
        """
        return GroupPlan(self, *self.placement(driver, drivers))

    def placement(
            self,
            driver: Driver | None = None,
            drivers: Mapping[str, Driver] | None = None
    ) -> tuple[tuple[Driver, ...], tuple[Driver, ...]]:
        """
        Return the driver of each galaxy and of each observation: the
        given one (the host by default), or another one of their label
        (see plan).

        Raises
        ------
        ConfigError
            If drivers names unknown galaxies or observations, or a label
            that is of a galaxy and of an observation.
        """
        if driver is None:
            # (imported here: the host driver needs its native module)
            from gbkfit.driver.drivers.host import DriverHost
            driver = DriverHost()
        drivers = dict(drivers or {})
        galaxies, observations = self._galaxy_labels, self._observation_labels
        if unknown := sorted(drivers.keys() - {*galaxies, *observations}):
            raise ConfigError(
                f"there are drivers of unknown galaxies or observations "
                f"{unknown}; the galaxies are {galaxies} and the "
                f"observations {observations}")
        if both := sorted(drivers.keys() & set(galaxies) & set(observations)):
            raise ConfigError(
                f"there are drivers of {both}, which are labels of a galaxy "
                f"and of an observation; name them apart")
        return (
            tuple(drivers.get(label, driver) for label in galaxies),
            tuple(drivers.get(label, driver) for label in observations))

    def with_data(self, datasets: Sequence[Dataset]) -> Self:
        """
        Return a group of the same galaxies and observations, with other
        data: one dataset for each observation (e.g. those of a bootstrap;
        see Observation.with_data), under its likelihood.

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

    def observation_label(self, index: int) -> str:
        """
        Return the id of the observation of the given index in the extra
        outputs (and in the outputs of the tasks): its name, or
        'observation{index}'.
        """
        return self._observation_labels[index]

    def pdescs(self) -> dict[str, ParamDesc]:
        """
        Return the parameters of the galaxies, prefixed, and those of the
        observations (see Observation.pdescs).
        """
        return self._pdescs

    def constants(self) -> dict[str, Any]:
        """Return the constants of the galaxies, prefixed."""
        return self._constants

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
        return self._seen_galaxies[index]

    def moved(self, index: int, params: ParamValues) -> ParamValues:
        """
        Return the parameters as the observation of the given index sees
        them: the centres and systemic velocities of what it sees moved
        into its frame (see the class).
        """
        return self._frames[index].moved(params)

    def model_params(self, index: int) -> tuple[str, ...]:
        """
        Return the names of the parameters that the model of the
        observation of the given index reads: those of the galaxies it
        sees and their separations (see DepthGroup), and its own but those
        of its likelihood, which only the comparison with its data reads
        (e.g. those of a fitted noise; see Likelihood.pdescs).
        """
        observation = self._observations[index]
        seen = self._seen_galaxies[index]
        likelihood = observation.likelihood()
        own = observation.pdescs().keys() - (
            set() if likelihood is None else likelihood.pdescs().keys())
        return (
            tuple(name for g in seen for name in self._mappings[g].values())
            + tuple(self._separations[g] for g in seen
                    if g in self._separations)
            + tuple(sorted(own)))


# Warn of observations whose origins are further apart on the sky (deg),
# or whose frames turn by more (deg; see _alignments)
_ALIGNMENT_SEPARATION_WARNING = 1.0
_ALIGNMENT_ROTATION_WARNING = 0.1


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


def _first_on_sky(observations: Sequence[Observation]) -> int | None:
    """
    Return the index of the first observation on the sky (see
    Observation.sky_origin), whose frame is that of the parameters of the
    galaxies; None if none is.
    """
    return next(
        (i for i, observation in enumerate(observations)
         if observation.sky_origin() is not None), None)


def _alignments(
        observations: Sequence[Observation]
) -> list[tuple[float, float]]:
    """
    Return where the origin of the frame of the first observation on the
    sky is in the frame of each (arcsec, x to the west and y to the
    north), of the RA and Dec of their origins (see
    Observation.sky_origin); (0, 0) for those that are not on the sky.
    Warn (see parseutils.warn) of origins more than
    _ALIGNMENT_SEPARATION_WARNING apart (e.g. one of data without
    coordinates, at RA and Dec 0), and of frames that turn by more than
    _ALIGNMENT_ROTATION_WARNING: the frames are aligned by a shift; that
    of an origin far from the first, near a pole, also turns by about the
    separation times tan(Dec).
    """
    f = _first_on_sky(observations)
    if f is None:
        return [(0.0, 0.0)] * len(observations)
    origins = [observation.sky_origin() for observation in observations]
    first = SkyCoord(*origins[f], unit='deg')
    alignments = []
    for i, origin in enumerate(origins):
        if origin is None:
            alignments.append((0.0, 0.0))
            continue
        origin = SkyCoord(*origin, unit='deg')
        east, north = origin.spherical_offsets_to(first)
        alignments.append(
            (-float(east.to_value('arcsec')),
             float(north.to_value('arcsec'))))
        separation = origin.separation(first).to_value('deg')
        if separation > _ALIGNMENT_SEPARATION_WARNING:
            parseutils.warn(
                f"the origins of the frames of observations {f} and {i} (the "
                f"RA and Dec of their reference pixels) are {separation:g} "
                f"deg apart: check their coordinates")
        elif separation > 0:
            # (the convergence of the meridians: the bearing from the origin
            # to the first, less the bearing back, less 180 deg)
            bearings = origin.position_angle(first) \
                - first.position_angle(origin)
            rotation = bearings.to_value('deg') % 360 - 180
            if abs(rotation) > _ALIGNMENT_ROTATION_WARNING:
                parseutils.warn(
                    f"the frames of observations {f} and {i} turn by "
                    f"{rotation:.2g} deg against each other, which their "
                    f"alignment leaves out: their reference pixels are "
                    f"{separation * 60:.2g} arcmin apart")
    return alignments


def _check_offsets(
        galaxies: Sequence[Galaxy],
        observations: Sequence[Observation],
        seen: Sequence[Sequence[int]]
) -> None:
    """
    Raise ConfigError if every observation on the sky that sees a galaxy
    has offsets of position, or of velocity (see ObservationSky): they
    would move its centre, or its systemic velocity, with nothing to place
    it.
    """
    for g, galaxy in enumerate(galaxies):
        offsets = [
            obs.offset_params()
            for obs, indices in zip(observations, seen)
            if g in indices and obs.sky_origin() is not None]
        for kind, k in (('position', 0), ('velocity', 2)):
            if offsets and all(params[k] is not None for params in offsets):
                raise ConfigError(
                    f"every observation of galaxy {galaxy.name() or g} has "
                    f"offsets of {kind}; one needs none, whose frame places "
                    f"the galaxy")


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
