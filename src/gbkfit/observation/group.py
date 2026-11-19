"""
Groups of observations: galaxies, the lenses in front of them and the
observations of them, evaluated together (see ObservationGroup), and the
galaxies that pass through each other along the line of sight (see
DepthGroup).
"""

import copy
import logging
from collections.abc import Collection, Mapping, Sequence
from typing import Any, NamedTuple, Self


from gbkfit.config import config_path, item_prefixes, warn
from gbkfit.cosmology import DEFAULT, Cosmology
from gbkfit.dataset import Dataset
from gbkfit.driver import Driver
from gbkfit.lensing import (
    PLANE_LINKING_VELOCITY, Lens, OfGalaxy, Plane, Tracer, plane_layout)
from gbkfit.model import DepthUnit, Galaxy
from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import iterutils
from ._depth_order import DepthGroup, DepthOrder
from ._frames import (
    FrameShift, alignments, check_offsets, first_on_sky, frame_shift,
    warn_frames)
from .group_plan import GroupPlan, place_drivers
from .observation import Observation, SeenGalaxies, SeenPlane


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
    galaxies it sees, and those galaxies as it sees them (see
    ObservationGroup.seen_by), with how it moves them into its frame.
    """
    observation: Observation
    label: str
    seen: tuple[int, ...]
    galaxies: SeenGalaxies


class ObservationGroup:
    """
    Galaxies, the lenses in front of them and the observations of them,
    evaluated together.

    Each observation names the galaxies it sees, which it may leave out
    when there is one galaxy; the light of several galaxies adds up, or
    through radiative transfer, is dimmed by the opacity of the galaxies
    in front (see depth_order). The parameters and constants of the
    galaxies are prefixed by their names when they have names (which they
    need when there are several; e.g. 'north_').

    The lenses (see Lens) bend the light of the galaxies behind them, of
    every observation of them: the galaxies and the lenses are laid out
    in planes by redshift (see plane_layout; every galaxy then needs a
    redshift), and the light of each plane is bent by the lenses of the
    planes in front of it (see Tracer), and dimmed by the opacity of
    their galaxies. The parameters of each lens are prefixed by its name
    (e.g. 'lens_main_theta_e'). A piece of a lens can be the mass of a
    galaxy of the group (see LensGalaxyMass and OfGalaxy): the galaxy must
    be at the plane of the lens and give what the piece needs (see
    Galaxy.check_needs), and no other piece may be its mass. Its mass then
    bends the light of every observation of the planes behind, whether
    the observation sees the galaxy or not.

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
    sky without them. The rays of its pixels meet the lenses moved as the
    galaxies (see Tracer). The observations that are not on the sky see
    the parameters as they are.

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
    lenses : Lens or Sequence of Lens, optional
        The lenses; none by default.
    depth_order : Sequence of str or DepthGroup, optional
        The galaxies along the line of sight, front to back, by name, or
        in groups of galaxies that can pass through each other (see
        DepthGroup): an observation of several galaxies of which one has
        opacity marches them in this order by radiative transfer (the
        light of each is dimmed by the opacity of those in front of it;
        see Scene.evaluation), and needs it then of all of them. In
        configurations, a group is {name: [galaxies]}. With lenses, it
        orders the galaxies of each plane, and the redshifts the planes
        (see plane_layout).
    cosmology : Cosmology, optional
        The cosmology of the galaxies, that of their setup (see
        Galaxy.plan_stage); Planck18 by default (see
        gbkfit.cosmology.DEFAULT).

    Raises
    ------
    ValueError
        If there are no galaxies or no observations, several galaxies
        without names, names repeat (of the galaxies, the lenses and the
        observations), an observation does not name the galaxies it sees
        (when there are several) or names unknown ones, depth_order names
        unknown galaxies or repeats them, has a group of fewer than two
        galaxies, or of the name of a galaxy or an observation, every
        observation of a galaxy has offsets of position or of velocity,
        or with lenses, a galaxy has no redshift, the planes cannot be
        laid out (see plane_layout), or a piece of a lens is the mass of a
        galaxy that it cannot be (see _check_masses).
    """

    def __init__(
            self,
            galaxies: Galaxy | Sequence[Galaxy],
            observations: Observation | Sequence[Observation],
            lenses: Lens | Sequence[Lens] = (),
            depth_order: Sequence[str | DepthGroup] | None = None,
            cosmology: Cosmology | None = None
    ):
        self._galaxies = iterutils.tuplify(galaxies)
        observations = iterutils.tuplify(observations)
        self._lenses = iterutils.tuplify(lenses)
        if not self._galaxies:
            raise ValueError("at least one galaxy is required")
        if not observations:
            raise ValueError("at least one observation is required")
        galaxy_names = [galaxy.name() for galaxy in self._galaxies]
        if len(galaxy_names) > 1 and None in galaxy_names:
            raise ValueError(
                "the galaxies need names when there are several")
        observation_names = [obs.name() for obs in observations]
        lens_names = [lens.name() for lens in self._lenses]
        self._prefixes = item_prefixes(
            galaxy_names, 'galaxies', 'galaxy', False)
        item_prefixes(
            observation_names, 'observations', 'observation', False)
        _check_names(galaxy_names, observation_names, lens_names)
        self._cosmology = DEFAULT if cosmology is None else cosmology
        self._depth = None if depth_order is None else DepthOrder(
            depth_order, galaxy_names, observation_names, self._prefixes)
        # The parameters of the galaxies, with those of where the masses
        # are on the sky of the galaxies that pieces of lenses are the
        # masses of (which lay them out there; see Galaxy.sky_pdescs and
        # _check_masses), and of their separations, to which those of the
        # lenses and of the observations are added
        lensing = {
            piece.galaxy() for lens in self._lenses for piece in lens.pieces()
            if isinstance(piece, OfGalaxy)}
        galaxy_pdescs, self._mappings = iterutils.merge_with_prefixes(
            [galaxy.pdescs() | (
                galaxy.sky_pdescs() if galaxy.name() in lensing else {})
             for galaxy in self._galaxies], self._prefixes)
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
        lens_pdescs, self._lens_mappings = iterutils.merge_with_prefixes(
            [lens.pdescs() for lens in self._lenses],
            [f'{name}_' for name in lens_names])
        if taken := sorted(lens_pdescs.keys() & galaxy_pdescs.keys()):
            raise ValueError(
                f"the parameters of the lenses have the names of parameters "
                f"of the galaxies: {taken}; rename the lenses")
        self._pdescs = _with_observation_pdescs(
            galaxy_pdescs | lens_pdescs, observations)
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
        # The paths of the light of the planes through the lenses, if any,
        # and the galaxies whose masses pieces of them are (see
        # _check_masses)
        self._tracer = None
        self._lensing: tuple[int, ...] = ()
        if self._lenses:
            self._tracer = Tracer(
                self._plane_layout(), self._lenses, self._cosmology)
            self._lensing = self._check_masses()
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
                observation, label, indices, self._galaxies_seen(
                    indices, frame is not None, frame_shift(
                        self._galaxies, self._mappings, indices, frame,
                        alignment)))
            for observation, label, indices, frame, alignment in zip(
                observations, labels, seen, frames, alignments(frames)))
        check_offsets(self._galaxies, frames, seen)
        warn_frames(self._galaxies, frames, seen)
        _warn_unused_masses(self._galaxies, observations, seen, self._lensing)
        self._warn_idle_lenses()
        self._check_observations()

    def _plane_layout(self) -> tuple[Plane, ...]:
        """
        Return the planes of the galaxies and the lenses, near to far (see
        plane_layout), which it logs. Raise ValueError if a galaxy has no
        redshift (or one of 0, which places it nowhere among them), or the
        planes cannot be laid out.
        """
        if missing := [
                self._galaxy_label(g)
                for g, galaxy in enumerate(self._galaxies)
                if not galaxy.redshift() > 0]:
            raise ValueError(
                f"with lenses, every galaxy needs a redshift above 0 (its "
                f"option redshift), which places it among them; these have "
                f"none: {missing}")
        depth_items = () if self._depth is None else [
            [self._galaxies[g].name() for g in members]
            for _, members in self._depth.items()]
        planes = plane_layout(
            [(galaxy.name(), galaxy.redshift()) for galaxy in self._galaxies],
            [(lens.name(), lens.redshift()) for lens in self._lenses],
            depth_items)
        for p, plane in enumerate(planes):
            _log.info(
                "plane %d at redshift %.6g: galaxies %s, lenses %s", p,
                plane.redshift,
                [self._galaxy_label(g) for g in plane.galaxies],
                [self._lenses[i].name() for i in plane.lenses])
        return planes

    def _check_masses(self) -> tuple[int, ...]:
        """
        Check the pieces of the lenses that are the masses of galaxies (see
        OfGalaxy): each is the mass of a galaxy of the group, at the plane
        of its lens (its mass bends light at its own distance), that gives
        what the piece needs (see Galaxy.check_needs); and no galaxy is the
        mass of two pieces (its mass would bend light twice). Return the
        galaxies whose masses they are, by index.

        Raises
        ------
        ValueError
            If one does not hold, of the path of the piece.
        """
        names = [galaxy.name() for galaxy in self._galaxies]
        planes = self._tracer.planes()
        plane_of = {
            g: p for p, plane in enumerate(planes) for g in plane.galaxies}
        lens_plane = {
            i: p for p, plane in enumerate(planes) for i in plane.lenses}
        pieces: dict[int, str] = {}
        for i, lens in enumerate(self._lenses):
            for k, piece in enumerate(lens.pieces()):
                if not isinstance(piece, OfGalaxy):
                    continue
                label = f"the piece {piece.name()!r} of lens {lens.name()!r}"
                name = piece.galaxy()
                with config_path('lenses', i, 'pieces', k):
                    if name not in names:
                        raise ValueError(
                            f"{label} is the mass of the galaxy {name!r}, "
                            f"which is not one of the galaxies: {names}")
                    g = names.index(name)
                    if g in pieces:
                        raise ValueError(
                            f"{label} and {pieces[g]} are both the mass of "
                            f"the galaxy {name!r}, which would bend light "
                            f"twice")
                    if plane_of[g] != lens_plane[i]:
                        raise ValueError(
                            f"{label} is the mass of the galaxy {name!r}, "
                            f"which bends light at its own redshift "
                            f"({self._galaxies[g].redshift()}): give the "
                            f"lens a redshift within "
                            f"{PLANE_LINKING_VELOCITY:g} km/s of it; it is "
                            f"at {lens.redshift()}")
                    self._galaxies[g].check_needs(piece.needs(label))
                pieces[g] = label
        return tuple(sorted(pieces))

    def _galaxy_label(self, index: int) -> str:
        """Return the name of a galaxy in messages, or its index."""
        name = self._galaxies[index].name()
        return str(index) if name is None else repr(name)

    def _warn_idle_lenses(self) -> None:
        """
        Warn (see warn) of each lens that bends the light of no galaxy
        that an observation sees: of none behind it.
        """
        if self._tracer is None:
            return
        planes = self._tracer.planes()
        plane_of = {
            g: p for p, plane in enumerate(planes) for g in plane.galaxies}
        farthest = max(
            plane_of[g] for member in self._members for g in member.seen)
        for p, plane in enumerate(planes):
            for i in plane.lenses:
                if p >= farthest:
                    warn(
                        f"lens {self._lenses[i].name()!r} bends the light of "
                        f"no galaxy that an observation sees: none is behind "
                        f"it (at redshift {self._lenses[i].redshift()})")

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

    def _galaxies_seen(
            self, seen: tuple[int, ...], on_sky: bool, shift: FrameShift
    ) -> SeenGalaxies:
        """
        Return the galaxies of the given indices as an observation that
        sees them sees them (see seen_by), on the sky or not, which it
        moves into its frame by the given shift: in their planes (see
        SeenPlane), each in the depth order of those of them in it (a
        group of which it sees one galaxy is that galaxy).
        """
        return SeenGalaxies(
            tuple(self._galaxies[j] for j in seen),
            tuple(self._mappings[j] for j in seen),
            tuple(self._galaxy_labels[j] for j in seen),
            self._seen_planes(seen, on_sky), self._sky_origin,
            self._cosmology, self._tracer, self._lens_mappings, shift)

    def _seen_planes(
            self, seen: tuple[int, ...], on_sky: bool
    ) -> tuple[SeenPlane, ...]:
        """
        Return the planes of the galaxies of the given indices, near to far
        (see SeenPlane), of an observation on the sky or not: without
        lenses, or not on the sky (e.g. a profile of a galaxy, intrinsic
        to it), one of all of them; with lenses, the image plane, of those
        of the planes of the tracer up to its first with lenses, and each
        plane behind it, those without the galaxies left out.
        """
        if self._tracer is None or not on_sky:
            order = None if self._depth is None else self._depth.seen(seen)
            return (SeenPlane(tuple(range(len(seen))), order, None),)
        planes = self._tracer.planes()
        first = next(p for p, plane in enumerate(planes) if plane.lenses)
        # The galaxies seen of each plane of the tracer, by index, and of
        # the image plane, by the planes of the tracer that it merges
        members = [
            [g for g in plane.galaxies if g in seen] for plane in planes]
        drawn = [(members[:first + 1], None)] + [
            ([galaxies], p) for p, galaxies in enumerate(members)
            if p > first]
        seen_planes = []
        for merged, traced in drawn:
            galaxies = sorted(g for plane in merged for g in plane)
            if not galaxies:
                continue
            # (the planes by redshift, near to far, and the galaxies of
            # each by the depth order, which leaves out those it does not
            # place)
            units = []
            for plane in merged:
                if len(plane) == 1:
                    units.append(DepthUnit((galaxies.index(plane[0]),)))
                elif plane and self._depth is not None:
                    units += [
                        unit._replace(galaxies=tuple(
                            galaxies.index(plane[k]) for k in unit.galaxies))
                        for unit in self._depth.seen(plane)]
            seen_planes.append(SeenPlane(
                tuple(seen.index(g) for g in galaxies), tuple(units) or None,
                traced))
        return tuple(seen_planes)

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
            placement: Mapping[str, Driver] | None = None,
            reference: ParamValues | None = None
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
        reference : ParamValues, optional
            The values of the parameters at which what is laid out once
            for all evaluations is laid out (see ParamSpace.reference):
            the grids of the planes behind lenses that observations see,
            which need them (with the masses on the sky there of the
            galaxies whose masses pieces of lenses are, see GroupPlan).

        Returns
        -------
        GroupPlan
            The plan.

        Raises
        ------
        ValueError
            If the placement names unknown galaxies or observations,
            what an observation sees does not go with it (see
            Observation.plan), a galaxy cannot plan its stage (see
            Galaxy.plan_stage), or the mass of a galaxy that a lens bends
            light with cannot be laid out on the sky at the reference
            values.
        """
        return GroupPlan(
            self, *place_drivers(self, driver, placement), reference)

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

    def lenses(self) -> tuple[Lens, ...]:
        """Return the lenses."""
        return self._lenses

    def lens_mappings(self) -> tuple[dict[str, str], ...]:
        """
        Return the names of the parameters of each lens among those of
        the group, by its own names (see Lens.pdescs).
        """
        return tuple(self._lens_mappings)

    def lensing_galaxies(self) -> tuple[int, ...]:
        """
        Return the galaxies whose masses pieces of the lenses are (see
        OfGalaxy), by index: their stages lay out their masses on the sky
        (see GroupPlan).
        """
        return self._lensing

    def bending_masses(self, index: int) -> tuple[int, ...]:
        """
        Return the galaxies whose masses bend the light of what the
        observation of the given index sees, by index: those whose masses
        pieces of the lenses in front of its planes are (see model_params),
        which it need not see.
        """
        names = [galaxy.name() for galaxy in self._galaxies]
        return tuple(sorted(
            names.index(name)
            for i in self._bending(self._members[index].galaxies)
            for name in self._lenses[i].galaxies()))

    def tracer(self) -> Tracer | None:
        """
        Return the paths of the light of the planes of the galaxies and
        the lenses through the lenses (see Tracer), if it has lenses.
        """
        return self._tracer

    def pdescs(self) -> dict[str, ParamDesc]:
        """
        Return the parameters of the galaxies (with those of where the
        masses are on the sky of the galaxies that lenses bend light by,
        see Galaxy.sky_pdescs) and of the lenses, prefixed, and those of
        the observations (see Observation.pdescs): those of their models
        (the likelihoods, which compare them with the data, add their own;
        see Likelihoods.pdescs).
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
        return self._members[index].galaxies.shift.moved(params)

    def frame_shifts(
            self, index: int, params: ParamValues
    ) -> tuple[float, float, float]:
        """
        Return how the observation of the given index moves what it sees
        into its frame (see moved): the shifts of the centres, x and y
        (arcsec), and of the systemic velocities (km/s), at the values of
        the offsets of its frame (see SkyFrame).
        """
        return self._members[index].galaxies.shift.shifts(params)

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
        sees and their separations (see DepthGroup), those of the lenses
        that bend their light (in front of a plane of them), and its own.
        """
        member = self._members[index]
        return (
            tuple(name for g in member.seen
                  for name in self._mappings[g].values())
            + tuple(self._separations[g] for g in member.seen
                    if g in self._separations)
            + tuple(name for i in self._bending(member.galaxies)
                    for name in self._lens_mappings[i].values())
            + tuple(sorted(member.observation.pdescs())))

    def _bending(self, seen: SeenGalaxies) -> tuple[int, ...]:
        """
        Return the lenses (by index) that bend the light of galaxies
        an observation sees: those of the planes of the tracer in front of
        the farthest plane it sees whose light they bend.
        """
        traced = [
            plane.traced for plane in seen.planes if plane.traced is not None]
        if not traced:
            return ()
        planes = self._tracer.planes()
        return tuple(sorted(
            i for p in range(max(traced)) for i in planes[p].lenses))


def _check_names(
        galaxies: Sequence[str | None],
        observations: Sequence[str | None],
        lenses: Sequence[str]
) -> None:
    """
    Raise ValueError if names repeat among the lenses, or among the
    galaxies, the observations and the lenses of a group (those of each of
    the first two are checked apart; see item_prefixes): they name their
    parameters, and the planes and outputs of the galaxies.
    """
    if len(set(lenses)) != len(lenses):
        raise ValueError(f"the names of the lenses repeat: {list(lenses)}")
    kinds = (
        ('galaxies', set(galaxies) - {None}),
        ('observations', set(observations) - {None}),
        ('lenses', set(lenses)))
    for k, (kind, names) in enumerate(kinds):
        for other, other_names in kinds[k + 1:]:
            if repeated := sorted(names & other_names):
                raise ValueError(
                    f"the {kind} and the {other} must have different names; "
                    f"repeated: {repeated}")


def _with_observation_pdescs(
        pdescs: Mapping[str, ParamDesc],
        observations: Sequence[Observation]
) -> dict[str, ParamDesc]:
    """
    Return the parameters of the galaxies of a group (and of their
    separations) and of its lenses with those of its observations (e.g.
    their flux scales; see Observation.pdescs). Raise ValueError if those
    of an observation have the names of others.
    """
    pdescs = dict(pdescs)
    for observation in observations:
        own = observation.pdescs()
        if repeated := sorted(own.keys() & pdescs.keys()):
            raise ValueError(
                f"the parameters of the observations have the names of "
                f"parameters of the galaxies, the lenses or other "
                f"observations: {repeated}; rename the observations")
        pdescs |= own
    return pdescs


def _warn_unused_masses(
        galaxies: Sequence[Galaxy],
        observations: Sequence[Observation],
        seen: Sequence[Sequence[int]],
        lensing: Collection[int]
) -> None:
    """
    Log a warning for each galaxy with mass that neither its components,
    the observations that see it nor a lens (those of the given indices
    are the masses of pieces of lenses) take (e.g. of components without
    'mass' velocity traits): only derived values can then use it.
    """
    for g, galaxy in enumerate(galaxies):
        if galaxy.mass_model() is None or galaxy.components_use_mass() \
                or g in lensing:
            continue
        if any(g in indices and observation.takes_mass()
               for observation, indices in zip(observations, seen)):
            continue
        name = f"galaxy {g}" if galaxy.name() is None \
            else f"galaxy {galaxy.name()!r}"
        _log.warning(
            f"the mass of {name} is used by none of its components ('mass' "
            f"velocity traits or a vertical equilibrium), none of its "
            f"observations (profiles of its mass) and no lens: only derived "
            f"values can use it")
