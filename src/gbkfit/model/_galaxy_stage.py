"""
The stage of a galaxy (GalaxyStage): what the galaxy computes once per
evaluation, for all its observations: the mass of the galaxy, with the
densities of the components it follows, its field in the gravity of the
galaxy, its mass laid out on the sky for the lenses of its mass, and
what each component takes from the galaxy (see ComponentStage); each of
its parts again only when the parameters it reads, or the parts it
takes, change.
"""

import dataclasses
import functools
import logging
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from gbkfit.cosmology import DEFAULT, Cosmology
from gbkfit.driver import Driver
from gbkfit.params import (
    ModelEvaluationError, ParamValues, ParamView, StageCache)
from ._capabilities import Capability
from ._fixed_point import fixed_point
from .components.base import Balanced, GalaxyMass, Placed
from .extras import Extras
from .mass import MassContext
from .mass.sky import (
    RingPlacement, SkyPlacement, SpheroidAxis, plane_tilt)

if TYPE_CHECKING:
    from .galaxy import Galaxy
    from .mass import MassModel
    from .components.base import ComponentStage, ComponentState
    from .mass.sky import SkyMass


__all__ = [
    'GalaxyEpoch',
    'GalaxyStage',
    'GalaxyState'
]


_log = logging.getLogger(__name__)

# The angle (degrees) between the plane of a part that the gravity takes
# as in the plane of the mass and the plane of the first, above which a
# warning says that the gravity takes the components too simply (see
# GalaxyStage.evaluate). Measured on an exponential disk whose outer rings
# tilt: that makes their circular velocity about 1% wrong (2026-10-14: 1%
# at a tilt of about 20 degrees).
_MAX_TILT = 20.0

# The largest difference (arcsec) between the centres of the components
# that the mass moves, holds up or follows: one centre, that of the mass;
# centres tied or fixed at the same values are equal to the last bits
_CENTRE_TOLERANCE = 1e-9

# The largest change of the pdf of a balanced layer in its last pass,
# relative to its peak, at which the layers have settled (see
# GalaxyStage._balance). A stop at a tolerance leaves steps of up to
# about 0.6 of it in the model, where the number of passes changes with
# the parameters: a forward difference of a fitter (steps of 1.5e-8)
# across one is off by 0.6 tolerance / 1.5e-8, 4e-5 at 1e-12, which
# costs 1-2 passes more than 1e-10 (measured 2026-11-18)
_BALANCE_TOLERANCE = 1e-12

# The passes after which the layers have not settled: 8-17 settle them
# in Newtonian gravity, about 60 where they barely do (measured
# 2026-11-18, the local term of Palatini f(R) gravity near collapse)
_BALANCE_PASSES = 100


@dataclasses.dataclass(frozen=True)
class GalaxyEpoch:
    """
    The universe at the epoch of a galaxy, its redshift, in a cosmology
    (see gbkfit.cosmology): its critical density (Msun/kpc^3), its matter
    density parameter and its Ricci scalar (kpc^-2) then, its Ricci
    scalar today, and the cosmic baryon fraction (None if the cosmology
    has no Omega_b).
    """
    critical_density: float
    omega_m: float
    ricci_scalar: float
    ricci_scalar_today: float
    baryon_fraction: float | None

    @classmethod
    def of(cls, cosmology: Cosmology, redshift: float) -> 'GalaxyEpoch':
        """Return the epoch of a redshift in a cosmology."""
        return cls(
            cosmology.critical_density(redshift),
            cosmology.omega_m(redshift), cosmology.ricci_scalar(redshift),
            cosmology.ricci_scalar(0.0), cosmology.baryon_fraction())


@dataclasses.dataclass(frozen=True)
class GalaxyState:
    """
    A galaxy in one evaluation, for all its observations (see
    GalaxyStage.evaluate): its mass (None without mass), what it gives
    each component that takes anything from it: its state, by the index
    of the component (see ComponentStage.state), and its mass laid out on
    the sky, for the lenses of its mass (see SkyMass; None if its stage
    does not lay it out).
    """
    mass: GalaxyMass | None
    states: dict[int, 'ComponentState']
    sky: 'SkyMass | None' = None


class _ComponentParts(NamedTuple):
    """
    The caches of the parts of a component in the stage of its galaxy
    (see GalaxyStage): its light, the groundwork of its state and its
    state.
    """
    light: StageCache[ParamValues]
    groundwork: StageCache[object]
    state: StageCache['ComponentState | None']


class GalaxyStage:
    """
    The stage of a galaxy: what it computes once per evaluation, for all
    its observations, on the given driver and in float64, in parts:

    1. the scale of the galaxy, from its distance (see GalaxyScale), if
       it has mass;
    2. the light of each component that takes part (see
       Component.plan_stage): the values of the parameters of its light,
       with what it makes of them (e.g. the light of image traits; see
       ComponentStage.values);
    3. the mass of the galaxy (see mass): its context, with the densities
       of the components it follows (of their light);
    4. its field, in the gravity of the galaxy (see Gravity.field);
    5. with the sky part (for the lenses of its mass; see the parameter
       sky_mass), its mass laid out on the sky (see SkyMass), placed by
       the components that the mass moves, holds up or follows (see
       Galaxy.placed and SkyPlacement);
    6. what each component takes from the galaxy: the groundwork of its
       state (see ComponentStage.groundwork; e.g. the forces in the Jeans
       equations of a spheroid), and its state (see ComponentStage.state):
       e.g. the rotation of a disk in the gravity of the mass, or the
       dispersion of its vertical equilibrium.

    Where the mass follows the layers of components whose thickness is
    their vertical equilibrium in its field (balanced, see Balanced: e.g.
    a gas disk of the equilibrium 'galaxy' that a traced disk follows),
    the field sets the layers that the mass takes: the mass and its field
    are one part (3 and 4), the fixed point of the layers, which those
    components then draw (see _balance).

    Each part reads the values of the parameters it declares alone: the
    scale those of the distance, the mass those of the mass, the field
    those of the gravity, the sky those that place the components where
    the mass is (see Placed.placement_params), and each part of a
    component those that its stage declares (see ComponentStage); and it
    takes the parts before it that it uses (the sky takes the mass, not
    its field). An evaluation computes a part again only when the values
    it reads, or the parts it takes, change (see StageCache and evaluate):
    e.g. a step in the position angle of a spheroid of Jeans kinematics
    computes none of them again, a step in the anisotropy of its stars not
    the groundwork of its state, and a step in a parameter of the gravity
    neither the mass nor the sky. A sky computed again of the same layers
    as the last (e.g. after a step in the rotation of a disk that the mass
    follows, which builds the mass again) is the last, of its version
    (see StageCache: same): what takes it is not evaluated again (e.g.
    the observations whose light its lenses bend).

    The components that the mass moves, holds up and follows are at one
    centre, that of the mass: an evaluation where their centres differ
    fails (see Placed.frames). It takes their parts as in one
    plane, whatever their orientations (their rotation is then projected
    with their own orientations); with extra outputs, a warning says
    where their planes tilt from that of the first enough to matter (see
    evaluate).

    Parameters
    ----------
    galaxy : Galaxy
        The galaxy: its components, its distance, its mass and the links
        of its components and its mass (see MassLinks), and its redshift
        (its epoch, and its distance if that is cosmological).
    driver : Driver
        The driver where the components evaluate their parts.
    sky_origin : tuple of float, optional
        The RA and Dec (degrees) of the origin of the frame of the
        parameters of the galaxy (that of the first observation on the
        sky; see ObservationGroup), which image traits need (see
        ImageMaps).
    cosmology : Cosmology, optional
        The cosmology of the galaxy, that of its setup: of its epoch, once
        (see GalaxyEpoch), and its distance; Planck18 by default (see
        gbkfit.cosmology.DEFAULT).
    sky_mass : bool, optional
        Whether it lays out the mass of the galaxy on the sky (the sky
        part; for the lenses of its mass); not by default.

    Raises
    ------
    ValueError
        If a component cannot plan its part (e.g. it has an image trait,
        but there is no sky origin), or its mass takes the cosmic baryon
        fraction, which the cosmology lacks; or with the sky part, if the
        galaxy cannot lay out its mass on the sky (see Galaxy.gives:
        Capability.LENSING).
    """

    def __init__(
            self,
            galaxy: 'Galaxy',
            driver: Driver,
            sky_origin: tuple[float, float] | None = None,
            cosmology: Cosmology | None = None,
            sky_mass: bool = False
    ):
        components = galaxy.components()
        mappings = [
            galaxy.component_param_names(i) for i in range(len(components))]
        mass_model = galaxy.mass_model()
        links = galaxy.links()
        followed = links.followed
        redshift = galaxy.redshift()
        self._galaxy = galaxy
        self._redshift = redshift
        self._cosmology = DEFAULT if cosmology is None else cosmology
        self._epoch: GalaxyEpoch | None = None
        if mass_model is not None:
            mass_model.check_cosmology(self._cosmology)
            self._epoch = GalaxyEpoch.of(self._cosmology, redshift)
        # The part of each component that takes part, by its index, and
        # its label
        self._stages: dict[int, 'ComponentStage'] = {}
        self._labels: dict[int, str] = {}
        for i, component in enumerate(components):
            label = component.name() or f'component{i}'
            stage = component.plan_stage(
                driver, sky_origin, links.of(component), label)
            if stage is not None:
                self._stages[i] = stage
                self._labels[i] = label
        # The components whose layers the mass follows and its field sets
        # (see _balance), and the mass components that take those layers
        self._balanced = tuple(
            i for i, stage in self._stages.items()
            if isinstance(stage, Balanced) and stage.balanced())
        self._coupled = () if mass_model is None else _coupled(
            mass_model, {components[i].name() for i in self._balanced})
        # The components that the mass follows, and the parts of those
        # where the mass is that say where they are, by index (see
        # Galaxy.placed)
        self._followed = tuple(
            i for i in self._stages if components[i].name() in followed)
        self._placed: dict[int, Placed] = {}
        for i in galaxy.placed():
            stage = self._stages.get(i)
            if not isinstance(stage, Placed):
                raise TypeError(
                    f"the component {self._labels.get(i, i)!r} is where the "
                    f"mass of its galaxy is, but its part in the stage does "
                    f"not say where its parts are (see Placed)")
            self._placed[i] = stage
        # The caches of its parts (see evaluate): of its scale and its
        # mass, if it has mass, and of each component that takes part, by
        # its index; they read the parameters of the galaxy
        lights = {
            i: StageCache(mappings[i][p] for p in stage.light_params())
            for i, stage in self._stages.items()}
        self._scale_cache: StageCache[float] | None = None
        self._mass_cache: StageCache[GalaxyMass] | None = None
        self._field_cache: StageCache[GalaxyMass] | None = None
        if galaxy.distance() is not None:
            self._scale_cache = StageCache(galaxy.distance().pdescs())
        if mass_model is not None:
            scale = () if self._scale_cache is None else (self._scale_cache,)
            followed_lights = tuple(lights[i] for i in self._followed)
            if self._balanced:
                # (the mass with its field, which sets the layers it takes)
                self._mass_cache = StageCache(
                    (*mass_model.pdescs(), *galaxy.gravity().pdescs()),
                    (*scale, *followed_lights))
            else:
                self._mass_cache = StageCache(
                    mass_model.pdescs(), (*scale, *followed_lights))
                self._field_cache = StageCache(
                    galaxy.gravity().pdescs(), (self._mass_cache,))
        # (a sky of the same layers is not new, see the class)
        self._sky_cache: StageCache[SkyMass] | None = None
        if sky_mass:
            if (gap := galaxy.gives()[Capability.LENSING]) is not None:
                raise ValueError(
                    f"the galaxy cannot lay out its mass on the sky: {gap}")
            self._sky_cache = StageCache(
                dict.fromkeys(
                    mappings[i][p] for i, stage in self._placed.items()
                    for p in stage.placement_params()),
                (self._mass_cache,), _same_sky)
        # (what the components take of the mass: its field)
        field = () if mass_model is None else (
            self._mass_cache if self._field_cache is None
            else self._field_cache,)
        self._parts: dict[int, _ComponentParts] = {}
        for i, stage in self._stages.items():
            groundwork = StageCache((), (lights[i], *field))
            state = StageCache(
                (mappings[i][p] for p in stage.state_params()),
                (lights[i], groundwork, *field))
            self._parts[i] = _ComponentParts(lights[i], groundwork, state)

    def caches(self) -> dict[str, StageCache[Any]]:
        """
        Return the caches of its parts (see evaluate), by name: 'scale', if
        it has a distance, 'mass' and 'field', if it has mass ('mass'
        alone, the mass with its field, with balanced layers), 'sky', with
        the sky part, and '{component}.light', '{component}.groundwork' and
        '{component}.state' of each component that takes part, by its name,
        or its position (e.g. 'component0').
        """
        caches: dict[str, StageCache[Any]] = {}
        if self._scale_cache is not None:
            caches |= dict(scale=self._scale_cache)
        if self._mass_cache is not None:
            caches |= dict(mass=self._mass_cache)
        if self._field_cache is not None:
            caches |= dict(field=self._field_cache)
        if self._sky_cache is not None:
            caches |= dict(sky=self._sky_cache)
        for i, parts in self._parts.items():
            caches |= {
                f'{self._labels[i]}.{name}': cache
                for name, cache in parts._asdict().items()}
        return caches

    def state_cache(self, index: int) -> StageCache['ComponentState | None']:
        """
        Return the cache of the state of the component of the given index
        (see evaluate), which takes part.
        """
        return self._parts[index].state

    def sky_cache(self) -> StageCache['SkyMass']:
        """
        Return the cache of its sky part (see evaluate), which it has.

        Raises
        ------
        ValueError
            If it has no sky part (see the class: sky_mass).
        """
        if self._sky_cache is None:
            raise ValueError(
                "the stage of the galaxy does not lay out its mass on the "
                "sky (see sky_mass)")
        return self._sky_cache

    def evaluate(
            self,
            params: ParamValues,
            out_extra: Extras | None = None
    ) -> GalaxyState:
        """
        Evaluate the galaxy: its mass, and what it gives its components.
        Each of its parts is that of its last evaluation if the values of
        the parameters it reads, and the parts it takes, are the same (see
        the class); with extra outputs, the states of the components are
        evaluated again, for theirs.

        Parameters
        ----------
        params : ParamValues
            The values of the parameters of the galaxy (see Galaxy.pdescs).
        out_extra : Extras, optional
            Where the extra outputs go, if wanted (those of the components,
            with their name, or their position, as the id 'component', see
            ComponentStage.state). With them and mass, a warning says where
            the parts of the components it moves, holds up and follows
            tilt from the plane of the first part of the first by more
            than 20 degrees (which makes their circular velocities about
            1% wrong, or more).

        Returns
        -------
        GalaxyState
            The galaxy.

        Raises
        ------
        ModelEvaluationError
            If the components that the mass moves, holds up or follows have
            different centres (see the class), or with the sky part, the
            mass cannot be laid out on the sky (see SkyProjected).
        """
        lights = {
            i: parts.light.evaluate(
                params, functools.partial(self._light, index=i))
            for i, parts in self._parts.items()}
        mass = None
        sky = None
        if self._mass_cache is not None:
            scale = None if self._scale_cache is None \
                else self._scale_cache.evaluate(params, self._scale)
            built = self._mass_cache.evaluate(params, functools.partial(
                self._mass, scale=scale, lights=lights))
            mass = built if self._field_cache is None \
                else self._field_cache.evaluate(
                    params, functools.partial(self._field, mass=built))
            self._check_frames(params, out_extra is not None)
            if self._sky_cache is not None:
                sky = self._sky_cache.evaluate(params, functools.partial(
                    self._sky, mass=built, lights=lights))
        states = {}
        for i, parts in self._parts.items():
            groundwork = parts.groundwork.evaluate(params, functools.partial(
                self._groundwork, index=i, light=lights[i], mass=mass))
            extra = None if out_extra is None \
                else out_extra.scoped(component=self._labels[i])
            evaluate = parts.state.evaluate if extra is None \
                else parts.state.refresh
            state = evaluate(params, functools.partial(
                self._state, index=i, light=lights[i],
                groundwork=groundwork, mass=mass, out_extra=extra))
            if state is not None:
                states[i] = state
        return GalaxyState(mass, states, sky)

    def mass(self, params: ParamValues) -> GalaxyMass | None:
        """
        Return the mass of the galaxy at the values of its parameters (see
        Galaxy.pdescs), computed anew (apart from its evaluations, see
        evaluate): its mass model, the values of the parameters of the
        mass and the gravity, its context (with the densities of the
        components it follows) and the length of an arcsec in it; None
        without mass.
        """
        if self._mass_cache is None:
            return None
        lights = {
            i: self._light(self._parts[i].light.view(params), i)
            for i in self._followed}
        scale = None if self._scale_cache is None \
            else self._scale(self._scale_cache.view(params))
        built = self._mass(self._mass_cache.view(params), scale, lights)
        if self._field_cache is None:
            return built
        return self._field(self._field_cache.view(params), built)

    def sky_mass(self, params: ParamValues) -> 'SkyMass':
        """
        Return the mass of the galaxy laid out on the sky at the values of
        its parameters (see Galaxy.pdescs), computed anew (apart from its
        evaluations, see evaluate; e.g. at the reference values of the
        parameters, see Reference).

        Parameters
        ----------
        params : ParamValues
            The values of the parameters of the galaxy.

        Returns
        -------
        SkyMass
            Its mass on the sky.

        Raises
        ------
        ValueError
            If it has no sky part (see the class: sky_mass).
        ModelEvaluationError
            If the components that place the mass have different centres,
            or the mass cannot be laid out on the sky.
        """
        cache = self.sky_cache()
        lights = {
            i: self._light(self._parts[i].light.view(params), i)
            for i in {*self._followed, *self._placed}}
        scale = None if self._scale_cache is None \
            else self._scale(self._scale_cache.view(params))
        built = self._mass(self._mass_cache.view(params), scale, lights)
        self._check_frames(params, False)
        return self._sky(cache.view(params), built, lights)

    def kpc_per_arcsec(self, params: ParamValues) -> float | None:
        """
        Return the length of an arcsec in the galaxy (kpc) at the values of
        its parameters, computed anew; None without a distance.
        """
        if self._scale_cache is None:
            return None
        return self._scale(self._scale_cache.view(params))

    def state(
            self,
            index: int,
            params: ParamValues,
            mass: GalaxyMass | None
    ) -> 'ComponentState | None':
        """
        Return what the galaxy gives the component of the given index, its
        state (see ComponentStage.state; none if it takes nothing), at the
        values of the parameters of the galaxy, in its mass (see mass;
        None without mass), computed anew and without extra outputs.
        """
        parts = self._parts.get(index)
        if parts is None:
            return None
        light = self._light(parts.light.view(params), index)
        groundwork = self._groundwork(
            parts.groundwork.view(params), index, light, mass)
        return self._state(
            parts.state.view(params), index, light, groundwork, mass, None)

    def _scale(self, view: ParamView) -> float:
        """
        Return the length of an arcsec in the galaxy (kpc), of the values
        of the parameters of its distance.
        """
        return self._galaxy.distance().kpc_per_arcsec(
            view, self._redshift, self._cosmology)

    def _light(self, view: ParamView, index: int) -> ParamValues:
        """
        Return the values of the parameters of the light of the component
        of the given index (see ComponentStage.values), of the values of
        those of the galaxy that it reads.
        """
        stage = self._stages[index]
        return stage.values(self._own(view, index, stage.light_params()))

    def _mass(
            self,
            view: ParamView,
            scale: float | None,
            lights: dict[int, ParamValues]
    ) -> GalaxyMass:
        """
        Return the mass of the galaxy as it is built, without its field (see
        _field), of the values of the parameters of its mass, the length of
        an arcsec in it (kpc, see _scale; None without a distance), and the
        light of each component that it follows (see _light), by its index;
        with balanced layers, of those of its gravity too, with its field
        and the layers (see _balance).
        """
        galaxy = self._galaxy
        components = galaxy.components()
        mass_model = galaxy.mass_model()
        epoch = self._epoch
        context = MassContext(
            epoch.critical_density, epoch.omega_m, epoch.ricci_scalar,
            epoch.ricci_scalar_today, {
                components[i].name(): self._stages[i].density(
                    lights[i], scale)
                for i in self._followed},
            baryon_fraction=epoch.baryon_fraction, kpc_per_arcsec=scale)
        # (the thicknesses that the mass sets of the components that it
        # follows, of the surface densities of their fluids, which do not
        # take them: then the densities of those thicknesses)
        built = GalaxyMass(mass_model, view, context, None)
        thicknesses = {
            i: self._stages[i].thickness(lights[i], built)
            for i in self._followed}
        if any(thicknesses.values()):
            context = dataclasses.replace(context, densities={
                components[i].name(): self._stages[i].density(
                    lights[i] | thicknesses[i], scale)
                for i in self._followed})
        built = GalaxyMass(mass_model, view, context, None)
        return self._balance(built, scale, lights) if self._balanced \
            else built

    def _balance(
            self,
            built: GalaxyMass,
            scale: float,
            lights: dict[int, ParamValues]
    ) -> GalaxyMass:
        """
        Return the mass of the galaxy with its field, where the layers that
        it follows and its field sets (of the balanced components, see
        Balanced) are their vertical equilibrium in the field: the fixed
        point of their unknowns (see BalancedLayer and fixed_point), from
        uniform layers every time (so that the mass is a function of the
        parameters alone), to _BALANCE_TOLERANCE of the peaks of their
        pdfs. The mass is that as it is built (see _mass), with the
        densities of the layers of the light of their components (by
        index), in a galaxy of the given length of an arcsec (kpc). Where
        the field adds up (Newtonian gravity, see Gravity.additive), that
        of the mass that does not take the layers is computed once, and
        each pass computes that of the rest (e.g. a table, see
        TabulatedPotential); otherwise, each pass computes the field.

        Raises
        ------
        ModelEvaluationError
            If the layers do not settle in _BALANCE_PASSES passes, or a
            layer has no solution (see BalancedLayer.unknown).
        """
        model = built.model
        gravity = self._galaxy.gravity()
        names = {i: self._labels[i] for i in self._balanced}
        layers = {i: self._stages[i].layer(lights[i], scale) for i in names}
        grids = {i: layer.grid() for i, layer in layers.items()}
        ends = np.cumsum([0] + [
            len(R) * len(z) for R, z in grids.values()])

        def split(x: np.ndarray) -> dict[int, np.ndarray]:
            return {
                i: x[start:end].reshape(len(grids[i][0]), -1)
                for i, start, end in zip(layers, ends[:-1], ends[1:])}

        def mass_of(x: np.ndarray) -> GalaxyMass:
            tables = {
                i: layers[i].values(unknown)
                for i, unknown in split(x).items()}
            context = dataclasses.replace(
                built.context, densities=built.context.densities | {
                    names[i]: self._stages[i].density(
                        lights[i] | tables[i], scale) for i in tables})
            return GalaxyMass(model, built.params, context, None, {
                names[i]: table for i, table in tables.items()})

        if gravity.additive():
            # (that of the mass that does not take the layers, the same
            # in every pass)
            fixed = [c.name() for c in model.components()
                     if c.name() not in self._coupled]
            start = mass_of(np.zeros(ends[-1])).context
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                pinned = {
                    i: model.height_potential(
                        *grid, built.params, start, fixed)
                    for i, grid in grids.items()}

            def potentials(mass: GalaxyMass) -> dict[int, np.ndarray]:
                return {
                    i: pinned[i] + model.height_potential(
                        *grid, mass.params, mass.context, self._coupled)
                    for i, grid in grids.items()}
        else:
            def potentials(mass: GalaxyMass) -> dict[int, np.ndarray]:
                field = gravity.field(model, mass.params, mass.context)
                return {
                    i: field.height_potential(*grid)
                    for i, grid in grids.items()}

        def g(x: np.ndarray) -> np.ndarray:
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                found = potentials(mass_of(x))
            return np.concatenate([
                layers[i].unknown(potential).ravel()
                for i, potential in found.items()])

        def changes(x: np.ndarray, mapped: np.ndarray) -> tuple:
            # (the largest of the layers: its size, radius and name)
            new = split(mapped)
            return max(
                (*layers[i].change(unknown, new[i]), names[i])
                for i, unknown in split(x).items())

        result = fixed_point(
            g, np.zeros(ends[-1]), lambda x, mapped: changes(x, mapped)[0],
            _BALANCE_TOLERANCE, _BALANCE_PASSES)
        if not result.settled:
            size, radius, name = changes(result.x, g(result.x))
            why = (
                "its change grows: there is no equilibrium near (e.g. the "
                "local term of Palatini f(R) gravity collapses a layer)"
                if result.growing else
                f"the pdf of {name!r} still changes by {size:.1g} of its "
                f"peak at R = {radius:g} arcsec")
            raise ModelEvaluationError(
                f"the vertical equilibrium of the layers of "
                f"{', '.join(map(repr, names.values()))}, which the mass "
                f"follows, did not settle in {result.passes} passes: {why}")
        mass = mass_of(result.x)
        return dataclasses.replace(
            mass, field=gravity.field(model, mass.params, mass.context))

    def _field(self, view: ParamView, mass: GalaxyMass) -> GalaxyMass:
        """
        Return the mass of the galaxy with its field in the gravity of the
        galaxy, of the values of the parameters of the gravity and the mass
        as it is built (see _mass): with the values of the parameters of
        both.
        """
        params = mass.params | view
        return dataclasses.replace(
            mass, params=params, field=self._galaxy.gravity().field(
                mass.model, params, mass.context))

    def _sky(
            self,
            view: ParamView,
            mass: GalaxyMass,
            lights: dict[int, ParamValues]
    ) -> 'SkyMass':
        """
        Return the mass of the galaxy laid out on the sky (see SkyMass), of
        the values of the parameters that place the components where the
        mass is (see Placed.placement_params), the mass as it is built (see
        _mass) and the light of each component that takes part (see
        _light), by its index: each of them adds where it is to the
        placement (see Placed.place and SkyPlacement), at the one centre of
        the mass (that of the first, see _check_frames).
        """
        kpc_per_arcsec = mass.context.kpc_per_arcsec
        placement = _SkyPlacements()
        centre = None
        for i, stage in self._placed.items():
            # (all the values of the component: its light reads those of
            # a disk, which its rings take, and those that place it)
            values = lights[i] | self._own(view, i, stage.placement_params())
            if centre is None:
                first = stage.frames(values)[0]
                centre = (first.x * kpc_per_arcsec, first.y * kpc_per_arcsec)
            stage.place(self._labels[i], values, kpc_per_arcsec, placement)
        return mass.model.sky_mass(
            mass.params, mass.context, placement.placement(centre))

    def _groundwork(
            self,
            view: ParamView,
            index: int,
            light: ParamValues,
            mass: GalaxyMass | None
    ) -> object:
        """
        Return the groundwork of the state of the component of the given
        index (see ComponentStage.groundwork), of its light (see _light),
        in the mass of the galaxy; it reads no parameters (view is empty).
        """
        return self._stages[index].groundwork(light, mass)

    def _state(
            self,
            view: ParamView,
            index: int,
            light: ParamValues,
            groundwork: object,
            mass: GalaxyMass | None,
            out_extra: Extras | None
    ) -> 'ComponentState | None':
        """
        Return the state of the component of the given index (see
        ComponentStage.state), of the values of the parameters of the
        galaxy that it reads, its light (see _light) and its groundwork,
        in the mass of the galaxy.
        """
        stage = self._stages[index]
        own = self._own(view, index, stage.state_params())
        return stage.state(own | light, groundwork, mass, out_extra)

    def _own(
            self, view: ParamValues, index: int, names: tuple[str, ...]
    ) -> ParamView:
        """
        Return the values of the parameters of the given names of the
        component of the given index (its own names), from those of the
        galaxy (e.g. a view of them).
        """
        mapping = self._galaxy.component_param_names(index)
        return ParamView((name, view[mapping[name]]) for name in names)

    def _check_frames(self, params: ParamValues, warn: bool) -> None:
        """
        Check where the gravity takes the components that the mass moves,
        holds up or follows (see Placed.frames): at one centre, that of
        the mass; and with warn, log a warning if their planes tilt from
        that of the first part of the first by more than _MAX_TILT.

        Raises
        ------
        ModelEvaluationError
            If their centres differ.
        """
        frames = [
            (self._labels[i], frame)
            for i, stage in self._placed.items()
            for frame in stage.frames(self._own(
                params, i, stage.placement_params()))]
        # (a galaxy of mass alone has none)
        if not frames:
            return
        ref, first = frames[0]
        for label, frame in frames[1:]:
            if not np.allclose(
                    (frame.x, frame.y), (first.x, first.y), rtol=0,
                    atol=_CENTRE_TOLERANCE):
                where = f" at the radius {frame.radius:g}" \
                    if frame.radius else ""
                raise ModelEvaluationError(
                    f"the mass of a galaxy has one centre, but the "
                    f"components it moves, holds up or follows have "
                    f"different centres: {label}{where} at ({frame.x:g}, "
                    f"{frame.y:g}), {ref} at ({first.x:g}, {first.y:g}); "
                    f"tie them (e.g. its xpos and ypos to those of {ref})")
        planes = [(label, f) for label, f in frames if f.posa is not None]
        if not warn or not planes:
            return
        ref, first = planes[0]
        tilt, label, radius = max(
            (plane_tilt((first.posa, first.incl), (f.posa, f.incl)), label,
             f.radius)
            for label, f in planes)
        if tilt > _MAX_TILT:
            where = f" at the radius {radius:g}" if radius else ""
            _log.warning(
                f"the gravity takes the components it moves, holds up and "
                f"follows as in one plane, but {label}{where} is tilted by "
                f"{tilt:.0f} degrees from the plane of {ref}: its circular "
                f"velocities can be off by 1% or more there")

class _SkyPlacements:
    """
    The placement of the mass of a galaxy on the sky as its components add
    themselves to it (see SkyPlacements and Placed.place): the rings of
    each disk and the axis of each spheroid, by name.
    """

    def __init__(self):
        self._disks: dict[str, RingPlacement] = {}
        self._spheroids: dict[str, SpheroidAxis] = {}

    def add_rings(self, name: str, rings: RingPlacement) -> None:
        """Add the rings of a disk (see RingPlacement), by name."""
        self._disks[name] = rings

    def add_axis(self, name: str, axis: SpheroidAxis) -> None:
        """Add the centre and axis of a spheroid, by name."""
        self._spheroids[name] = axis

    def placement(self, centre: tuple[float, float]) -> SkyPlacement:
        """
        Return the placement of the mass (see SkyPlacement) at its centre
        (kpc), of the disks and the spheroids added.
        """
        return SkyPlacement(centre, self._disks, self._spheroids)


def _coupled(mass_model: 'MassModel', layers: set[str]) -> tuple[str, ...]:
    """
    Return the names of the mass components that take the layers of the
    components of the given names (see GalaxyStage._balance): those that
    follow them, and those that respond to those (e.g. a halo contracted
    by a disk that follows one).
    """
    coupled = {
        component.name() for component in mass_model.components()
        if component.followed_layer() in layers}
    while more := {
            component.name() for component in mass_model.components()
            if coupled & set(component.responds_to())} - coupled:
        coupled |= more
    return tuple(
        component.name() for component in mass_model.components()
        if component.name() in coupled)


def _same_sky(new: 'SkyMass', last: 'SkyMass') -> bool:
    """
    Check whether a mass on the sky is the last one (see StageCache: same):
    of the same length of an arcsec, and part by part of layers of the
    same unit shapes (see SkyLayer.same_shape) and the same scales.
    """
    if new.kpc_per_arcsec != last.kpc_per_arcsec \
            or list(new.parts) != list(last.parts):
        return False
    for name, layers in new.parts.items():
        others = last.parts[name]
        if len(layers) != len(others) or not all(
                layer.scale == other.scale and layer.same_shape(other)
                for layer, other in zip(layers, others)):
            return False
    return True
