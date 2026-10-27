"""
The stage of a galaxy (GalaxyStage): what the galaxy computes once per
evaluation, for all its observations: the mass of the galaxy, with the
densities of the components it follows, and what each component takes
from the galaxy (see ComponentStage); each of its parts again only when
the parameters it reads, or the parts it takes, change.
"""

import dataclasses
import functools
import logging
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from gbkfit.driver import Driver
from gbkfit.params import ParamValues
from gbkfit.params._cache import ParamView, StageCache
from ..extras import Extras
from ..gravity import GravityField
from ..mass import MassContext, MassModel

if TYPE_CHECKING:
    from ._component_set import ComponentSet
    from .base import ComponentStage, ComponentState


__all__ = [
    'GalaxyMass',
    'GalaxyScale',
    'GalaxyStage',
    'GalaxyState'
]


_log = logging.getLogger(__name__)

# The offset of the centre of a ring from the reference centre, as a
# share of its radius, and the angle (degrees) between their planes,
# above which a warning says that the gravity takes the disks too simply
# (see GalaxyStage.evaluate). Measured on an exponential disk whose outer
# rings drift or tilt: each makes its own circular velocity about 1%
# wrong (2026-10-14: 1% at a drift of about 15% of the radius, and at a
# tilt of about 20 degrees).
_MAX_OFFSET = 0.15
_MAX_TILT = 20.0


@dataclasses.dataclass(frozen=True)
class GalaxyScale:
    """
    The scale of a galaxy in one evaluation, from its distance (see
    Distance): the length of an arcsec in the galaxy (kpc), and the
    critical density (Msun/kpc^3), the matter density parameter and the
    Ricci scalar (kpc^-2) of the universe at its epoch.
    """
    kpc_per_arcsec: float
    critical_density: float
    omega_m: float
    ricci_scalar: float


@dataclasses.dataclass(frozen=True)
class GalaxyMass:
    """
    The mass of a galaxy in one evaluation: its model, the values of the
    parameters of the mass and the gravity of the galaxy, the context of
    the mass, the length of an arcsec in the galaxy (kpc), and the field
    of the mass in the gravity of the galaxy (see Gravity).
    """
    model: MassModel
    params: ParamValues
    context: MassContext
    kpc_per_arcsec: float
    field: GravityField


@dataclasses.dataclass(frozen=True)
class GalaxyState:
    """
    A galaxy in one evaluation, for all its observations (see
    GalaxyStage.evaluate): its mass (None without mass), and what it gives
    each component that takes anything from it: its state, by the index
    of the component (see ComponentStage.state).
    """
    mass: GalaxyMass | None
    states: dict[int, 'ComponentState']


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
       of the components it follows (of their light), and its field in
       the gravity of the galaxy;
    4. what each component takes from the galaxy: the groundwork of its
       state (see ComponentStage.groundwork; e.g. the forces in the Jeans
       equations of a spheroid), and its state (see ComponentStage.state):
       e.g. the rotation of a disk in the gravity of the mass, or the
       dispersion of its vertical equilibrium.

    Each part reads the values of the parameters it declares alone: the
    scale those of the distance, the mass those of the mass and the
    gravity, and each part of a component those that its stage declares
    (see ComponentStage); and it takes the parts before it that it uses.
    An evaluation computes a part again only when the values it reads, or
    the parts it takes, change (see StageCache and evaluate): e.g. a step
    in the position angle of a spheroid of Jeans kinematics computes none
    of them again, nor a step in the anisotropy of its stars the
    groundwork of its state.

    It takes the rings of the disks that the mass moves, holds up and
    follows as concentric and flat: around one centre, in one plane,
    whatever their geometries (their rotation is then projected with
    their own geometries). With extra outputs, a warning says where their
    centres or planes depart from those of the first enough to matter
    (see evaluate).

    Parameters
    ----------
    component_set : ComponentSet
        The components of the galaxy, its distance and its mass.
    driver : Driver
        The driver where the components evaluate their parts.
    redshift : float
        The redshift of the galaxy (for its distance, see Distance).
    sky_origin : tuple of float, optional
        The RA and Dec (degrees) of the origin of the frame of the
        parameters of the galaxy (that of the first observation on the
        sky; see ObservationGroup), which image traits need (see
        ImageMaps).

    Raises
    ------
    ConfigError
        If a component cannot plan its part (e.g. it has an image trait,
        but there is no sky origin).
    """

    def __init__(
            self,
            component_set: 'ComponentSet',
            driver: Driver,
            redshift: float,
            sky_origin: tuple[float, float] | None = None
    ):
        components = component_set.components()
        mappings = component_set.mappings()
        mass_model = component_set.mass_model()
        followed = set() if mass_model is None else {
            name for mass_component in mass_model.components()
            for name in mass_component.followed_components()}
        self._component_set = component_set
        self._redshift = redshift
        # The part of each component that takes part, by its index, and
        # its label
        self._stages: dict[int, 'ComponentStage'] = {}
        self._labels: dict[int, str] = {}
        for i, component in enumerate(components):
            label = component.name() or f'component{i}'
            stage = component.plan_stage(
                driver, sky_origin, component.name() in followed, label)
            if stage is not None:
                self._stages[i] = stage
                self._labels[i] = label
        # The components that the mass follows
        self._followed = tuple(
            i for i in self._stages if components[i].name() in followed)
        # The caches of its parts (see evaluate): of its scale and its
        # mass, if it has mass, and of each component that takes part, by
        # its index; they read the parameters of the galaxy
        lights = {
            i: StageCache(mappings[i][p] for p in stage.light_params())
            for i, stage in self._stages.items()}
        self._scale_cache: StageCache[GalaxyScale] | None = None
        self._mass_cache: StageCache[GalaxyMass] | None = None
        if mass_model is not None:
            self._scale_cache = StageCache(
                component_set.distance().pdescs())
            self._mass_cache = StageCache(
                (*mass_model.pdescs(), *component_set.gravity().pdescs()),
                (self._scale_cache, *(lights[i] for i in self._followed)))
        mass = () if self._mass_cache is None else (self._mass_cache,)
        self._parts: dict[int, _ComponentParts] = {}
        for i, stage in self._stages.items():
            groundwork = StageCache((), (lights[i], *mass))
            state = StageCache(
                (mappings[i][p] for p in stage.state_params()),
                (lights[i], groundwork, *mass))
            self._parts[i] = _ComponentParts(lights[i], groundwork, state)

    def caches(self) -> dict[str, StageCache[Any]]:
        """
        Return the caches of its parts (see evaluate), by name: 'scale' and
        'mass', if it has mass, and '{component}.light',
        '{component}.groundwork' and '{component}.state' of each component
        that takes part, by its name, or its position (e.g. 'component0').
        """
        caches: dict[str, StageCache[Any]] = {}
        if self._mass_cache is not None:
            caches |= dict(scale=self._scale_cache, mass=self._mass_cache)
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
            the rings of the disks it moves, holds up and follows depart
            from the centre or the plane of the first ring of the first: by
            more than 15% of their radius, or 20 degrees (which makes their
            circular velocities about 1% wrong, or more).

        Returns
        -------
        GalaxyState
            The galaxy.
        """
        lights = {
            i: parts.light.evaluate(
                params, functools.partial(self._light, index=i))
            for i, parts in self._parts.items()}
        mass = None
        if self._mass_cache is not None:
            scale = self._scale_cache.evaluate(params, self._scale)
            mass = self._mass_cache.evaluate(params, functools.partial(
                self._mass, scale=scale, lights=lights))
            if out_extra is not None:
                self._warn_geometry(params)
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
        return GalaxyState(mass, states)

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
        return self._mass(
            self._mass_cache.view(params),
            self._scale(self._scale_cache.view(params)), lights)

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

    def _scale(self, view: ParamView) -> GalaxyScale:
        """
        Return the scale of the galaxy, of the values of the parameters of
        its distance.
        """
        distance, redshift = self._component_set.distance(), self._redshift
        return GalaxyScale(
            distance.kpc_per_arcsec(view, redshift),
            distance.critical_density(view, redshift),
            distance.omega_m(view, redshift),
            distance.ricci_scalar(view, redshift))

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
            scale: GalaxyScale,
            lights: dict[int, ParamValues]
    ) -> GalaxyMass:
        """
        Return the mass of the galaxy, of the values of the parameters of
        its mass and its gravity, its scale, and the light of each
        component that it follows (see _light), by its index.
        """
        component_set = self._component_set
        components = component_set.components()
        mass_model = component_set.mass_model()
        distance = component_set.distance()
        context = MassContext(
            scale.critical_density, scale.omega_m, {
                components[i].name(): self._stages[i].density(
                    lights[i], scale.kpc_per_arcsec)
                for i in self._followed},
            baryon_fraction=distance.baryon_fraction(),
            ricci_scalar=scale.ricci_scalar,
            ricci_scalar_today=distance.ricci_scalar_today(),
            kpc_per_arcsec=scale.kpc_per_arcsec)
        return GalaxyMass(
            mass_model, view, context, scale.kpc_per_arcsec,
            component_set.gravity().field(mass_model, view, context))

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
            self, view: ParamView, index: int, names: tuple[str, ...]
    ) -> ParamView:
        """
        Return the values of the parameters of the given names of the
        component of the given index (its own names), from a view of those
        of the galaxy.
        """
        mapping = self._component_set.mappings()[index]
        return ParamView((name, view[mapping[name]]) for name in names)

    def _warn_geometry(self, params: ParamValues) -> None:
        """
        Log a warning if the rings that the gravity takes as concentric and
        flat (see ComponentStage.rings) depart from the centre or the plane
        of the first ring of the first by more than _MAX_OFFSET of their
        radius (on the sky), or _MAX_TILT.
        """
        components = self._component_set.components()
        mappings = self._component_set.mappings()
        rings = [
            (self._labels[i], *ring)
            for i, stage in self._stages.items()
            for ring in stage.rings({
                p: params[mappings[i][p]] for p in components[i].pdescs()})]
        # (a galaxy of mass alone has none)
        if not rings:
            return
        ref, _, x0, y0, posa0, incl0 = rings[0]
        normal0 = _normal(posa0, incl0)
        offset = max(
            (np.hypot(x - x0, y - y0) / r, label, r)
            for label, r, x, y, _, _ in rings if r > 0)
        tilt = max(
            (np.degrees(np.arccos(min(1.0, abs(
                _normal(posa, incl) @ normal0)))), label, r)
            for label, r, _, _, posa, incl in rings)
        if offset[0] > _MAX_OFFSET:
            _log.warning(
                f"the gravity takes the disks it moves, holds up and "
                f"follows as concentric, but the centre of the ring of "
                f"{offset[1]} at the radius {offset[2]:g} is "
                f"{offset[0]:.0%} of its radius from that of the first ring "
                f"of {ref} (on the sky): their circular velocities can be "
                f"off by 1% or more there")
        if tilt[0] > _MAX_TILT:
            _log.warning(
                f"the gravity takes the disks it moves, holds up and "
                f"follows as flat, but the ring of {tilt[1]} at the radius "
                f"{tilt[2]:g} is tilted by {tilt[0]:.0f} degrees from the "
                f"plane of the first ring of {ref}: their circular "
                f"velocities can be off by 1% or more there")


def _normal(posa: float, incl: float) -> np.ndarray:
    """
    Return the normal of the plane of a ring of the given position angle
    and inclination (degrees): a unit vector, whose angle to that of
    another ring is the angle between their planes (or its supplement).
    """
    posa, incl = np.radians(posa), np.radians(incl)
    return np.array([
        np.sin(incl) * np.cos(posa), np.sin(incl) * np.sin(posa),
        np.cos(incl)])
