"""
The stage of a galaxy (GalaxyStage): what the galaxy computes once per
evaluation, for all its observations: the mass of the galaxy, with the
densities of the components it follows, and what each component takes
from the galaxy (see ComponentStage).
"""

import dataclasses
import logging
from typing import TYPE_CHECKING

import numpy as np

from gbkfit.driver import Driver
from gbkfit.params import ParamValues
from ..extras import Extras
from ..gravity import GravityField
from ..mass import MassContext, MassModel

if TYPE_CHECKING:
    from ._component_set import ComponentSet
    from .base import ComponentStage, ComponentState


__all__ = [
    'GalaxyMass',
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
class GalaxyMass:
    """
    The mass of a galaxy in one evaluation: its model, the values of the
    parameters of the galaxy, the context of the mass, the length of an
    arcsec in the galaxy (kpc), and the field of the mass in the gravity
    of the galaxy (see Gravity).
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
    GalaxyStage.evaluate): its mass (None without mass), what it gives
    each component that takes anything from it, by the index of the
    component (see ComponentStage.given), and the state of each
    component that has one (see ComponentStage.state).
    """
    mass: GalaxyMass | None
    given: dict[int, ParamValues]
    states: dict[int, 'ComponentState']


class GalaxyStage:
    """
    The stage of a galaxy: what it computes once per evaluation, for all
    its observations, on the given driver and in float64, in three steps:

    1. the values of the parameters of each component that takes part
       (see Component.plan_stage), with what it makes before the mass
       (e.g. the light of image traits; see ComponentStage.values);
    2. the mass of the galaxy (see mass): its context, with the densities
       of the components it follows, and its field in the gravity of the
       galaxy;
    3. what each component takes from the galaxy (see
       ComponentStage.given): e.g. the rotation of a disk in the gravity
       of the mass, or the dispersion of its vertical equilibrium.

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

    def evaluate(
            self,
            params: ParamValues,
            out_extra: Extras | None = None
    ) -> GalaxyState:
        """
        Evaluate the galaxy: its mass, and what it gives its components.

        Parameters
        ----------
        params : ParamValues
            The values of the parameters of the galaxy (see Galaxy.pdescs).
        out_extra : Extras, optional
            Where the extra outputs go, if wanted (those of the components,
            with their name, or their position, as the id 'component', see
            given). With them and mass, a warning says where the rings of
            the disks it moves, holds up and follows depart from the
            centre or the plane of the first ring of the first: by more
            than 15% of their radius, or 20 degrees (which makes their
            circular velocities about 1% wrong, or more).

        Returns
        -------
        GalaxyState
            The galaxy.
        """
        values = self.values(params)
        mass = self.mass(params, values)
        if out_extra is not None and mass is not None:
            self._warn_geometry(values)
        return GalaxyState(
            mass, self.given(values, mass, out_extra),
            self.states(values, mass))

    def values(self, params: ParamValues) -> dict[int, ParamValues]:
        """
        Return the values of the parameters of each component that takes
        part, by its own names, by its index (see ComponentStage.values),
        from those of the galaxy.
        """
        components = self._component_set.components()
        mappings = self._component_set.mappings()
        return {
            i: stage.values(
                {p: params[mappings[i][p]] for p in components[i].pdescs()})
            for i, stage in self._stages.items()}

    def mass(
            self,
            params: ParamValues,
            values: dict[int, ParamValues] | None = None
    ) -> GalaxyMass | None:
        """
        Return the mass of the galaxy at the values of its parameters (see
        Galaxy.pdescs) and of those of its components (see values; made
        here by default): its mass model, the values, its context (with
        the densities of the components it follows) and the length of an
        arcsec in it; None without mass.
        """
        component_set = self._component_set
        mass_model = component_set.mass_model()
        if mass_model is None:
            return None
        if values is None:
            values = self.values(params)
        components = component_set.components()
        distance = component_set.distance()
        redshift = self._redshift
        kpc_per_arcsec = distance.kpc_per_arcsec(params, redshift)
        context = MassContext(
            distance.critical_density(params, redshift),
            distance.omega_m(params, redshift), {
                components[i].name(): self._stages[i].density(
                    values[i], kpc_per_arcsec)
                for i in self._followed},
            baryon_fraction=distance.baryon_fraction(),
            ricci_scalar=distance.ricci_scalar(params, redshift),
            ricci_scalar_today=distance.ricci_scalar_today(),
            kpc_per_arcsec=kpc_per_arcsec)
        return GalaxyMass(
            mass_model, params, context, kpc_per_arcsec,
            component_set.gravity().field(mass_model, params, context))

    def given(
            self,
            values: dict[int, ParamValues],
            mass: GalaxyMass | None,
            out_extra: Extras | None = None
    ) -> dict[int, ParamValues]:
        """
        Return what the galaxy gives each component that takes part, by its
        index (see ComponentStage.given), at the values of their
        parameters (see values) and in the mass of the galaxy (see mass;
        None without mass). The extra outputs of each component, if
        wanted, have its name, or its position, as the id 'component'.
        """
        return {
            i: stage.given(
                values[i], mass, None if out_extra is None
                else out_extra.scoped(component=self._labels[i]))
            for i, stage in self._stages.items()}

    def states(
            self,
            values: dict[int, ParamValues],
            mass: GalaxyMass | None
    ) -> dict[int, 'ComponentState']:
        """
        Return the state of each component that has one, by its index (see
        ComponentStage.state), at the values of their parameters (see
        values) and in the mass of the galaxy (see mass; None without
        mass).
        """
        states = {
            i: stage.state(values[i], mass)
            for i, stage in self._stages.items()}
        return {i: state for i, state in states.items() if state is not None}

    def _warn_geometry(self, values: dict[int, ParamValues]) -> None:
        """
        Log a warning if the rings that the gravity takes as concentric and
        flat (see ComponentStage.rings) depart from the centre or the plane
        of the first ring of the first by more than _MAX_OFFSET of their
        radius (on the sky), or _MAX_TILT.
        """
        rings = [
            (self._labels[i], *ring)
            for i, stage in self._stages.items()
            for ring in stage.rings(values[i])]
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
