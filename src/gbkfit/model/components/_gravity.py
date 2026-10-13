"""
The gravity of a galaxy (GravityPlan): the circular velocity of its mass
at the radii of the components it moves, evaluated once for all the
observations of the galaxy.
"""

from typing import TYPE_CHECKING

import numpy as np

from gbkfit.driver import Driver
from gbkfit.params import ParamValues
from ..mass import MassContext
from .disks._profiles import DiskDensity, pressure_support

if TYPE_CHECKING:
    from ._component_set import ComponentSet


__all__ = [
    'GravityPlan'
]


class GravityPlan:
    """
    The gravity of a galaxy: the circular velocity of its mass at the
    radii of the components it moves (those with the velocity trait
    'mass', see Component.circular_velocity_params), with the pressure
    support of each where its traits ask for it (see
    Disk.pressure_supported_params).

    It is evaluated once for all the observations of the galaxy, on the
    given driver and in float64, whatever the drivers and dtypes of the
    observations (pressure support differentiates the profiles of a disk,
    which float32 makes noisy): the disks of the components that its mass
    follows (see DiskDensity), and of those with pressure support, sample
    their profiles there; the rest is on the host.

    Parameters
    ----------
    component_set : ComponentSet
        The components of the galaxy, its distance and its mass.
    driver : Driver
        The driver where the disks sample their profiles.
    """

    def __init__(self, component_set: 'ComponentSet', driver: Driver):
        components = component_set.components()
        mass_model = component_set.mass_model()
        followed = set() if mass_model is None else {
            name for mass_component in mass_model.components()
            for name in mass_component.followed_components()}
        self._component_set = component_set
        self._moved = tuple(
            i for i, cmp in enumerate(components)
            if cmp.circular_velocity_params())
        self._followed = tuple(
            i for i, cmp in enumerate(components) if cmp.name() in followed)
        planned = sorted(set(self._followed) | {
            i for i in self._moved
            if components[i].density_disk().pressure_supported_params()})
        self._disk_plans = {
            i: components[i].density_disk().plan(driver, 0, np.float64)
            for i in planned}

    def evaluate(self, params: ParamValues) -> dict[int, ParamValues]:
        """
        Evaluate the gravity.

        Parameters
        ----------
        params : ParamValues
            The values of the parameters of the galaxy (see Galaxy.pdescs).

        Returns
        -------
        dict of int to ParamValues
            The values of the parameters that the gravity gives each
            component it moves (see Component.circular_velocity_params),
            by the index of the component: the circular velocity at their
            radii (km/s), NaN where its square is negative or infinite
            (there is no circular orbit); none without mass.
        """
        component_set = self._component_set
        mass_model = component_set.mass_model()
        if mass_model is None:
            return {}
        components = component_set.components()
        distance = component_set.distance()
        own = {i: self._component_params(i, params) for i in self._disk_plans}
        kpc_per_arcsec = distance.kpc_per_arcsec(params)
        context = MassContext(
            distance.critical_density(params), distance.omega_m(params), {
                components[i].name(): DiskDensity(
                    components[i].density_disk(), self._disk_plans[i],
                    own[i], kpc_per_arcsec)
                for i in self._followed})
        # The mass is evaluated once, at the radii of all the components
        radii = {i: components[i].circular_velocity_params()
                 for i in self._moved}
        every = [r for cmp_radii in radii.values() for r in cmp_radii.values()]
        vcirc2 = mass_model.vcirc2(
            np.concatenate(every) * kpc_per_arcsec, params, context)
        pieces = iter(np.split(vcirc2, np.cumsum([len(r) for r in every])))
        values = {}
        for i, cmp_radii in radii.items():
            disk = components[i].density_disk()
            values[i] = {}
            for name, r in cmp_radii.items():
                v2 = next(pieces)
                if name in disk.pressure_supported_params():
                    v2 = v2 + pressure_support(
                        disk, self._disk_plans[i], own[i], r)
                values[i][name] = np.sqrt(
                    np.where(np.isfinite(v2) & (v2 >= 0), v2, np.nan))
        return values

    def _component_params(
            self, index: int, params: ParamValues
    ) -> ParamValues:
        """
        Return the values of the parameters of the component of the given
        index (by its own names), from those of the galaxy.
        """
        component = self._component_set.components()[index]
        mapping = self._component_set.mappings()[index]
        return {p: params[mapping[p]] for p in component.pdescs()}
