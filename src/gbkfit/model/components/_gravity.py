"""
The gravity of a galaxy (GravityPlan): the circular velocity of its mass
at the radii of the components it moves, and the dispersion of the disks
it holds up (their vertical equilibrium), evaluated once for all the
observations of the galaxy.
"""

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

import astropy.units as u
import numpy as np
from astropy.table import Table

from gbkfit.driver import Driver
from gbkfit.params import ParamValues
from ..mass import MassContext, MassModel
from ..mass.units import G
from ._units import mass_per_unit
from .disks._disk import EQUILIBRIUM_DISPERSION
from .disks._profiles import (
    DiskDensity, jeans_dispersions, pressure_support, rms_heights, rotation)
from .traits import PressureSupport

if TYPE_CHECKING:
    from ._component_set import ComponentSet
    from .disks._disk import Disk, DiskPlan


__all__ = [
    'GravityPlan'
]


_log = logging.getLogger(__name__)

# The offset of the centre of a ring from the reference centre, as a
# share of its radius, and the angle (degrees) between their planes,
# above which a warning says that the gravity takes the disks too simply
# (see GravityPlan.evaluate). Measured on an exponential disk whose outer
# rings drift or tilt: each makes its own circular velocity about 1%
# wrong (2026-10-14: 1% at a drift of about 15% of the radius, and at a
# tilt of about 20 degrees).
_MAX_OFFSET = 0.15
_MAX_TILT = 20.0


@dataclasses.dataclass(frozen=True)
class _Mass:
    """
    The mass of a galaxy in one evaluation: its model, the values of the
    parameters of the galaxy, the context of the mass, and the length of
    an arcsec in the galaxy (kpc).
    """
    model: MassModel
    params: ParamValues
    context: MassContext
    kpc_per_arcsec: float


class GravityPlan:
    """
    The gravity of a galaxy: the circular velocity of its mass at the
    radii of the components it moves (those with the velocity trait
    'mass', see Component.circular_velocity_params), with the pressure
    support of each where its traits ask for it (see
    Disk.pressure_supported_params); and the dispersion of the vertical
    equilibrium of the disks it holds up, at their subnodes (those with
    an equilibrium, see Kinematics): in the circular velocity of the mass
    ('central'; without mass, that of the rotation of each disk), or in
    the surface density of a mass component ('self_gravitating').

    It takes the disks it moves and follows as concentric and flat: their
    rings around one centre, in one plane, whatever their geometries (the
    rotation of the rings is then projected with their own geometries).
    With extra outputs, a warning says where their centres or planes
    depart from those of the first enough to matter (see evaluate).

    It is evaluated once for all the observations of the galaxy, on the
    given driver and in float64, whatever the drivers and dtypes of the
    observations (pressure support differentiates the profiles of a disk,
    which float32 makes noisy): the disks of the components that its mass
    follows (see DiskDensity), of those with pressure support and of
    those it holds up sample their profiles there; the rest is on the
    host.

    Parameters
    ----------
    component_set : ComponentSet
        The components of the galaxy, its distance and its mass.
    driver : Driver
        The driver where the disks sample their profiles.
    redshift : float
        The redshift of the galaxy (for its distance, see Distance).
    """

    def __init__(
            self,
            component_set: 'ComponentSet',
            driver: Driver,
            redshift: float
    ):
        components = component_set.components()
        mass_model = component_set.mass_model()
        followed = set() if mass_model is None else {
            name for mass_component in mass_model.components()
            for name in mass_component.followed_components()}
        self._component_set = component_set
        self._redshift = redshift
        self._moved = tuple(
            i for i, cmp in enumerate(components)
            if cmp.circular_velocity_params())
        self._followed = tuple(
            i for i, cmp in enumerate(components) if cmp.name() in followed)
        self._balanced = tuple(
            i for i, cmp in enumerate(components)
            if (disk := cmp.density_disk()) is not None
            and disk.equilibrium() is not None)
        # (the mass of a unit of the density of each, if it has a unit)
        self._mass_per_unit = {
            i: mass_per_unit(components[i].emission()) for i in self._followed}
        planned = sorted(set(self._followed) | set(self._balanced) | {
            i for i in self._moved
            if components[i].density_disk().pressure_supported_params()})
        self._disk_plans = {
            i: components[i].density_disk().plan(driver, 0, np.float64)
            for i in planned}

    def evaluate(
            self,
            params: ParamValues,
            out_extra: dict[str, Any] | None = None
    ) -> dict[int, ParamValues]:
        """
        Evaluate the gravity.

        Parameters
        ----------
        params : ParamValues
            The values of the parameters of the galaxy (see Galaxy.pdescs).
        out_extra : dict, optional
            Where the extra outputs go, if wanted: for each parameter that
            it gives a component (see Component.circular_velocity_params),
            '{component}_{parameter}_rotation', a table of its radii, in
            the plane of the disk at its midplane ('radius', arcsec;
            'radius_kpc'), the circular velocity of each component of the
            mass ('vcirc_{name}', km/s; signed: negative where the
            component pulls outwards) and of all ('vcirc'), and with
            pressure support, its term ('pressure', (km/s)^2), and the
            values given ('vrot', km/s); for each disk it holds up,
            '{component}_equilibrium', a table of its subnodes ('radius',
            arcsec), its rms height there ('height', arcsec), what its
            equilibrium takes there: the circular velocity ('vcirc', km/s;
            'central') or the surface density ('surface_density',
            Msun/kpc^2; 'self_gravitating'; none for 'jeans'), and the
            dispersion of its equilibrium ('sigma', km/s). The components
            are named after
            their names, or their positions ('component0'). With them and
            mass, a warning says where the rings of the disks it moves,
            holds up and follows depart from the centre or the plane of
            the first ring of the first: by more than 15% of their radius,
            or 20 degrees (which makes their circular velocities about 1%
            wrong, or more).

        Returns
        -------
        dict of int to ParamValues
            The values that the gravity gives each component it moves or
            holds up, by the index of the component: the circular
            velocity at the radii of its parameters (km/s, see
            Component.circular_velocity_params), NaN where its square is
            negative or infinite (there is no circular orbit); and the
            dispersion of the equilibrium of its disk at its subnodes
            (km/s, as EQUILIBRIUM_DISPERSION, see Kinematics).
        """
        component_set = self._component_set
        mass_model = component_set.mass_model()
        own = {i: self._component_params(i, params) for i in self._disk_plans}
        if mass_model is None:
            return {
                i: {EQUILIBRIUM_DISPERSION: self._equilibrium(
                    i, own[i], None, None, out_extra)}
                for i in self._balanced}
        components = component_set.components()
        distance = component_set.distance()
        if out_extra is not None:
            self._warn_geometry(params)
        redshift = self._redshift
        kpc_per_arcsec = distance.kpc_per_arcsec(params, redshift)
        context = MassContext(
            distance.critical_density(params, redshift),
            distance.omega_m(params, redshift), {
                components[i].name(): DiskDensity(
                    components[i].density_disk(), self._disk_plans[i],
                    own[i], kpc_per_arcsec,
                    self._mass_per_unit[i])
                for i in self._followed})
        mass = _Mass(mass_model, params, context, kpc_per_arcsec)
        # The mass is evaluated once, at the radii of all the components:
        # those of the parameters it gives, then the subnodes of the disks
        # it holds up in a central equilibrium
        radii = {i: components[i].circular_velocity_params()
                 for i in self._moved}
        central = [
            i for i in self._balanced
            if components[i].density_disk().equilibrium().mode == 'central']
        every = [r for cmp_radii in radii.values() for r in cmp_radii.values()]
        nmoved = len(every)
        every += [components[i].density_disk().subrnodes() for i in central]
        ends = np.cumsum([len(r) for r in every])[:-1]
        if every:
            parts = mass_model.vcirc2_components(
                np.concatenate(every) * kpc_per_arcsec, params, context)
            # (the sum of the parts is the vcirc2 of the mass model)
            vcirc2 = np.split(sum(parts.values()), ends)
        else:
            # (only disks in a self-gravitating equilibrium)
            parts, vcirc2 = {}, []
        # The equilibria first: the pressure support of their disks takes
        # their dispersion
        central_vcirc2 = dict(zip(central, vcirc2[nmoved:]))
        values = {
            i: {EQUILIBRIUM_DISPERSION: self._equilibrium(
                i, own[i], mass, central_vcirc2.get(i), out_extra)}
            for i in self._balanced}
        moved_vcirc2 = iter(vcirc2[:nmoved])
        part_pieces = {
            mass_name: iter(np.split(part, ends))
            for mass_name, part in parts.items()}
        for i, cmp_radii in radii.items():
            disk = components[i].density_disk()
            values.setdefault(i, {})
            for name, r in cmp_radii.items():
                v2 = next(moved_vcirc2)
                pressure = None
                if support := disk.pressure_supported_params().get(name):
                    density = None if support.density is None \
                        else _pressure_density(
                            mass_model, support,
                            np.asarray(r, dtype=float) * kpc_per_arcsec,
                            params, context)
                    pressure = pressure_support(
                        disk, self._disk_plans[i], own[i], r, support.mode,
                        density, values[i].get(EQUILIBRIUM_DISPERSION))
                    v2 = v2 + pressure
                values[i][name] = np.sqrt(
                    np.where(np.isfinite(v2) & (v2 >= 0), v2, np.nan))
                cmp_parts = {
                    mass_name: next(pieces)
                    for mass_name, pieces in part_pieces.items()}
                if out_extra is not None:
                    label = components[i].name() or f'component{i}'
                    out_extra[f'{label}_{name}_rotation'] = _rotation_table(
                        np.asarray(r, dtype=float), kpc_per_arcsec,
                        cmp_parts, pressure, values[i][name])
        return values

    def _equilibrium(
            self,
            index: int,
            params: ParamValues,
            mass: _Mass | None,
            vcirc2: np.ndarray | None,
            out_extra: dict[str, Any] | None
    ) -> np.ndarray:
        """
        Return the dispersion of the vertical equilibrium (km/s) of the
        disk of the component of the given index at its subnodes (see
        Kinematics), from the values of its parameters (by its own names)
        and the mass of the galaxy (None without mass); for a central
        equilibrium with mass, from the squared circular velocity of the
        mass at the subnodes ((km/s)^2), and without, from that of the
        rotation of the disk. With extra outputs, its table (see
        evaluate).
        """
        component = self._component_set.components()[index]
        disk = component.density_disk()
        equilibrium = disk.equilibrium()
        plan = self._disk_plans[index]
        radii = np.asarray(disk.subrnodes())
        heights = rms_heights(disk, plan, params, radii)
        if equilibrium.mode == 'central':
            if mass is None:
                vcirc2 = rotation(disk, plan, params, radii) ** 2
            sigma = _central_dispersion(radii, heights, vcirc2)
            taken = dict(vcirc=_signed_sqrt(vcirc2) * u.km / u.s)
        elif equilibrium.mode == 'jeans':
            sigma = _jeans_dispersion(disk, plan, params, radii, mass)
            taken = {}
        else:
            name = equilibrium.density
            surface = mass.model.component(name).surface_density(
                radii * mass.kpc_per_arcsec,
                mass.model.component_params(name, mass.params), mass.context)
            sigma = _self_gravitating_dispersion(
                heights * mass.kpc_per_arcsec, surface)
            taken = dict(surface_density=surface * u.solMass / u.kpc ** 2)
        if out_extra is not None:
            label = component.name() or f'component{index}'
            out_extra[f'{label}_equilibrium'] = _equilibrium_table(
                radii, heights, taken, sigma)
        return sigma

    def _warn_geometry(self, params: ParamValues) -> None:
        """
        Log a warning if the rings of the disks it moves, holds up and
        follows depart from the centre or the plane of the first ring of
        the first by more than _MAX_OFFSET of their radius (on the sky),
        or _MAX_TILT.
        """
        components = self._component_set.components()
        rings = []
        for i in dict.fromkeys(self._moved + self._balanced + self._followed):
            values = self._component_params(i, params)
            radii = np.asarray(components[i].density_disk().rnodes())
            geometry = np.broadcast_arrays(radii, *(
                np.asarray(values[name], dtype=float)
                for name in ('xpos', 'ypos', 'posa', 'incl')))
            label = components[i].name() or f'component{i}'
            rings += [(label, *ring) for ring in zip(*geometry)]
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


def _pressure_density(
        mass_model: MassModel,
        support: PressureSupport,
        R: np.ndarray,
        params: ParamValues,
        context: MassContext
) -> np.ndarray:
    """
    Return the density of the mass component of a pressure support (see
    traits.VPTraitMass) at the radii R (kpc): at its midplane for
    'jeans', its surface density for 'self_gravitating'.
    """
    component = mass_model.component(support.density)
    own = mass_model.component_params(support.density, params)
    if support.mode == 'jeans':
        return component.density(R, np.zeros_like(R), own, context)
    return component.surface_density(R, own, context)


def _central_dispersion(
        radii: np.ndarray, heights: np.ndarray, vcirc2: np.ndarray
) -> np.ndarray:
    """
    Return the dispersion (km/s) of the central equilibrium (see
    Kinematics) at the radii (arcsec) of the given rms heights (arcsec)
    and squared circular velocities ((km/s)^2): h Omega, Omega^2 = vcirc2
    / R^2; at the centre, where it is 0 / 0, Omega at the next radius;
    NaN where vcirc2 is negative or infinite.
    """
    omega2 = np.divide(
        vcirc2, radii ** 2, out=np.zeros_like(vcirc2), where=radii > 0)
    if radii[0] == 0:
        omega2[0] = omega2[1]
    return heights * np.sqrt(
        np.where(np.isfinite(omega2) & (omega2 >= 0), omega2, np.nan))


def _jeans_dispersion(
        disk: 'Disk',
        plan: 'DiskPlan',
        params: ParamValues,
        radii: np.ndarray,
        mass: _Mass
) -> np.ndarray:
    """
    Return the dispersion (km/s) of the jeans equilibrium (see Kinematics)
    of a disk at its subnodes (radii, arcsec), in the vertical force of
    the mass of its galaxy (see jeans_dispersions); at the centre, the
    value at the next subnode.
    """
    kpc = mass.kpc_per_arcsec

    def vertical_force(R: np.ndarray, z: np.ndarray) -> np.ndarray:
        R, z = np.broadcast_arrays(R, z)
        return mass.model.vertical_force(
            R * kpc, z * kpc, mass.params, mass.context) * kpc
    off = radii > 0
    sigma = np.empty_like(radii)
    sigma[off] = jeans_dispersions(
        disk, plan, params, radii[off], vertical_force)
    sigma[~off] = sigma[off][0]
    return sigma


def _self_gravitating_dispersion(
        heights: np.ndarray, surface: np.ndarray
) -> np.ndarray:
    """
    Return the dispersion (km/s) of the self-gravitating equilibrium (see
    Kinematics) of the given rms heights (kpc) and surface densities
    (Msun/kpc^2): sqrt(2 sqrt(3) G Sigma h); NaN where Sigma is negative.
    """
    sigma2 = 2 * np.sqrt(3) * G * surface * heights
    return np.sqrt(np.where(sigma2 >= 0, sigma2, np.nan))


def _signed_sqrt(v2: np.ndarray) -> np.ndarray:
    """
    Return the signed square root of squared velocities: negative where
    they are.
    """
    return np.sign(v2) * np.sqrt(np.abs(v2))


def _equilibrium_table(
        radii: np.ndarray,
        heights: np.ndarray,
        taken: dict[str, u.Quantity],
        sigma: np.ndarray
) -> Table:
    """
    Return the table of the vertical equilibrium of a disk at its subnodes
    (radii, arcsec): its rms heights (arcsec), the columns of what it
    takes, by name, and its dispersion (km/s; see GravityPlan.evaluate).
    """
    table = Table()
    table['radius'] = radii * u.arcsec
    table['height'] = heights * u.arcsec
    for name, column in taken.items():
        table[name] = column
    table['sigma'] = sigma * u.km / u.s
    return table


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


def _rotation_table(
        radii: np.ndarray,
        kpc_per_arcsec: float,
        parts: dict[str, np.ndarray],
        pressure: np.ndarray | None,
        rotation: np.ndarray
) -> Table:
    """
    Return the table of the rotation of a component at the given radii
    (arcsec): the square of the circular velocity of each component of
    the mass (parts, by name), its pressure support (if any) and its
    rotation (see GravityPlan.evaluate).
    """
    kms = u.km / u.s
    table = Table()
    table['radius'] = radii * u.arcsec
    table['radius_kpc'] = radii * kpc_per_arcsec * u.kpc
    for name, part in parts.items():
        table[f'vcirc_{name}'] = _signed_sqrt(part) * kms
    table['vcirc'] = _signed_sqrt(sum(parts.values())) * kms
    if pressure is not None:
        table['pressure'] = pressure * kms ** 2
    table['vrot'] = rotation * kms
    return table
