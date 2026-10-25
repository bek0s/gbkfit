"""
The part of a disk in the stage of its galaxy (DiskStage): the face-on
light of its image traits, its density for the mass that follows it, and
what its galaxy gives it: the dispersion of its vertical equilibrium, its
rotation in the gravity of the mass (with its pressure support), the
factors of that rotation at heights, and the tables of its density waves.
"""

import astropy.units as u
import numpy as np
from astropy.table import Table

from gbkfit.driver import Driver
from gbkfit.params import ParamValues
from ...extras import Extras
from ...mass import MassContext, MassModel
from ...mass.units import G
from .._images import ImageMaps
from .._units import mass_per_unit
from .._stage import GalaxyMass
from ..base import ComponentStage
from ..roles import Emission
from ..traits import PressureSupport
from ._disk import EQUILIBRIUM_DISPERSION, NodalDisk, DiskPlan
from ._profiles import (
    NodalDiskDensity, density_waves, jeans_dispersions, pressure_support,
    rms_heights, rotation)


__all__ = [
    'DiskStage',
    'plan_disk_stage'
]


# The descriptions of the extra outputs (see DiskStage.given)
_ROTATION_DESCRIPTION = (
    "The rotation that the gravity of the galaxy gives a parameter of the "
    "component, at its radii in the plane of the disk (radius, arcsec; "
    "radius_kpc): the Newtonian circular velocity of each component of the "
    "mass (vcirc_<name>, km/s; negative where it pulls outwards) and of "
    "all (vnewton), the circular velocity in the gravity of the galaxy "
    "(vcirc), with pressure support its term (pressure, (km/s)^2), and the "
    "rotation given (vrot, km/s).")
_EQUILIBRIUM_DESCRIPTION = (
    "The vertical equilibrium of the disk of the component at its "
    "subnodes (radius, arcsec): its rms height (height, arcsec), what the "
    "equilibrium takes there (the circular velocity, vcirc, km/s, or the "
    "surface density, surface_density, Msun/kpc^2), and the velocity "
    "dispersion that holds it up (sigma, km/s).")

def plan_disk_stage(
        disk: NodalDisk,
        emission: Emission | None,
        driver: Driver,
        sky_origin: tuple[float, float] | None,
        followed: bool,
        label: str
) -> 'DiskStage | None':
    """
    Plan the part of a disk of the given emission in the stage of its
    galaxy (see Component.plan_stage), if it takes anything from the
    galaxy (the circular velocity of the mass, an equilibrium, density
    waves, the light of image traits) or the mass follows it; None
    otherwise.
    """
    takes = disk.circular_velocity_params() or disk.equilibrium() is not None \
        or disk.wave_params() or disk.image_params()
    if not (takes or followed):
        return None
    return DiskStage(
        disk, label, driver, sky_origin, followed,
        mass_per_unit(emission) if followed else None)


class DiskStage(ComponentStage):
    """
    The part of a disk in the stage of its galaxy (see GalaxyStage): the
    face-on light of its image traits (see ImageMaps), its density, which
    the mass of the galaxy can follow (see NodalDiskDensity), and what the
    galaxy gives it (see given): the circular velocity of the mass at the
    radii of the parameters of its velocity trait 'mass' (see
    NodalDisk.circular_velocity_params), with its pressure support where its
    traits ask for it (see NodalDisk.pressure_supported_params), and at each
    height, where they ask for it (the velocity height trait 'mass', see
    NodalDisk.rotation_factor_params); the dispersion of its vertical
    equilibrium at its subnodes, if it has one (see Kinematics): in the
    circular velocity of the mass ('central'; without mass, that of its
    rotation), in the vertical force of the mass ('jeans'), or in the
    surface density of a mass component ('self_gravitating'); and the
    tables of its density waves (the traits 'spiral', see
    NodalDisk.wave_params), in its rotation, at its subnodes.

    Its profiles are sampled on the given driver and in float64, whatever
    the drivers and dtypes of the observations (pressure support and
    density waves differentiate them, which float32 makes noisy), when the
    mass follows it, it has pressure support, density waves or an
    equilibrium; the rest is on the host.

    Parameters
    ----------
    disk : NodalDisk
        The disk.
    label : str
        The name of its component, or its position (e.g. 'component0'),
        for the messages.
    driver : Driver
        The driver where it samples its profiles.
    sky_origin : tuple of float, optional
        The RA and Dec (degrees) of the origin of the frame of the
        parameters of the galaxy, which its image traits need (see
        ImageMaps).
    followed : bool
        Whether the mass of its galaxy follows its density.
    mass_per_unit : float, optional
        The mass of a unit of its density (see mass_per_unit), if it has a
        unit.

    Raises
    ------
    ConfigError
        If it has an image trait, but there is no sky origin.
    """

    def __init__(
            self,
            disk: NodalDisk,
            label: str,
            driver: Driver,
            sky_origin: tuple[float, float] | None,
            followed: bool,
            mass_per_unit: float | None
    ):
        self._disk = disk
        self._label = label
        self._followed = followed
        self._mass_per_unit = mass_per_unit
        self._images = None if not disk.image_params() \
            else ImageMaps(disk, label, sky_origin)
        sampled = followed or disk.equilibrium() is not None \
            or disk.wave_params() or disk.pressure_supported_params()
        self._plan = disk.plan(driver, 0, np.float64) if sampled else None

    def values(self, own: ParamValues) -> ParamValues:
        """
        Return the values of its parameters (by its own names), with the
        face-on light of its image traits, if any.
        """
        if self._images is None:
            return own
        return own | self._images.values(own)

    def density(
            self, values: ParamValues, kpc_per_arcsec: float
    ) -> NodalDiskDensity:
        return NodalDiskDensity(
            self._disk, self._plan, values, kpc_per_arcsec,
            self._mass_per_unit)

    def given(
            self,
            values: ParamValues,
            mass: GalaxyMass | None,
            out_extra: Extras | None
    ) -> ParamValues:
        """
        Return what its galaxy gives it, by the names of the parameters
        (see the class; EQUILIBRIUM_DISPERSION for the dispersion of its
        equilibrium): the circular velocity at the radii of its
        parameters (km/s), 0 where its square is negative and NaN where it
        is infinite (see _rotation); the factors of its rotation at the
        heights of its velocity height trait 'mass', at its subnodes (see
        _rotation_factors); the dispersion of its equilibrium at its
        subnodes (km/s); the tables of its density waves; and the face-on
        light of its image traits. With extra outputs: for each parameter
        of its circular velocity, '{parameter}_rotation', a table of its
        radii, in the plane of the disk at its midplane ('radius', arcsec;
        'radius_kpc'), the Newtonian circular velocity of each component
        of the mass ('vcirc_{name}', km/s; signed: negative where the
        component pulls outwards) and of all ('vnewton'), the circular
        velocity of the mass in the gravity of the galaxy ('vcirc'; that of
        all with Newtonian gravity, see Gravity), and with pressure
        support, its term ('pressure', (km/s)^2), and the values given
        ('vrot', km/s); with an equilibrium, 'equilibrium', a table of its
        subnodes ('radius', arcsec), its rms height there ('height',
        arcsec), what its equilibrium takes there: the circular velocity
        ('vcirc', km/s; 'central') or the surface density
        ('surface_density', Msun/kpc^2; 'self_gravitating'; none for
        'jeans'), and the dispersion of its equilibrium ('sigma', km/s);
        and a warning if it has image traits and is near edge-on (see
        ImageMaps.warn_inclination).
        """
        disk = self._disk
        if out_extra is not None and self._images is not None:
            self._images.warn_inclination(values)
        given: ParamValues = {}
        equilibrium = None
        if disk.equilibrium() is not None:
            equilibrium = self._equilibrium(values, mass, out_extra)
            given[EQUILIBRIUM_DISPERSION] = equilibrium
        if mass is not None and disk.circular_velocity_params():
            for name, radii in disk.circular_velocity_params().items():
                given[name] = self._given_rotation(
                    name, radii, values, mass, equilibrium, out_extra)
            if heights := disk.rotation_factor_params():
                given |= self._rotation_factors(
                    heights, values, mass, equilibrium)
        if disk.wave_params():
            given |= density_waves(disk, self._plan, values | given)
        if self._images is not None:
            given |= {name: values[name] for name in disk.image_params()}
        return given

    def rings(self, values: ParamValues) -> list[tuple[float, ...]]:
        """
        Return the rings of its disk that the gravity takes as concentric
        and flat, if the mass follows it, moves it or holds it up: the
        radius, the centre and the plane of each (radius, xpos, ypos,
        posa, incl; arcsec and degrees); none otherwise.
        """
        disk = self._disk
        if not (self._followed or disk.circular_velocity_params()
                or disk.equilibrium() is not None):
            return []
        radii = np.asarray(disk.rnodes())
        frame = disk.frame_params()
        geometry = np.broadcast_arrays(radii, *(
            np.asarray(values[name], dtype=float)
            for name in (frame.x, frame.y, frame.posa, frame.incl)))
        return list(zip(*geometry))

    def _given_rotation(
            self,
            name: str,
            radii: tuple[float, ...],
            values: ParamValues,
            mass: GalaxyMass,
            equilibrium: np.ndarray | None,
            out_extra: Extras | None
    ) -> np.ndarray:
        """
        Return the rotation (km/s) that the gravity gives its parameter of
        the given name, at its radii (arcsec): of the circular velocity of
        the mass, and its pressure support, if any (with the dispersion of
        its equilibrium, if any). With extra outputs, its table (see
        given).
        """
        disk = self._disk
        kpc_per_arcsec = mass.kpc_per_arcsec
        R = np.asarray(radii, dtype=float) * kpc_per_arcsec
        vcirc2 = mass.field.vcirc2(R)
        v2 = vcirc2
        pressure = None
        if support := disk.pressure_supported_params().get(name):
            density = None if support.density is None else _pressure_density(
                mass.model, support, R, mass.params, mass.context)
            pressure = pressure_support(
                disk, self._plan, values, radii, support.mode, density,
                equilibrium)
            v2 = v2 + pressure
        result = _rotation(v2)
        if out_extra is not None:
            parts = mass.model.vcirc2_components(R, mass.params, mass.context)
            out_extra.add(
                f'{name}_rotation', _rotation_table(
                    np.asarray(radii, dtype=float), kpc_per_arcsec, parts,
                    vcirc2, pressure, result),
                f"Rotation curve ({name})", _ROTATION_DESCRIPTION)
        return result

    def _rotation_factors(
            self,
            heights: dict[str, float],
            values: ParamValues,
            mass: GalaxyMass,
            equilibrium: np.ndarray | None
    ) -> ParamValues:
        """
        Return the factors of its rotation at the heights of its velocity
        height trait 'mass' (heights, arcsec, by the names of their
        parameters), at its subnodes: v(R, z) / v(R, 0), v(R, z)^2 = -R
        F_R(R, z) in the gravity of the mass, plus its pressure support at
        z, if any (see traits.VHTraitMass), with the dispersion of its
        vertical equilibrium (if any), v as _rotation takes it; 1 where
        v(R, 0) is 0.
        """
        disk = self._disk
        kpc_per_arcsec = mass.kpc_per_arcsec
        r = np.asarray(disk.subrnodes(), dtype=float)
        z = np.array(list(heights.values()))
        R = r * kpc_per_arcsec
        force_R, _ = mass.field.forces(R[:, None], z * kpc_per_arcsec)
        v2 = -R[:, None] * force_R
        support = next(iter(disk.pressure_supported_params().values()), None)
        if support is not None:
            density = None if support.density is None else _pressure_density(
                mass.model, support, R, mass.params, mass.context,
                z * kpc_per_arcsec)
            v2 = v2 + pressure_support(
                disk, self._plan, values, r, support.mode, density,
                equilibrium, z, kpc_per_arcsec)
        v = _rotation(v2)
        factors = np.divide(
            v, v[:, :1], out=np.ones_like(v), where=v[:, :1] != 0)
        return dict(zip(heights, factors.T))

    def _equilibrium(
            self,
            values: ParamValues,
            mass: GalaxyMass | None,
            out_extra: Extras | None
    ) -> np.ndarray:
        """
        Return the dispersion of the vertical equilibrium (km/s) of its
        disk at its subnodes (see Kinematics), from the values of its
        parameters and the mass of the galaxy (None without mass); for a
        central equilibrium, from the squared circular velocity of the mass
        at the subnodes, and without mass, from that of its rotation. With
        extra outputs, its table (see given).
        """
        disk = self._disk
        equilibrium = disk.equilibrium()
        plan = self._plan
        radii = np.asarray(disk.subrnodes())
        heights = rms_heights(disk, plan, values, radii)
        if equilibrium.mode == 'central':
            vcirc2 = rotation(disk, plan, values, radii) ** 2 if mass is None \
                else mass.field.vcirc2(radii * mass.kpc_per_arcsec)
            sigma = _central_dispersion(radii, heights, vcirc2)
            taken = dict(vcirc=_signed_sqrt(vcirc2) * u.km / u.s)
        elif equilibrium.mode == 'jeans':
            sigma = _jeans_dispersion(disk, plan, values, radii, mass)
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
            out_extra.add(
                'equilibrium', _equilibrium_table(
                    radii, heights, taken, sigma),
                "Vertical equilibrium", _EQUILIBRIUM_DESCRIPTION)
        return sigma


def _pressure_density(
        mass_model: MassModel,
        support: PressureSupport,
        R: np.ndarray,
        params: ParamValues,
        context: MassContext,
        z: np.ndarray | None = None
) -> np.ndarray:
    """
    Return the density of the mass components of a pressure support (see
    traits.VPTraitMass), their sum, at the radii R (kpc): for 'jeans', at
    their midplane, or at the heights z (kpc; an array of shape (len(R),
    len(z))); for 'self_gravitating', their surface density.
    """
    def density(name: str) -> np.ndarray:
        component = mass_model.component(name)
        own = mass_model.component_params(name, params)
        if support.mode == 'self_gravitating':
            return component.surface_density(R, own, context)
        if z is None:
            return component.density(R, np.zeros_like(R), own, context)
        return component.density(R[:, None], z, own, context)
    return sum(density(name) for name in support.density)


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
        disk: NodalDisk,
        plan: DiskPlan,
        params: ParamValues,
        radii: np.ndarray,
        mass: GalaxyMass
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
        return mass.field.vertical_force(R * kpc, z * kpc) * kpc
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


def _rotation(v2: np.ndarray) -> np.ndarray:
    """
    Return the rotation (km/s) of the given squared rotation ((km/s)^2):
    0 where it is negative (the mass pulls outwards, or the pressure
    exceeds the gravity: no rotation, as DysmalPy takes it), NaN where it
    is infinite (there is no circular orbit, e.g. at a jump of the surface
    density of a thin disk).
    """
    return np.sqrt(np.where(np.isfinite(v2), np.maximum(v2, 0), np.nan))


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
    takes, by name, and its dispersion (km/s; see DiskStage.given).
    """
    table = Table()
    table['radius'] = radii * u.arcsec
    table['height'] = heights * u.arcsec
    for name, column in taken.items():
        table[name] = column
    table['sigma'] = sigma * u.km / u.s
    return table


def _rotation_table(
        radii: np.ndarray,
        kpc_per_arcsec: float,
        parts: dict[str, np.ndarray],
        vcirc2: np.ndarray,
        pressure: np.ndarray | None,
        rotation: np.ndarray
) -> Table:
    """
    Return the table of the rotation of a disk at the given radii
    (arcsec): the Newtonian square of the circular velocity of each
    component of the mass (parts, by name) and that of all, the square of
    the circular velocity of the mass in the gravity of the galaxy
    (vcirc2), its pressure support (if any) and its rotation (see
    DiskStage.given).
    """
    kms = u.km / u.s
    table = Table()
    table['radius'] = radii * u.arcsec
    table['radius_kpc'] = radii * kpc_per_arcsec * u.kpc
    for name, part in parts.items():
        table[f'vcirc_{name}'] = _signed_sqrt(part) * kms
    table['vnewton'] = _signed_sqrt(sum(parts.values())) * kms
    table['vcirc'] = _signed_sqrt(vcirc2) * kms
    if pressure is not None:
        table['pressure'] = pressure * kms ** 2
    table['vrot'] = rotation * kms
    return table
