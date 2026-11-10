"""
The part of a disk in the stage of its galaxy (DiskStage): the face-on
light of its image traits, its density for the mass that follows it, and
what its galaxy gives it (DiskState): the dispersion of its vertical
equilibrium, its rotation in the gravity of the mass (with its pressure
support), the factors of that rotation at heights, and the tables of its
density waves.
"""

import dataclasses
from collections.abc import Callable, Sequence

import astropy.units as u
import numpy as np
from astropy.table import Table

from gbkfit.driver import Driver
from gbkfit.params import ModelEvaluationError, ParamValues
from ...extras import Extras
from ...mass import MassContext, MassModel
from ...mass.units import G
from .._images import ImageMaps
from .._units import mass_per_unit
from .._stage import GalaxyMass
from ..base import ComponentStage, ComponentState
from ..roles import Emission
from ._disk import NodalDisk, DiskPlan
from ._profiles import (
    NodalDiskDensity, density_waves, jeans_dispersions, pressure_support,
    rms_heights, rotation, surface_density)


__all__ = [
    'DiskStage',
    'DiskState',
    'plan_disk_stage'
]


# The descriptions of the extra outputs (see DiskStage.state)
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

@dataclasses.dataclass(frozen=True)
class DiskState(ComponentState):
    """
    What the galaxy of a disk gives it in one evaluation (see
    DiskStage.state), which its plans read: the values of the parameters
    of its traits that the galaxy computes (see traits.Trait.from_galaxy),
    by name, of each kind: its rotation in the gravity of the mass at the
    radii of each parameter (km/s; see traits.RotationInput), the factors
    of that rotation at heights, at its subnodes (see
    traits.RotationFactorsInput), the tables of its density waves at its
    subnodes (see traits.WaveInput), and the face-on light of its image
    traits (see traits.ImageInput); and the dispersion of its vertical
    equilibrium at its subnodes (km/s), if it has one.
    """
    rotation: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    rotation_factors: dict[str, np.ndarray] = dataclasses.field(
        default_factory=dict)
    waves: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    image_light: dict[str, np.ndarray] = dataclasses.field(
        default_factory=dict)
    equilibrium: np.ndarray | None = None

    def inputs(self) -> ParamValues:
        """
        Return the values of the parameters of the traits that the galaxy
        computes, of all kinds (see the class), by name.
        """
        return self.rotation | self.rotation_factors | self.waves \
            | self.image_light


def plan_disk_stage(
        disk: NodalDisk,
        emission: Emission | None,
        driver: Driver,
        sky_origin: tuple[float, float] | None,
        followed: bool,
        label: str,
        name: str | None
) -> 'DiskStage | None':
    """
    Plan the part of a disk of the given emission in the stage of its
    galaxy (see Component.plan_stage), if it takes anything from the
    galaxy (the circular velocity of the mass, an equilibrium, density
    waves, the light of image traits) or the mass follows it; None
    otherwise. name is that of its component (None if unnamed), whose
    fluid it takes (see MassModel.fluid).
    """
    takes = disk.circular_velocity_params() or disk.equilibrium() is not None \
        or disk.wave_params() or disk.image_params()
    if not (takes or followed):
        return None
    return DiskStage(
        disk, label, name, driver, sky_origin, followed,
        mass_per_unit(emission) if followed else None)


class DiskStage(ComponentStage):
    """
    The part of a disk in the stage of its galaxy (see GalaxyStage): the
    face-on light of its image traits (see ImageMaps), its density, which
    the mass of the galaxy can follow (see NodalDiskDensity), and what the
    galaxy gives it (see state): the circular velocity of the mass at the
    radii of the parameters of its velocity trait 'mass' (see
    NodalDisk.circular_velocity_params), with its pressure support, if it
    has one (see NodalDisk.pressure_supported_params), and at each height,
    where its traits ask for it (the velocity height trait 'mass', see
    NodalDisk.rotation_factor_params); the dispersion of its vertical
    equilibrium at its subnodes, if it has one (see
    traits.DPTraitEquilibrium): in the rotation of its traits
    ('rotation'), in the vertical force of the mass ('galaxy'), or in the
    surface density of its fluid ('sheet'); and the
    tables of its density waves (the traits 'spiral', see
    NodalDisk.wave_params), in its rotation, at its subnodes: its state
    (see DiskState).

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
    name : str, optional
        The name of its component, whose fluid it takes (see
        MassModel.fluid).
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
    ValueError
        If it has an image trait, but there is no sky origin.
    """

    def __init__(
            self,
            disk: NodalDisk,
            label: str,
            name: str | None,
            driver: Driver,
            sky_origin: tuple[float, float] | None,
            followed: bool,
            mass_per_unit: float | None
    ):
        self._disk = disk
        self._label = label
        self._name = name
        self._followed = followed
        self._mass_per_unit = mass_per_unit
        self._images = None if not disk.image_params() \
            else ImageMaps(disk, label, sky_origin)
        sampled = followed or disk.equilibrium() is not None \
            or disk.wave_params() or disk.pressure_supported_params()
        self._plan = disk.plan(driver, 0, np.float64) if sampled else None

    def light_params(self) -> tuple[str, ...]:
        """
        Return the names of all the parameters of its disk: its plan, which
        samples its density and its other profiles, writes them all (its
        state then reads none besides those of its light).
        """
        return tuple(self._disk.pdescs())

    def values(self, own: ParamValues) -> ParamValues:
        """
        Return the values of the parameters of its disk (by its own names),
        with the face-on light of its image traits, if any.
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

    def state(
            self,
            values: ParamValues,
            groundwork: object,
            mass: GalaxyMass | None,
            out_extra: Extras | None
    ) -> DiskState:
        """
        Return what its galaxy gives it (see the class and DiskState): the
        circular velocity at the radii of its parameters (km/s), 0 where
        its square is negative and NaN where it is infinite (see
        _rotation); the factors of its rotation at the heights of its
        velocity height trait 'mass', at its subnodes (see
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
        ('vcirc', km/s; 'rotation') or the surface density
        ('surface_density', Msun/kpc^2; 'sheet'; none for 'galaxy'), and
        the dispersion of its equilibrium ('sigma', km/s);
        and a warning if it has image traits and is near edge-on (see
        ImageMaps.warn_inclination).
        """
        disk = self._disk
        if out_extra is not None and self._images is not None:
            self._images.warn_inclination(values)
        equilibrium = None
        if disk.equilibrium() is not None:
            equilibrium = self._equilibrium(values, mass, out_extra)
        rotation, factors = {}, {}
        if mass is not None and disk.circular_velocity_params():
            rotation = {
                name: self._mass_rotation(
                    name, radii, values, mass, equilibrium, out_extra)
                for name, radii in disk.circular_velocity_params().items()}
            if heights := disk.rotation_factor_params():
                factors = self._rotation_factors(
                    heights, values, mass, equilibrium)
        state = DiskState(
            rotation=rotation, rotation_factors=factors,
            image_light={name: values[name] for name in disk.image_params()},
            equilibrium=equilibrium)
        if disk.wave_params():
            state = dataclasses.replace(
                state, waves=density_waves(disk, self._plan, values, state))
        return state

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

    def _mass_rotation(
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
        state).
        """
        disk = self._disk
        kpc_per_arcsec = mass.kpc_per_arcsec
        R = np.asarray(radii, dtype=float) * kpc_per_arcsec
        vcirc2 = mass.field.vcirc2(R)
        v2 = vcirc2
        pressure = None
        if support := disk.pressure_supported_params().get(name):
            density = None if not disk.fluid() else _pressure_density(
                mass.model, disk.fluid(), support.type(), R, mass.params,
                mass.context)
            pressure = pressure_support(
                disk, self._plan, values, radii, support.type(), density,
                equilibrium)
            v2 = v2 + pressure
        result = _rotation(
            v2, radii, self._lit(values, radii), f"the rotation of '{name}'")
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
            density = None if not disk.fluid() else _pressure_density(
                mass.model, disk.fluid(), support.type(), R, mass.params,
                mass.context, z * kpc_per_arcsec)
            v2 = v2 + pressure_support(
                disk, self._plan, values, r, support.type(), density,
                equilibrium, z, kpc_per_arcsec)
        v = _rotation(
            v2, r, self._lit(values, r),
            "the rotation at the heights of 'mass'")
        factors = np.divide(
            v, v[:, :1], out=np.ones_like(v), where=v[:, :1] != 0)
        return dict(zip(heights, factors.T))

    def _lit(
            self, values: ParamValues, radii: Sequence[float] | np.ndarray
    ) -> Callable[[], np.ndarray]:
        """
        Return a function that returns where its disk has light at the
        radii (arcsec), from the values of its parameters (see
        _check_solved): everywhere if it has no plan (one that samples its
        profiles, made for what takes them).
        """
        def lit() -> np.ndarray:
            if self._plan is None:
                return np.ones(len(radii), dtype=bool)
            return surface_density(self._disk, self._plan, values, radii) > 0
        return lit

    def _equilibrium(
            self,
            values: ParamValues,
            mass: GalaxyMass | None,
            out_extra: Extras | None
    ) -> np.ndarray:
        """
        Return the dispersion of the vertical equilibrium (km/s) of its
        disk at its subnodes (see traits.DPTraitEquilibrium), from the
        values of its parameters and the mass of the galaxy (None without
        mass), by its gravity: 'rotation', the squared rotation of its
        tangential traits at the subnodes; 'galaxy', the vertical force
        of the mass; 'sheet', the surface density of its fluid (see
        MassModel.fluid). With extra outputs, its table (see state).
        """
        disk = self._disk
        gravity = disk.equilibrium().gravity()
        plan = self._plan
        radii = np.asarray(disk.subrnodes())
        heights = rms_heights(disk, plan, values, radii)
        lit = self._lit(values, radii)
        if gravity == 'rotation':
            vcirc2 = rotation(disk, plan, values, radii) ** 2
            sigma = _rotation_dispersion(radii, lit, heights, vcirc2)
            taken = dict(vcirc=_signed_sqrt(vcirc2) * u.km / u.s)
        elif gravity == 'galaxy':
            sigma = _galaxy_dispersion(disk, plan, values, radii, mass)
            taken = {}
        else:
            R = radii * mass.kpc_per_arcsec
            surface = sum(
                mass.model.component(name).surface_density(
                    R, mass.model.component_params(name, mass.params),
                    mass.context)
                for name in mass.model.fluid(self._name, disk.fluid()))
            sigma = _sheet_dispersion(
                radii, lit, heights * mass.kpc_per_arcsec, surface)
            taken = dict(surface_density=surface * u.solMass / u.kpc ** 2)
        if out_extra is not None:
            out_extra.add(
                'equilibrium', _equilibrium_table(
                    radii, heights, taken, sigma),
                "Vertical equilibrium", _EQUILIBRIUM_DESCRIPTION)
        return sigma


def _pressure_density(
        mass_model: MassModel,
        fluid: tuple[str, ...],
        mode: str,
        R: np.ndarray,
        params: ParamValues,
        context: MassContext,
        z: np.ndarray | None = None
) -> np.ndarray:
    """
    Return the density of the fluid of a pressure support (the mass disks
    of the given names; see traits.PressureSupport), their sum, at the
    radii R (kpc), by its mode: for 'jeans', at their midplane, or at the
    heights z (kpc; an array of shape (len(R), len(z))); for
    'self_gravitating', their surface density.
    """
    def density(name: str) -> np.ndarray:
        component = mass_model.component(name)
        own = mass_model.component_params(name, params)
        if mode == 'self_gravitating':
            return component.surface_density(R, own, context)
        if z is None:
            return component.density(R, np.zeros_like(R), own, context)
        return component.density(R[:, None], z, own, context)
    return sum(density(name) for name in fluid)


def _rotation_dispersion(
        radii: np.ndarray,
        lit: Callable[[], np.ndarray],
        heights: np.ndarray,
        vcirc2: np.ndarray
) -> np.ndarray:
    """
    Return the dispersion (km/s) of the equilibrium 'rotation' (see
    traits.DPTraitEquilibrium) at the radii (arcsec), where the disk has
    light or not (see _check_solved), of the given rms heights (arcsec)
    and squared circular velocities ((km/s)^2): h Omega, Omega^2 = vcirc2
    / R^2; at the centre, where it is 0 / 0, Omega at the next radius; 0
    where it has no solution and no light.

    Raises
    ------
    ModelEvaluationError
        If vcirc2 is negative or not finite at a radius with light.
    """
    omega2 = np.divide(
        vcirc2, radii ** 2, out=np.zeros_like(vcirc2), where=radii > 0)
    if radii[0] == 0:
        omega2[0] = omega2[1]
    solved = np.isfinite(omega2) & (omega2 >= 0)
    _check_solved(
        solved, lit, radii, "the equilibrium 'rotation'",
        "the squared rotation is negative or not finite")
    return heights * np.sqrt(np.where(solved, omega2, 0))


def _galaxy_dispersion(
        disk: NodalDisk,
        plan: DiskPlan,
        params: ParamValues,
        radii: np.ndarray,
        mass: GalaxyMass
) -> np.ndarray:
    """
    Return the dispersion (km/s) of the equilibrium 'galaxy' (see
    traits.DPTraitEquilibrium) of a disk at its subnodes (radii, arcsec), in
    the vertical force of the mass of its galaxy (see jeans_dispersions);
    at the centre, the value at the next subnode.
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


def _sheet_dispersion(
        radii: np.ndarray,
        lit: Callable[[], np.ndarray],
        heights: np.ndarray,
        surface: np.ndarray
) -> np.ndarray:
    """
    Return the dispersion (km/s) of the equilibrium 'sheet' (see
    traits.DPTraitEquilibrium) at the radii (arcsec), where the
    disk has light or not (see _check_solved), of the given rms heights
    (kpc) and surface densities (Msun/kpc^2): sqrt(2 sqrt(3) G Sigma h);
    0 where it has no solution and no light.

    Raises
    ------
    ModelEvaluationError
        If Sigma is negative at a radius with light.
    """
    sigma2 = 2 * np.sqrt(3) * G * surface * heights
    solved = sigma2 >= 0
    _check_solved(
        solved, lit, radii, "the equilibrium 'sheet'",
        "the surface density of the mass is negative")
    return np.sqrt(np.where(solved, sigma2, 0))


def _rotation(
        v2: np.ndarray,
        radii: Sequence[float] | np.ndarray,
        lit: Callable[[], np.ndarray],
        what: str
) -> np.ndarray:
    """
    Return the rotation (km/s) of the given squared rotation ((km/s)^2)
    at the radii (arcsec, along its first axis), where the disk has light
    or not (see _check_solved), what (for the error): 0 where it is
    negative (the mass pulls outwards, or the pressure exceeds the
    gravity: no rotation, as DysmalPy takes it), and where it is not
    finite and the disk has no light (e.g. beyond the end of its density,
    by which the pressure support divides).

    Raises
    ------
    ModelEvaluationError
        If it is not finite at a radius with light (there is no circular
        orbit, e.g. at a jump of the surface density of a thin disk).
    """
    finite = np.isfinite(v2)
    _check_solved(
        finite, lit, radii, what,
        "it is not finite (no circular orbit, e.g. at a jump of the "
        "surface density of a thin disk)")
    return np.sqrt(np.where(finite, np.maximum(v2, 0), 0))


def _check_solved(
        solved: np.ndarray,
        lit: Callable[[], np.ndarray],
        radii: Sequence[float] | np.ndarray,
        what: str,
        why: str
) -> None:
    """
    Check that what (e.g. "the central equilibrium") has a solution at
    each of the radii (arcsec, along the first axis of solved) where the
    disk has light (the mask that lit returns, one value per radius,
    asked only if it has none somewhere): elsewhere, no model sees it.

    Raises
    ------
    ModelEvaluationError
        If it has none at a radius with light, with the first and why.
    """
    # (an error, not NaN tables: a comparison would score a model of NaN
    # as one without values there, a flat chi-square that can beat a real
    # fit; a fitter takes the error as a point it cannot evaluate: audit
    # 2026-10-31 #8)
    unsolved = ~solved.reshape(len(solved), -1).all(axis=1)
    if unsolved.any():
        unsolved &= lit()
    if unsolved.any():
        radius = float(np.asarray(radii)[unsolved][0])
        raise ModelEvaluationError(
            f"{what} has no solution at R = {radius:g} arcsec: {why}")


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
    takes, by name, and its dispersion (km/s; see DiskStage.state).
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
    DiskStage.state).
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
