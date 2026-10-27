"""
The Jeans kinematics of a spheroid (JeansKinematics): its stars in a
steady state in the gravity of its galaxy, their mean velocity and
dispersions from the Jeans equations of a velocity ellipsoid aligned with
the cylinder (JAM_cyl, Cappellari 2008), tabled on (R, |z|) in each
evaluation (JeansTables, see jeans_tables) from what the light and the
gravity alone set of them (JeansForces, see jeans_forces), the state of
their motion (JeansMotion).
"""

import dataclasses
import math
from typing import Any

import numpy as np
import scipy.interpolate

from gbkfit.driver import Driver
from gbkfit.params import (
    ModelEvaluationError, ParamDesc, ParamScalarDesc, ParamValues)
from ...gravity import GravityField
from .._native import ParamBuffer
from .._stage import GalaxyMass
from ..base import ComponentState
from ._kinematics import SpheroidKinematics, SpheroidLight, SpheroidMotion


__all__ = [
    'JeansForces',
    'JeansKinematics',
    'JeansMotion',
    'JeansTables',
    'jeans_forces',
    'jeans_tables'
]


# The tables (see jeans_tables): the number of radii and of heights,
# geometric from rmax / _TABLE_SPAN to rmax, the heights beyond rmax up to
# _TAIL_SPAN rmax that the integrals along z take, and the steps of those
# integrals between two heights (to about 1e-3 of sigma within 3
# dispersions of a Gaussian tracer, whose tail falls the fastest)
_TABLE_SIZE = 64
_TABLE_SPAN = 1e4
_TAIL_SIZE = 32
_TAIL_SPAN = 100.0
_SUBSTEPS = 4


class JeansKinematics(SpheroidKinematics):
    """
    The kinematics of the stars of a spheroid in a steady state in the
    gravity of its galaxy, of a velocity ellipsoid aligned with the
    cylinder (JAM_cyl, Cappellari 2008): at each (R, z), from the forces
    F_R and F_z of all the mass of the galaxy and the density nu of the
    light of the spheroid (its whole profile),

        nu sigma_z^2 = int_z^inf nu (-F_z) dz'
        nu <v_phi^2> = b [R d(nu sigma_z^2)/dR + nu sigma_z^2] + R nu (-F_R)
        sigma_R^2 = b sigma_z^2,  b = 1 / (1 - beta_z)
        <v_phi> = kappa (<v_phi^2> - sigma_R^2)^(1/2)
        sigma_phi^2 = <v_phi^2> - <v_phi>^2

    of the anisotropy beta_z = 1 - sigma_z^2 / sigma_R^2 and the rotation
    kappa (0: none; 1: that of an isotropic rotator where beta_z = 0), two
    parameters. Where <v_phi^2> is below sigma_R^2 the stars cannot rotate
    so: <v_phi> is 0 there (as JamPy), and sigma_phi^2 = <v_phi^2>. NaN
    where <v_phi^2> is negative, or where kappa makes <v_phi>^2 exceed
    <v_phi^2>; 0 where it has no light (beyond the floats). Each point of the spheroid moves with a Gaussian
    distribution of velocities of that mean and those dispersions (the
    velocity ellipsoid seen along the line of sight): the Jeans equations
    give the mean and the dispersion of the stars, not the shape of their
    distribution, so fit those (see SpheroidKinematics).
    """

    @staticmethod
    def type() -> str:
        return 'jeans'

    def pdescs(self) -> dict[str, ParamDesc]:
        return dict(
            beta_z=ParamScalarDesc(
                desc="the anisotropy, 1 - sigma_z^2 / sigma_R^2",
                maximum=1, unit=''),
            kappa=ParamScalarDesc(
                desc="the rotation, the share of <v_phi^2> - sigma_R^2 "
                     "whose square root is the mean rotation",
                unit=''))

    def uses_mass(self) -> bool:
        return True

    def motion(self, rmax: float) -> 'JeansMotion':
        return JeansMotion(self, rmax)


@dataclasses.dataclass(frozen=True)
class JeansTables(ComponentState):
    """
    The mean rotation and the dispersions (km/s) of the stars of a spheroid
    of Jeans kinematics in one evaluation (see jeans_tables): on the
    radii R and heights |z| (arcsec) exp(lnr_zero + i lnr_step) and
    exp(lnz_zero + j lnz_step), arrays of shape (nR, nz): <v_phi>,
    sigma_R, sigma_phi and sigma_z.
    """
    lnr_zero: float
    lnr_step: float
    lnz_zero: float
    lnz_step: float
    vphi: np.ndarray
    sigma_r: np.ndarray
    sigma_phi: np.ndarray
    sigma_z: np.ndarray


def table_shape() -> tuple[int, int]:
    """Return the shape of the tables (nR, nz)."""
    return _TABLE_SIZE, _TABLE_SIZE


def light(
        terms: np.ndarray,
        function: tuple[float, float],
        table: tuple[np.ndarray, np.ndarray] | None,
        R: np.ndarray,
        z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the density of the light of a spheroid and its derivatives
    along R and z at the points (R, z) (arcsec), of the given terms, their
    function and its table (see LightProfile): its whole profile, the sum
    over its terms of a f(m / s), m^2 = R^2 + z^2 / q^2, f(u) = exp(T(ln
    u) - c u^p), f'(u) = f(u) (T'(ln u) / u - c p u^(p - 1)), T linear
    between the points of its table, along its slope below them, and f 0
    beyond them (as the kernels take it).
    """
    c, p = function
    density = np.zeros(np.broadcast(R, z).shape)
    d_R = np.zeros_like(density)
    d_z = np.zeros_like(density)
    for amplitude, scale, q in terms:
        m = np.sqrt(R ** 2 + z ** 2 / q ** 2)
        u = m / scale
        with np.errstate(divide='ignore', invalid='ignore'):
            lnu = np.log(u)
            t, slope = _table(lnu, table) if table is not None \
                else (np.zeros_like(u), np.zeros_like(u))
            f = np.exp(t - c * u ** p)
            df = f * (slope / u - c * p * u ** (p - 1))
            dm_R = np.where(m > 0, R / m, 0.0)
            dm_z = np.where(m > 0, z / (q ** 2 * m), 0.0)
        density += amplitude * f
        d_R += amplitude * df / scale * dm_R
        d_z += amplitude * df / scale * dm_z
    return density, d_R, d_z


def _table(
        lnu: np.ndarray, table: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the table T of a function (see light) at ln u, and its slope
    dT / d ln u: -infinity beyond its last point.
    """
    values, (zero, step, slope) = table
    p = (lnu - zero) / step
    last = len(values) - 1
    i = np.clip(np.floor(p).astype(int), 0, last - 1)
    slopes = np.diff(values) / step
    inner = p < 0
    t = np.where(
        inner, values[0] + slope * p * step,
        values[i] + (p - i) * step * slopes[i])
    t = np.where(p > last, -np.inf, t)
    return t, np.where(inner, slope, slopes[i])


def _interpolated(
        lnz_nodes: np.ndarray, values: np.ndarray, lnz: np.ndarray
) -> np.ndarray:
    """
    Return the values of a force on rows at the heights lnz (ln z) from
    those at lnz_nodes, by a cubic spline in ln z: of the logarithm of its
    magnitude where its sign holds along a row (a power of z between the
    nodes is then exact), and of the force itself where it does not.
    """
    sign = np.sign(values[:, :1])
    held = np.all(np.sign(values) == sign, axis=1, keepdims=True) \
        & (sign != 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        knots = np.where(held, np.log(sign * values), values)
    spline = scipy.interpolate.CubicSpline(lnz_nodes, knots, axis=1)(lnz)
    return np.where(held, sign * np.exp(spline), spline)


@dataclasses.dataclass(frozen=True)
class JeansForces:
    """
    What the light of a spheroid and the gravity of its galaxy alone set
    of its Jeans tables in one evaluation (see jeans_forces), the
    groundwork of the tables, which its beta_z and kappa then make (see
    jeans_tables): on the radii R and heights |z| of the tables (see
    JeansTables), arrays of shape (nR, nz): the density nu of its light,
    nu sigma_z^2, R d(nu sigma_z^2)/dR and R nu (-F_R) (nu times
    (km/s)^2).
    """
    lnr_zero: float
    lnr_step: float
    nu: np.ndarray
    nu_sz2: np.ndarray
    r_dnu_sz2: np.ndarray
    r_nu_force: np.ndarray


def jeans_forces(
        light_terms: SpheroidLight,
        rmax: float,
        field: GravityField,
        kpc_per_arcsec: float
) -> JeansForces:
    """
    Return the groundwork of the Jeans tables of a spheroid (see
    JeansForces and JeansKinematics): of the light of its profile (the
    arguments of light but R and z: terms, function, table), its rmax
    (arcsec), which bounds the tables, in the gravity of its galaxy
    (field), in a galaxy of the given length of an arcsec (kpc).

    The integrals along z are from the heights of the tables up to
    _TAIL_SPAN rmax, in _SUBSTEPS steps between two heights, each exact
    for an integrand that is a power of z between its ends (or by the
    trapezoid rule where it changes sign), with the forces at the
    heights, interpolated between them (see _interpolated); R d(nu
    sigma_z^2)/dR
    is not differentiated numerically: the forces have no curl (dF_z/dR
    = dF_R/dz), so by parts it is R [int_z^inf (dnu/dR) (-F_z) dz' + nu
    F_R + int_z^inf (dnu/dz') F_R dz'].
    """
    lnr = np.linspace(np.log(rmax / _TABLE_SPAN), np.log(rmax), _TABLE_SIZE)
    # (the heights of the tables, every _SUBSTEPS of those of the
    # integrals, which go on beyond rmax at the same step)
    step = (lnr[1] - lnr[0]) / _SUBSTEPS
    nz = (_TABLE_SIZE + _TAIL_SIZE - 1) * _SUBSTEPS + 1
    lnz = lnr[0] + step * np.arange(nz)
    heights = slice(0, _TABLE_SIZE * _SUBSTEPS, _SUBSTEPS)
    R, z = np.meshgrid(np.exp(lnr), np.exp(lnz), indexing='ij')
    nu, nu_R, nu_z = light(*light_terms, R, z)
    # (in kpc: the forces, at every _SUBSTEPS height and between them by
    # interpolation, smooth where nu can fall steeply; and the derivatives
    # of nu per kpc)
    k = kpc_per_arcsec
    nodes = slice(None, None, _SUBSTEPS)
    force_R, force_z = (
        _interpolated(lnz[nodes], force, lnz)
        for force in field.forces(R[:, nodes] * k, z[:, nodes] * k))
    nu_R, nu_z = nu_R / k, nu_z / k

    def from_above(integrand: np.ndarray) -> np.ndarray:
        # int_z^inf of integrand dz (kpc), as int of integrand z d(ln z):
        # each step exactly for a power of z between its ends, or by the
        # trapezoid rule where it changes sign or is flat
        g = integrand * z * k
        g1, g2 = g[:, :-1], g[:, 1:]
        ratio = np.divide(g1, g2, out=np.zeros_like(g1), where=g2 != 0)
        power = (ratio > 0) & (ratio != 1)
        steps = np.where(
            power, (g1 - g2) * step / np.log(np.where(power, ratio, 2.0)),
            (g1 + g2) / 2 * step)
        return np.concatenate(
            (np.cumsum(steps[:, ::-1], axis=1)[:, ::-1],
             np.zeros((len(lnr), 1))), axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        nu_sz2 = from_above(nu * -force_z)
        r_dnu_sz2 = R * k * (
            from_above(nu_R * -force_z) + nu * force_R
            + from_above(nu_z * force_R))
        r_nu_force = R * k * nu * -force_R
    table = slice(None), heights
    return JeansForces(
        float(lnr[0]), float(lnr[1] - lnr[0]), nu[table], nu_sz2[table],
        r_dnu_sz2[table], r_nu_force[table])


def jeans_tables(
        kinematics_params: ParamValues, forces: JeansForces
) -> JeansTables:
    """
    Return the Jeans tables of a spheroid (see JeansKinematics): of the
    values of its parameters beta_z and kappa, and their groundwork (see
    jeans_forces).
    """
    beta_z = float(kinematics_params['beta_z'])
    kappa = float(kinematics_params['kappa'])
    if not beta_z < 1:
        raise ModelEvaluationError(
            f"the Jeans equations have no solution at beta_z = {beta_z}: "
            f"sigma_R^2 = sigma_z^2 / (1 - beta_z)")
    b = 1 / (1 - beta_z)
    nu = forces.nu
    with np.errstate(divide='ignore', invalid='ignore'):
        nu_vphi2 = b * (forces.r_dnu_sz2 + forces.nu_sz2) + forces.r_nu_force
        sz2 = forces.nu_sz2 / nu
        vphi2 = nu_vphi2 / nu
        sr2 = b * sz2
        rotation = vphi2 - sr2
        # Where <v_phi^2> < sigma_R^2, no real kappa (<v_phi^2> -
        # sigma_R^2)^(1/2) exists: the stars cannot rotate with sigma_phi =
        # sigma_R. As JamPy (Cappellari 2008, 2020), their mean rotation is
        # then 0, and sigma_phi^2 = <v_phi^2>: the second moments of the
        # Jeans equations hold everywhere, and only the split of <v_phi^2>
        # into rotation and dispersion (the assumption that kappa scales)
        # gives way. A choice: NaN there would leave ordinary fast rotators
        # (beta_z above about 0.6 (1 - q) at their centres) without a model.
        vphi = kappa * np.sqrt(np.maximum(rotation, 0))
        sphi2 = vphi2 - vphi ** 2
        # (0 where there is no light, of no weight: the kernels interpolate
        # the tables, so they must be numbers there)
        sphi = np.sqrt(np.where(sphi2 >= 0, sphi2, np.nan))
        values = [
            np.where(nu > 0, value, 0.0)
            for value in (vphi, np.sqrt(sr2), sphi, np.sqrt(sz2))]
        return JeansTables(
            forces.lnr_zero, forces.lnr_step, forces.lnr_zero,
            forces.lnr_step, *values)


class JeansMotion(SpheroidMotion):
    """
    The motion of the stars of a spheroid of JeansKinematics, of the given
    rmax: its tables in each evaluation (its state, see jeans_tables), in
    the groups 'jeans' (the four planes of the tables) and 'jeans_range'
    of the buffer.
    """

    def __init__(self, kinematics: JeansKinematics, rmax: float):
        self._kinematics = kinematics
        self._rmax = rmax

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._kinematics.pdescs()

    def takes_light(self) -> bool:
        return True

    def state_params(self) -> tuple[str, ...]:
        return tuple(self._kinematics.pdescs())

    def groundwork(
            self, light: SpheroidLight, mass: GalaxyMass | None
    ) -> JeansForces:
        if mass is None:
            raise ValueError(
                "the Jeans kinematics of a spheroid need the mass of its "
                "galaxy")
        return jeans_forces(
            light, self._rmax, mass.field, mass.kpc_per_arcsec)

    def state(
            self, values: ParamValues, groundwork: object
    ) -> JeansTables:
        if not isinstance(groundwork, JeansForces):
            raise ValueError(
                "the Jeans kinematics of a spheroid need the groundwork of "
                "its tables (see JeansForces)")
        return jeans_tables(values, groundwork)

    def sizes(self) -> dict[str, int]:
        return dict(jeans=4 * math.prod(table_shape()), jeans_range=4)

    def native(
            self, driver: Driver, dtype: np.dtype, buffer: ParamBuffer
    ) -> dict[str, Any]:
        return dict(
            jeans=buffer.device('jeans'),
            jeans_range=buffer.device('jeans_range'),
            jeans_shape=table_shape())

    def write(
            self,
            buffer: ParamBuffer,
            params: ParamValues,
            state: ComponentState | None
    ) -> None:
        if not isinstance(state, JeansTables):
            raise ValueError(
                "the Jeans kinematics of a spheroid need its tables (its "
                "state, see JeansTables)")
        buffer.host('jeans')[:] = np.stack((
            state.vphi, state.sigma_r, state.sigma_phi,
            state.sigma_z)).ravel()
        buffer.host('jeans_range')[:] = (
            state.lnr_zero, state.lnr_step, state.lnz_zero, state.lnz_step)
