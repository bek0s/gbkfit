"""
The parameterisations of halos, and their overdensities.

A halo of density rho_s f(r / r_s) (f: the shape of its profile) has
the mass M(<r) = 4 pi rho_s r_s^3 m(r / r_s), with m(x) = int_0^x f(t)
t^2 dt. Its mass M_D and radius r_D are those of the sphere of mean
density D rho_ref, with D the overdensity and rho_ref the critical or
the mean matter density at the epoch of the galaxy; its velocity is
V_D^2 = G M_D / r_D, and its concentration c = r_D / r_s, where r_s is
the scale radius of its profile (each profile states which).

A halo is computed from (rho_s, r_s). It is given by those (log10
rho_s, r_s: 'density_radius'), by (log10 M_D, c: 'mass_concentration')
or by (V_D, c: 'velocity_concentration'), which convert to them
exactly.
"""

import re
from collections.abc import Callable

import numpy as np

from gbkfit.params import ParamValues
from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils.parseutils import ConfigError
from .base import MassContext
from .units import G


__all__ = [
    'Overdensity',
    'PARAMETERISATIONS',
    'check_parameterisation',
    'parameterisation_pdescs',
    'halo_scale_radius',
    'halo_scales'
]


PARAMETERISATIONS = (
    'mass_concentration', 'velocity_concentration', 'density_radius')


def check_parameterisation(parameterisation: str) -> None:
    """Raise ConfigError unless it is one of PARAMETERISATIONS."""
    if parameterisation not in PARAMETERISATIONS:
        raise ConfigError(
            f"parameters must be one of {list(PARAMETERISATIONS)}; it is "
            f"{parameterisation!r}")


class Overdensity:
    """
    An overdensity: '<D>c' (D times the critical density, e.g. '200c'),
    '<D>m' (D times the mean matter density, e.g. '200m') or 'vir' (the
    virial overdensity of Bryan & Norman 1998, for a flat universe:
    (18 pi^2 + 82 x - 39 x^2) times the critical density, with x =
    Omega_m(z) - 1).

    Parameters
    ----------
    spec : str
        The overdensity.

    Raises
    ------
    ConfigError
        If the overdensity is not valid.
    """

    def __init__(self, spec: str):
        match = re.fullmatch(r'(\d+(?:\.\d+)?)([cm])', spec) \
            if isinstance(spec, str) else None
        if spec != 'vir' and (match is None or float(match[1]) <= 0):
            raise ConfigError(
                f"overdensity must be '<number>c', '<number>m' or 'vir' "
                f"(e.g. '200c'); it is {spec!r}")
        self._spec = spec
        self._factor = float(match[1]) if match else None
        self._reference = match[2] if match else None

    def __str__(self) -> str:
        return self._spec

    def density(self, context: MassContext) -> float:
        """Return D rho_ref (Msun/kpc^3) at the epoch of the context."""
        if self._spec == 'vir':
            x = context.omega_m - 1
            return (18 * np.pi ** 2 + 82 * x - 39 * x ** 2) \
                * context.critical_density
        reference = context.critical_density if self._reference == 'c' \
            else context.omega_m * context.critical_density
        return self._factor * reference


def parameterisation_pdescs(
        parameterisation: str) -> dict[str, ParamScalarDesc]:
    """Return the parameters of a parameterisation, by name."""
    c = ParamScalarDesc(
        desc="the concentration, r_D / r_s", minimum=0, unit='')
    return dict(
        mass_concentration=dict(
            log_m=ParamScalarDesc(
                desc="log10 of the mass M_D", unit='dex(solMass)'),
            c=c),
        velocity_concentration=dict(
            v=ParamScalarDesc(
                desc="the circular velocity V_D at r_D", minimum=0,
                unit='km/s'),
            c=c),
        density_radius=dict(
            log_rho=ParamScalarDesc(
                desc="log10 of the scale density rho_s",
                unit='dex(solMass / kpc3)'),
            rs=ParamScalarDesc(
                desc="the scale radius r_s", minimum=0, unit='kpc')),
    )[parameterisation]


def halo_scales(
        parameterisation: str,
        params: ParamValues,
        overdensity_density: float | None,
        mass_shape: Callable[[float], float]
) -> tuple[float, float]:
    """
    Return (rho_s, r_s) of a halo from the values of the parameters of
    its parameterisation, D rho_ref (Msun/kpc^3; None for
    'density_radius', which needs none), and m(x), its mass shape (see
    the module docstring).
    """
    if parameterisation == 'density_radius':
        return 10 ** params['log_rho'], params['rs']
    mass, radius = _mass_and_radius(
        parameterisation, params, overdensity_density)
    c = params['c']
    r_s = radius / c
    return mass / (4 * np.pi * r_s ** 3 * mass_shape(c)), r_s


def halo_scale_radius(
        parameterisation: str,
        params: ParamValues,
        overdensity_density: float | None
) -> float:
    """
    Return r_s (kpc) of a halo from the values of the parameters of its
    parameterisation, and D rho_ref (see halo_scales): it does not depend
    on the shape of its profile.
    """
    if parameterisation == 'density_radius':
        return params['rs']
    _, radius = _mass_and_radius(
        parameterisation, params, overdensity_density)
    return radius / params['c']


def _mass_and_radius(
        parameterisation: str,
        params: ParamValues,
        overdensity_density: float
) -> tuple[float, float]:
    """
    Return M_D (Msun) and r_D (kpc) of a halo of the parameterisation of
    a mass or of a velocity (see halo_scales).
    """
    if parameterisation == 'mass_concentration':
        mass = 10 ** params['log_m']
        radius = np.cbrt(3 * mass / (4 * np.pi * overdensity_density))
    else:
        # V^2 = G M / r and M = (4 / 3) pi D rho_ref r^3
        radius = params['v'] / np.sqrt(
            4 / 3 * np.pi * G * overdensity_density)
        mass = params['v'] ** 2 * radius / G
    return mass, radius
