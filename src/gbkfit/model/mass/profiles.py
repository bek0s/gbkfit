"""
The profiles of spheroids (see spheroids.MassSpheroid): spherical
densities, given as functions of the spherical radius r with their
enclosed masses; the halos are in halos.
"""

import abc
from typing import Any

import numpy as np
import numpy.typing as npt

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import parseutils
from . import _numerics
from .base import MassContext
from .units import G


__all__ = [
    'ProfileSersic',
    'SpheroidProfile'
]


class SpheroidProfile(parseutils.TypedSerializable, abc.ABC):
    """
    The profile of a spheroid: its density and its enclosed mass, as
    functions of the spherical radius r (kpc), and its potential
    (numerically, unless a profile gives it exactly).
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    @abc.abstractmethod
    def pdescs(self) -> dict[str, ParamDesc]:
        """Return its parameters, by name."""
        pass

    @abc.abstractmethod
    def density(
            self,
            r: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its density (Msun/kpc^3) at the spherical radii r (kpc; see
        MassComponent.vcirc2 for the other arguments).
        """
        pass

    @abc.abstractmethod
    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its mass (Msun) within the spherical radii r (kpc; see
        MassComponent.vcirc2 for the other arguments).
        """
        pass

    def potential(
            self,
            r: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return its potential ((km/s)^2), 0 at infinity, at the spherical
        radii r (kpc; see MassComponent.vcirc2 for the other arguments).

        Raises
        ------
        ValueError
            If its potential is infinite.
        """
        # -G (M(<r) / r + 4 pi int_r^inf rho r' dr'), by quadrature
        return _numerics.spherical_potential(
            lambda x: self.density(x, params, context),
            lambda x: self.enclosed_mass(x, params, context), r)


class ProfileSersic(SpheroidProfile):
    """
    The spherical deprojection of a Sersic profile of total mass 10^log_m
    (Msun), effective radius re (kpc; within which half of the projected
    mass is) and index n: the surface density S_e exp(-b ((R / re)^(1/n)
    - 1)), with b(n) exact. The density, enclosed mass and potential are
    computed numerically (see _numerics.sersic_tables), for n in [0.3,
    12].
    """

    @staticmethod
    def type() -> str:
        return 'sersic'

    def pdescs(self) -> dict[str, ParamDesc]:
        return dict(
            log_m=ParamScalarDesc(desc="log10 of the mass (Msun)"),
            re=ParamScalarDesc(desc="the effective radius (kpc)", minimum=0),
            n=ParamScalarDesc(
                desc="the Sersic index", minimum=0.3, maximum=12))

    def density(
            self,
            r: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        mass, re, tables = self._scales(params)
        return mass / re ** 3 * tables.density(np.asarray(r) / re)

    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        mass, re, tables = self._scales(params)
        return mass * tables.mass(np.asarray(r, dtype=float) / re)

    def potential(
            self,
            r: np.ndarray,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        mass, re, tables = self._scales(params)
        return G * mass / re * tables.potential(r / re)

    @staticmethod
    def _scales(
            params: ParamValues
    ) -> tuple[float, float, _numerics.SphericalTables]:
        """Return its mass, its effective radius and its unit tables."""
        return (10 ** params['log_m'], params['re'],
                _numerics.sersic_tables(float(params['n'])))
