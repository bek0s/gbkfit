"""
Spherical spheroids (e.g. a bulge), given by their surface density.
"""

import numpy as np
import numpy.typing as npt

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.params.pdescs import ParamScalarDesc
from . import _numerics
from .base import MassContext, MassSpherical
from .units import G


__all__ = [
    'MassSersic'
]


class MassSersic(MassSpherical):
    """
    The spherical deprojection of a Sersic profile of total mass 10^log_m
    (Msun), effective radius re (kpc; within which half of the projected
    mass is) and index n: the surface density S_e exp(-b ((R / re)^(1/n)
    - 1)), with b(n) exact. The density, enclosed mass and potential are
    computed numerically (see _numerics.sersic_tables), for n in [0.3,
    12].

    Parameters
    ----------
    name : str
        Its name (see MassComponent).
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

    def density_spherical(
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
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        mass, re, tables = self._scales(params)
        return G * mass / re * tables.potential(np.hypot(R, z) / re)

    @staticmethod
    def _scales(
            params: ParamValues
    ) -> tuple[float, float, _numerics.SphericalTables]:
        """Return its mass, its effective radius and its unit tables."""
        return (10 ** params['log_m'], params['re'],
                _numerics.sersic_tables(float(params['n'])))
