"""
Point masses (e.g. a central black hole).
"""

import numpy as np
import numpy.typing as npt

from gbkfit.params.pdescs import ParamScalarDesc
from .base import MassContext, MassSpherical
from .units import G


__all__ = [
    'MassPointMass'
]


class MassPointMass(MassSpherical):
    """
    A point mass of 10^log_m Msun at the centre: v^2 = G M / r and the
    potential -G M / r. Its density is 0 away from the centre.

    Parameters
    ----------
    name : str
        Its name (see MassComponent).
    """

    @staticmethod
    def type() -> str:
        return 'point_mass'

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(log_m=ParamScalarDesc(desc="log10 of the mass (Msun)"))

    def density_spherical(
            self,
            r: np.ndarray,
            params: dict[str, float],
            context: MassContext
    ) -> np.ndarray:
        return np.zeros_like(np.asarray(r, dtype=float))

    def enclosed_mass(
            self,
            r: npt.ArrayLike,
            params: dict[str, float],
            context: MassContext
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        return np.where(r > 0, 10 ** params['log_m'], 0.0)

    def potential(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: dict[str, float],
            context: MassContext
    ) -> np.ndarray:
        r = np.hypot(R, z)
        return -G * 10 ** params['log_m'] / r
