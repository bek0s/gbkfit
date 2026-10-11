"""
Axisymmetric disks, thin or thick (see vertical for their vertical
laws). The circular velocity of a disk in its plane follows from the
Hankel transform S(k) of its surface density (Binney & Tremaine 2008,
2.165): v^2(R) = 2 pi G R int_0^inf k J1(k R) S(k) T(k) dk, with T(k)
the transform of its vertical law (1 for a thin disk).
"""

import functools
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.interpolate
import scipy.special

from gbkfit.params.pdescs import ParamScalarDesc
from . import _numerics
from .base import MassComponent, MassContext
from .units import G
from .vertical import check_vertical_law, vertical_density, vertical_transform


__all__ = [
    'MassExponentialDisk'
]


# The table of the shape of thick exponential disks of each vertical law
# (see _exponential_shape): ln F on grids of ln x (32 points per decade)
# and ln eta, interpolated by a bicubic spline (to about 1e-6, relative)
_TABLE_X = np.geomspace(1e-4, 1e4, 257)
_TABLE_ETA = np.geomspace(1e-3, 1, 49)


def _exponential_shape_direct(
        x: np.ndarray, vertical: str, eta: float) -> np.ndarray:
    """Return F(x) of _exponential_shape, computed at each x."""
    return x * _numerics.bessel_j1_integral(
        x, lambda q: (1 + q ** 2) ** -1.5
        * vertical_transform(vertical, q, eta))


@functools.lru_cache(maxsize=None)
def _exponential_shape_table(
        vertical: str) -> scipy.interpolate.RectBivariateSpline:
    """Return the table of the shape of thick disks of a vertical law."""
    table = [_exponential_shape_direct(_TABLE_X, vertical, eta)
             for eta in _TABLE_ETA]
    return scipy.interpolate.RectBivariateSpline(
        np.log(_TABLE_ETA), np.log(_TABLE_X), np.log(table))


def _exponential_shape(
        x: np.ndarray, vertical: str, eta: float) -> np.ndarray:
    """
    Return F(x) = x int_0^inf q J1(q x) (1 + q^2)^(-3/2) T(q eta) dq, the
    shape of the circular velocity of a thick exponential disk (v^2 = G M
    / rd F(R / rd), with eta = hz / rd), at x > 0: from its table where
    the table covers x and eta, and otherwise on a log grid (see
    _numerics.on_log_grid).
    """
    out = np.empty_like(x)
    inside = (x >= _TABLE_X[0]) & (x <= _TABLE_X[-1]) \
        & (_TABLE_ETA[0] <= eta <= _TABLE_ETA[-1])
    if inside.any():
        out[inside] = np.exp(_exponential_shape_table(vertical).ev(
            np.log(eta), np.log(x[inside])))
    if not inside.all():
        out[~inside] = _numerics.on_log_grid(
            x[~inside],
            lambda xs: _exponential_shape_direct(xs, vertical, eta))
    return out


class MassExponentialDisk(MassComponent):
    """
    An exponential disk of total mass 10^log_m (Msun) and scale length rd
    (kpc): the surface density M / (2 pi rd^2) exp(-R / rd), with S(k) =
    M / (2 pi) (1 + k^2 rd^2)^(-3/2). Thin, its circular velocity has the
    closed form of Freeman (1970); thick, of the scale height hz (kpc),
    it is the integral above, computed numerically (interpolated in a
    table for hz / rd from 1e-3 to 1 and R / rd from 1e-4 to 1e4, to about
    1e-6, and computed directly otherwise).

    Parameters
    ----------
    name : str
        Its name (see MassComponent).
    vertical : str, optional
        Its vertical law (see vertical.VERTICAL_LAWS): 'thin' (the
        default), 'exponential', 'sech2' or 'gauss'.

    Raises
    ------
    ConfigError
        If the vertical law is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'exponential_disk'

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(vertical=self._vertical)

    def __init__(self, name: str, vertical: str = 'thin'):
        super().__init__(name)
        check_vertical_law(vertical)
        self._vertical = vertical

    def vertical(self) -> str:
        """Return its vertical law."""
        return self._vertical

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        pdescs = dict(
            log_m=ParamScalarDesc(desc="log10 of the mass (Msun)"),
            rd=ParamScalarDesc(desc="the scale length (kpc)", minimum=0))
        if self._vertical != 'thin':
            pdescs['hz'] = ParamScalarDesc(
                desc="the scale height (kpc)", minimum=0)
        return pdescs

    def vcirc2(
            self,
            R: npt.ArrayLike,
            params: dict[str, float],
            context: MassContext
    ) -> np.ndarray:
        mass = 10 ** params['log_m']
        rd = params['rd']
        R = np.asarray(R, dtype=float)
        if self._vertical == 'thin':
            # 2 G M / rd y^2 (I0 K0 - I1 K1), with y = R / (2 rd); the
            # scaled Bessel functions keep the products finite at large
            # y, and the limit at y = 0 is 0
            y = R / (2 * rd)
            y_safe = np.where(y > 0, y, 1)
            bessel = (
                scipy.special.i0e(y_safe) * scipy.special.k0e(y_safe)
                - scipy.special.i1e(y_safe) * scipy.special.k1e(y_safe))
            return np.where(y > 0, 2 * G * mass / rd * y_safe ** 2 * bessel, 0)
        # v^2 = G M / rd F(R / rd) (see _exponential_shape)
        x = R / rd
        out = np.zeros_like(R)
        positive = x > 0
        if positive.any():
            out[positive] = _exponential_shape(
                x[positive], self._vertical, params['hz'] / rd)
        return G * mass / rd * out

    def density(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: dict[str, float],
            context: MassContext
    ) -> np.ndarray:
        if self._vertical == 'thin':
            raise NotImplementedError(
                "a thin disk has a surface density, not a density")
        mass = 10 ** params['log_m']
        rd = params['rd']
        surface = mass / (2 * np.pi * rd ** 2) * np.exp(
            -np.asarray(R, dtype=float) / rd)
        return surface * vertical_density(self._vertical, z, params['hz'])
