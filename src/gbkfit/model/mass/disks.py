"""
Axisymmetric disks, thin or thick (see vertical for their vertical
laws). The circular velocity of an exponential disk in its plane follows
from the Hankel transform S(k) of its surface density (Binney & Tremaine
2008, 2.165): v^2(R) = 2 pi G R int_0^inf k J1(k R) S(k) T(k) dk, with
T(k) the transform of its vertical law (1 for a thin disk); that of a
disk of any profile, from the pull of its rings (see _disk_gravity).
"""

import functools
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.interpolate
import scipy.special

from gbkfit.params import ParamDesc, ParamValues
from gbkfit.params.pdescs import ParamScalarDesc, ParamVectorDesc
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _disk_gravity, _numerics
from .base import MassComponent, MassContext
from .units import G
from .vertical import (
    VERTICAL_LAWS, check_vertical_law, vertical_density, vertical_transform)


__all__ = [
    'MassExponentialDisk',
    'MassNodewiseDisk',
    'MassTraitDisk'
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

    def pdescs(self) -> dict[str, ParamDesc]:
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
            params: ParamValues,
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
            params: ParamValues,
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


class MassNodewiseDisk(MassComponent):
    """
    A disk whose surface density is given at radial nodes (kpc): 10^log_sigma
    (Msun/kpc^2) at each node but the last, where it is 0, linear in
    between, and 0 outside the nodes (a first node above 0 leaves a hole,
    with a jump at its edge). Thin, or thick of the scale height hz (kpc)
    of a vertical law. Its circular velocity is the pull of its rings (see
    _disk_gravity), exact for its linear profile to about 1e-10 (thin) or
    1e-7 (thick); a thin disk pulls infinitely at a jump of its surface
    density.

    Parameters
    ----------
    name : str
        Its name (see MassComponent).
    nodes : Sequence of float
        The radii (kpc) of its nodes: at least two, from 0 up, increasing.
    vertical : str, optional
        Its vertical law (see vertical.VERTICAL_LAWS): 'thin' (the
        default), 'exponential', 'sech2' or 'gauss'.

    Raises
    ------
    ConfigError
        If the nodes or the vertical law are not valid.
    """

    # The surface density falls to 0 at the last node rather than being a
    # parameter there, and is given by its log10 elsewhere: values at every
    # node (linear, from 0) would be more flexible (e.g. 0 at inner nodes),
    # but a disk that ends with a jump pulls infinitely at its edge if
    # thin, and a fit could make its density negative

    @staticmethod
    def type() -> str:
        return 'nodewise_disk'

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(
            nodes=self._nodes.tolist(), vertical=self._vertical)

    def __init__(
            self, name: str, nodes: Sequence[float], vertical: str = 'thin'):
        super().__init__(name)
        nodes = np.asarray(nodes, dtype=float)
        if (nodes.ndim != 1 or len(nodes) < 2 or not np.all(np.isfinite(nodes))
                or nodes[0] < 0 or np.any(np.diff(nodes) <= 0)):
            raise ConfigError(
                f"the nodes of a nodewise disk must be at least two radii, "
                f"from 0 up, increasing; they are {nodes.tolist()}")
        check_vertical_law(vertical)
        self._nodes = nodes
        self._vertical = vertical

    def nodes(self) -> np.ndarray:
        """Return the radii (kpc) of its nodes."""
        return self._nodes

    def vertical(self) -> str:
        """Return its vertical law."""
        return self._vertical

    def pdescs(self) -> dict[str, ParamDesc]:
        pdescs = dict(log_sigma=ParamVectorDesc(
            len(self._nodes) - 1,
            desc="log10 of the surface density (Msun/kpc^2) at each node "
                 "but the last"))
        if self._vertical != 'thin':
            pdescs['hz'] = ParamScalarDesc(
                desc="the scale height (kpc)", minimum=0)
        return pdescs

    def surface_density(
            self, R: npt.ArrayLike, params: ParamValues) -> np.ndarray:
        """
        Return its surface density (Msun/kpc^2) at the radii R (kpc; see
        vcirc2 for params).
        """
        sigma = np.append(10 ** np.asarray(params['log_sigma']), 0)
        return np.interp(R, self._nodes, sigma, left=0, right=0)

    def vcirc2(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        if self._vertical == 'thin':
            gravity = _disk_gravity.thin_disk_gravity(self._nodes, R.ravel())
            v2 = gravity.vcirc2(self.surface_density(gravity.radii, params))
        else:
            gravity = _disk_gravity.thick_disk_gravity(self._nodes, R.ravel())
            v2 = gravity.vcirc2(
                self.surface_density(gravity.radii, params)[None],
                vertical_density(
                    self._vertical, gravity.heights, params['hz']))
        return v2.reshape(R.shape)

    def density(
            self,
            R: npt.ArrayLike,
            z: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        if self._vertical == 'thin':
            raise NotImplementedError(
                "a thin disk has a surface density, not a density")
        return self.surface_density(R, params) * vertical_density(
            self._vertical, z, params['hz'])


class MassTraitDisk(MassComponent):
    """
    A disk whose mass follows the brightness of a component of the model,
    never the data: upsilon (Msun/kpc^2 per unit of the brightness) times
    the azimuthal mean of the brightness of the component, as it evaluates
    it (its brightness traits, where its selection traits keep it), at its
    subnodes and linear between them (exactly its node-wise traits), or
    at those of rings of width rstep. Vertically, as the component (its
    brightness height traits; thin without them), or of a vertical law
    (of the scale height hz, kpc). A component with warps (tilted or loose
    rings, vertical distortions) is taken flat: its rings coplanar and
    concentric, as the rotation of the velocity traits 'mass' takes them.
    Its circular velocity is the pull of its rings (see _disk_gravity); a
    thin disk pulls infinitely at a jump of its surface density (e.g. at
    the edge of the component, unless the brightness is 0 there).

    Parameters
    ----------
    name : str
        Its name (see MassComponent).
    component : str
        The name of the component of the model that it follows.
    vertical : str, optional
        Its vertical law: 'component' (the default), as the component, or
        a law of vertical.VERTICAL_LAWS ('thin', 'exponential', 'sech2'
        or 'gauss').
    rstep : float, optional
        The width (in the units of the rings of the component, e.g. arcsec)
        of the rings at whose subnodes it takes the brightness; by
        default, the subnodes of the component.

    Raises
    ------
    ConfigError
        If the name of the component, the vertical law or rstep is not
        valid.
    """

    @staticmethod
    def type() -> str:
        return 'trait_disk'

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(
            component=self._component, vertical=self._vertical,
            rstep=self._rstep)

    def __init__(
            self,
            name: str,
            component: str,
            vertical: str = 'component',
            rstep: float | None = None
    ):
        super().__init__(name)
        parseutils.check_name(component)
        if vertical != 'component':
            check_vertical_law(vertical)
        if rstep is not None and not rstep > 0:
            raise ConfigError(f"rstep must be positive; it is {rstep}")
        self._component = component
        self._vertical = vertical
        self._rstep = rstep

    def component(self) -> str:
        """Return the name of the component it follows."""
        return self._component

    def vertical(self) -> str:
        """Return its vertical law."""
        return self._vertical

    def followed_components(self) -> tuple[str, ...]:
        return (self._component,)

    def pdescs(self) -> dict[str, ParamDesc]:
        pdescs = dict(upsilon=ParamScalarDesc(
            desc="the mass per unit of the brightness of the component "
                 "(Msun/kpc^2)", minimum=0))
        if self._vertical in VERTICAL_LAWS and self._vertical != 'thin':
            pdescs['hz'] = ParamScalarDesc(
                desc="the scale height (kpc)", minimum=0)
        return pdescs

    def vcirc2(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        if self._component not in context.densities:
            raise ValueError(
                f"the mass component {self.name()!r} follows the component "
                f"{self._component!r}, whose density its context does not "
                f"give")
        density = context.densities[self._component]
        knots = density.knots(self._rstep)
        upsilon = params['upsilon']
        R = np.asarray(R, dtype=float)
        if self._vertical == 'thin' \
                or self._vertical == 'component' and not density.thick():
            gravity = _disk_gravity.thin_disk_gravity(knots, R.ravel())
            v2 = gravity.vcirc2(upsilon * np.sum(
                density.surface_densities(gravity.radii), axis=0))
        elif self._vertical == 'component':
            gravity = _disk_gravity.thick_disk_gravity(knots, R.ravel())
            v2 = gravity.vcirc2(
                upsilon * density.surface_densities(gravity.radii),
                density.height_pdfs(gravity.radii, gravity.heights))
        else:
            gravity = _disk_gravity.thick_disk_gravity(knots, R.ravel())
            v2 = gravity.vcirc2(
                upsilon * np.sum(
                    density.surface_densities(gravity.radii), axis=0,
                    keepdims=True),
                vertical_density(
                    self._vertical, gravity.heights, params['hz']))
        return v2.reshape(R.shape)
