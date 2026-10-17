"""
The surface densities of mass disks (see disks.MassDisk), and the
gravity of a disk of each, of any vertical structure (see vertical).
That of an exponential disk follows from the Hankel transform S(k) of
its surface density (Binney & Tremaine 2008, 2.165): v^2(R) = 2 pi G R
int_0^inf k J1(k R) S(k) T(k) dk, with T(k) the transform of the pdf of
its height (1 for a thin disk); that of the others, from the pull of
their rings (see _disk_gravity).
"""

import abc
import functools
from collections.abc import Callable, Sequence
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
from .base import MassContext
from .units import G
from .vertical import DiskVertical, VerticalComponent, VerticalLaw


__all__ = [
    'DiskSurface',
    'SurfaceComponent',
    'SurfaceExponential',
    'SurfaceGaussianRing',
    'SurfaceNodewise',
    'disk_surface_parser'
]


# The table of the shape of thick exponential disks of each vertical law
# (see _exponential_shape): ln F on grids of ln x (32 points per decade)
# and ln eta, interpolated by a bicubic spline (to about 1e-6, relative)
_TABLE_X = np.geomspace(1e-4, 1e4, 257)
_TABLE_ETA = np.geomspace(1e-3, 1, 49)

# The pull of the rings of an exponential disk as thick as a component
# (see SurfaceExponential): polynomials of this degree, on panels of
# widths w = (the last subnode of the component) / 2^m, the largest such
# not above rd, out to this many w (beyond 32 rd, where the surface
# density is below exp(-32) of its central value)
_COMPONENT_DEGREE = 8
_COMPONENT_PANELS = 64


# The gravity of thin Gaussian rings (see SurfaceGaussianRing): of their
# surface density within this many dispersions of the peak (beyond, below
# exp(-50) of it), by Gauss-Legendre quadrature on panels of half a
# dispersion, and of widths that halve towards each radius (where the
# kernel has a logarithm and a pole) down to 2^-40 of a dispersion
_RING_REACH = 10
_RING_X, _RING_W = np.polynomial.legendre.leggauss(8)
_RING_STEPS = 2.0 ** -np.arange(0, 41)


def _ring_vcirc2(
        R: np.ndarray,
        surface: Callable[[np.ndarray], np.ndarray],
        peak: float,
        sigma: float
) -> np.ndarray:
    """
    Return v^2 at the radii R > 0 of a thin disk of the surface density
    surface (a function) within _RING_REACH sigma of peak: 2 G PV int S(a)
    [a K / (a + R) - a E / (a - R)] da (see _disk_gravity), with the
    principal value of the pole -2 R / (a - R) as S(R) times that of 1 /
    (a - R), analytic, plus the integral of (S(a) - S(R)) / (a - R).
    """
    lo = max(0.0, peak - _RING_REACH * sigma)
    hi = peak + _RING_REACH * sigma
    r = R[:, None]
    ring = peak + sigma * np.arange(-_RING_REACH, _RING_REACH + 0.25, 0.5)
    steps = sigma * _RING_STEPS
    cuts = np.concatenate((
        np.broadcast_to(ring, (len(R), len(ring))), r - steps, r + steps,
        np.broadcast_to([lo, hi], (len(R), 2))), axis=1)
    cuts = np.sort(np.clip(cuts, lo, hi), axis=1)
    a0, a1 = cuts[:, :-1, None], cuts[:, 1:, None]
    a = (((a1 - a0) * _RING_X + a1 + a0) / 2).reshape(len(R), -1)
    w = ((a1 - a0) * _RING_W / 2).reshape(len(R), -1)
    d = a - r
    # (points that round to R, of no weight, are left out)
    used = d != 0
    d_safe = np.where(used, d, 1)
    density = surface(a)
    at_r = surface(r)
    with np.errstate(invalid='ignore', divide='ignore'):
        kernel = np.where(used, _disk_gravity.thin_kernel(r, a, d_safe), 0)
        pole = np.sum(np.where(
            used, w * (density - at_r) / d_safe, 0), axis=1)
        pole += at_r[:, 0] * np.log(np.abs((hi - R) / (lo - R)))
    return G * (np.sum(w * density * kernel, axis=1) - 2 * R * pole)


def _exponential_shape_direct(
        x: np.ndarray, transform: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    Return F(x) of _exponential_shape at each x, for T(q) given in units
    of 1 / rd.
    """
    return x * _numerics.bessel_j1_integral(
        x, lambda q: (1 + q ** 2) ** -1.5 * transform(q))


@functools.lru_cache(maxsize=None)
def _exponential_shape_table(
        law: type[VerticalLaw]) -> scipy.interpolate.RectBivariateSpline:
    """Return the table of the shape of thick disks of a vertical law."""
    table = [
        _exponential_shape_direct(
            _TABLE_X, functools.partial(law.transform, h=eta))
        for eta in _TABLE_ETA]
    return scipy.interpolate.RectBivariateSpline(
        np.log(_TABLE_ETA), np.log(_TABLE_X), np.log(table))


def _exponential_shape(
        x: np.ndarray, law: type[VerticalLaw], eta: float) -> np.ndarray:
    """
    Return F(x) = x int_0^inf q J1(q x) (1 + q^2)^(-3/2) T(q eta) dq, the
    shape of the circular velocity of a thick exponential disk of a
    vertical law (v^2 = G M / rd F(R / rd), with eta = hz / rd), at x > 0:
    from its table where the table covers x and eta, and otherwise on a
    log grid (see _numerics.on_log_grid).
    """
    out = np.empty_like(x)
    inside = (x >= _TABLE_X[0]) & (x <= _TABLE_X[-1]) \
        & (_TABLE_ETA[0] <= eta <= _TABLE_ETA[-1])
    if inside.any():
        out[inside] = np.exp(_exponential_shape_table(law).ev(
            np.log(eta), np.log(x[inside])))
    if not inside.all():
        out[~inside] = _numerics.on_log_grid(
            x[~inside], lambda xs: _exponential_shape_direct(
                xs, functools.partial(law.transform, h=eta)))
    return out


def _component_knots(knots: np.ndarray, rd: float) -> np.ndarray:
    """
    Return the knots of the pull of the rings of an exponential disk of
    the scale length rd as thick as a component of the given subnodes:
    0, the subnodes, and the edges of panels of the width w of
    _COMPONENT_PANELS (which changes in steps, so that the knots, and
    the weights of the rings, change only when it does).
    """
    m = max(0, int(np.ceil(np.log2(knots[-1] / rd))))
    width = knots[-1] / 2 ** m
    grid = width * np.arange(min(_COMPONENT_PANELS, 2 ** m) + 1)
    return np.union1d(np.union1d(0, knots), grid)


class DiskSurface(parseutils.TypedSerializable, abc.ABC):
    """
    The surface density of a mass disk: its mass seen face-on, and the
    gravity of the disk of it and a vertical structure.
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    def followed_component(self) -> str | None:
        """
        Return the name of the component of the galaxy whose density it
        follows; None here.
        """
        return None

    @abc.abstractmethod
    def pdescs(self) -> dict[str, ParamDesc]:
        """Return its parameters, by name."""
        pass

    @abc.abstractmethod
    def surface_density(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return the surface density (Msun/kpc^2) at the radii R (kpc; see
        MassComponent.vcirc2 for the other arguments).

        Raises
        ------
        NotImplementedError
            If it does not give its surface density.
        """
        pass

    @abc.abstractmethod
    def vcirc2(
            self,
            R: npt.ArrayLike,
            vertical: DiskVertical,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """
        Return the square of the circular velocity in the plane of the
        disk of this surface density and the vertical structure (see
        MassComponent.vcirc2; params has the values of the parameters of
        both).
        """
        pass


class SurfaceExponential(DiskSurface):
    """
    An exponential disk of total mass 10^log_m (Msun) and scale length rd
    (kpc): the surface density M / (2 pi rd^2) exp(-R / rd), with S(k) =
    M / (2 pi) (1 + k^2 rd^2)^(-3/2). Its circular velocity is v^2 = G M
    / rd F(R / rd), with F of the integral above:

    - thin, the closed form of Freeman (1970);
    - of a vertical law, of the scale height hz, computed numerically
      (interpolated in a table for hz / rd from 1e-3 to 1 and R / rd from
      1e-4 to 1e4, to about 1e-6, and computed directly otherwise);
    - as thick as a component (see vertical.VerticalComponent): its
      shape beyond its last subnode, where it is held, at every radius,
      by the integral above (with T(k) of the shape, computed on the
      heights of _disk_gravity), plus the difference to its shape within
      the last subnode, by the pull of the rings there (polynomials of
      degree 8 on panels no wider than rd out to 32 rd at least, beyond
      which the surface density is below exp(-32) of its central value);
      to about 1e-6.
    """

    @staticmethod
    def type() -> str:
        return 'exponential'

    def pdescs(self) -> dict[str, ParamDesc]:
        return dict(
            log_m=ParamScalarDesc(desc="log10 of the mass (Msun)"),
            rd=ParamScalarDesc(desc="the scale length (kpc)", minimum=0))

    def surface_density(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        mass = 10 ** params['log_m']
        rd = params['rd']
        return mass / (2 * np.pi * rd ** 2) * np.exp(
            -np.asarray(R, dtype=float) / rd)

    def vcirc2(
            self,
            R: npt.ArrayLike,
            vertical: DiskVertical,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        mass = 10 ** params['log_m']
        rd = params['rd']
        R = np.asarray(R, dtype=float)
        if vertical.thin(context):
            # 2 G M / rd y^2 (I0 K0 - I1 K1), with y = R / (2 rd); the
            # scaled Bessel functions keep the products finite at large
            # y, and the limit at y = 0 is 0
            y = R / (2 * rd)
            y_safe = np.where(y > 0, y, 1)
            bessel = (
                scipy.special.i0e(y_safe) * scipy.special.k0e(y_safe)
                - scipy.special.i1e(y_safe) * scipy.special.k1e(y_safe))
            return np.where(y > 0, 2 * G * mass / rd * y_safe ** 2 * bessel, 0)
        if isinstance(vertical, VerticalLaw):
            # v^2 = G M / rd F(R / rd) (see _exponential_shape)
            x = R / rd
            out = np.zeros_like(R)
            positive = x > 0
            if positive.any():
                out[positive] = _exponential_shape(
                    x[positive], type(vertical), params['hz'] / rd)
            return G * mass / rd * out
        return self._vcirc2_component(R, vertical, params, context)

    def _vcirc2_component(
            self,
            R: np.ndarray,
            vertical: VerticalComponent,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        """Return vcirc2 as thick as a component (see above)."""
        mass = 10 ** params['log_m']
        rd = params['rd']
        gravity = _disk_gravity.thick_disk_gravity(
            _component_knots(vertical.knots(context), rd), R.ravel(),
            _COMPONENT_DEGREE)
        shapes = vertical.shapes(gravity.radii, gravity.heights, context)
        outer = shapes[-1]

        # The outer shape at every radius: v^2 = G M / rd F(R / rd), with
        # T(q / rd) of the shape, smooth in ln k (see on_log_grid)
        def transform(q: np.ndarray) -> np.ndarray:
            return _numerics.on_log_grid(
                q / rd, lambda k: gravity.height_transform(outer, k))
        x = R / rd
        out = np.zeros_like(R)
        positive = x > 0
        if positive.any():
            out[positive] = _numerics.on_log_grid(
                x[positive],
                lambda xs: _exponential_shape_direct(xs, transform))
        # The difference to the shape within the subnodes, of the rings
        sigma = self.surface_density(gravity.radii, params, context)
        inner = gravity.vcirc2(
            np.array([sigma, -sigma]),
            np.array([shapes, np.broadcast_to(outer, shapes.shape)]))
        return G * mass / rd * out + inner.reshape(R.shape)


class SurfaceGaussianRing(DiskSurface):
    """
    A Gaussian ring of total mass 10^log_m (Msun), of the peak radius
    r_peak (kpc) and the full width at half maximum fwhm (kpc): the
    surface density S_0 exp(-(R - r_peak)^2 / (2 s^2)), s = fwhm / (2
    (2 ln 2)^(1/2)), with M = 2 pi S_0 (s^2 exp(-r_peak^2 / (2 s^2)) +
    r_peak s (pi / 2)^(1/2) (1 + erf(r_peak / (s 2^(1/2))))). Thin only
    (as DysmalPy's GaussianRing): its circular velocity is the pull of its
    rings, by quadrature (see _ring_vcirc2), to about 1e-10 of its largest
    value; negative inside it, which it pulls outwards.
    """

    @staticmethod
    def type() -> str:
        return 'gaussian_ring'

    def pdescs(self) -> dict[str, ParamDesc]:
        return dict(
            log_m=ParamScalarDesc(desc="log10 of the mass (Msun)"),
            r_peak=ParamScalarDesc(
                desc="the radius of the peak (kpc)", minimum=0),
            fwhm=ParamScalarDesc(
                desc="the full width at half maximum (kpc)", minimum=0))

    def surface_density(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        peak, sigma = params['r_peak'], self._sigma(params)
        mass = 10 ** params['log_m']
        integral = sigma ** 2 * np.exp(-peak ** 2 / (2 * sigma ** 2)) \
            + peak * sigma * np.sqrt(np.pi / 2) * (
                1 + scipy.special.erf(peak / (sigma * np.sqrt(2))))
        return mass / (2 * np.pi * integral) * np.exp(
            -(np.asarray(R, dtype=float) - peak) ** 2 / (2 * sigma ** 2))

    def vcirc2(
            self,
            R: npt.ArrayLike,
            vertical: DiskVertical,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        if not vertical.thin(context):
            raise NotImplementedError(
                "a Gaussian ring is thin only (vertical: thin)")
        R = np.asarray(R, dtype=float)
        radii = R.ravel()
        out = np.zeros_like(radii)
        positive = radii > 0
        out[positive] = _ring_vcirc2(
            radii[positive],
            lambda a: self.surface_density(a, params, context),
            params['r_peak'], self._sigma(params))
        return out.reshape(R.shape)

    @staticmethod
    def _sigma(params: ParamValues) -> float:
        """Return its dispersion s (kpc)."""
        return params['fwhm'] / (2 * np.sqrt(2 * np.log(2)))


class SurfaceNodewise(DiskSurface):
    """
    A surface density given at radial nodes (kpc): 10^log_sigma
    (Msun/kpc^2) at each node but the last, where it is 0, linear in
    between, and 0 outside the nodes (a first node above 0 leaves a hole,
    with a jump at its edge). Its circular velocity is the pull of its
    rings (see _disk_gravity), exact for its linear profile to about
    1e-10 (thin) or 1e-7 (thick; as thick as a component, polynomials of
    degree 2 between the nodes and the subnodes of the component, where
    both are linear); a thin disk pulls infinitely at a jump of its
    surface density.

    Parameters
    ----------
    nodes : Sequence of float
        The radii (kpc) of its nodes: at least two, from 0 up, increasing.

    Raises
    ------
    ConfigError
        If the nodes are not valid.
    """

    # The surface density falls to 0 at the last node rather than being a
    # parameter there, and is given by its log10 elsewhere: values at every
    # node (linear, from 0) would be more flexible (e.g. 0 at inner nodes),
    # but a disk that ends with a jump pulls infinitely at its edge if
    # thin, and a fit could make its density negative

    @staticmethod
    def type() -> str:
        return 'nodewise'

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(nodes=self._nodes.tolist())

    def __init__(self, nodes: Sequence[float]):
        nodes = np.asarray(nodes, dtype=float)
        if (nodes.ndim != 1 or len(nodes) < 2 or not np.all(np.isfinite(nodes))
                or nodes[0] < 0 or np.any(np.diff(nodes) <= 0)):
            raise ConfigError(
                f"the nodes of a nodewise surface density must be at least "
                f"two radii, from 0 up, increasing; they are "
                f"{nodes.tolist()}")
        self._nodes = nodes

    def nodes(self) -> np.ndarray:
        """Return the radii (kpc) of its nodes."""
        return self._nodes

    def pdescs(self) -> dict[str, ParamDesc]:
        return dict(log_sigma=ParamVectorDesc(
            len(self._nodes) - 1,
            desc="log10 of the surface density (Msun/kpc^2) at each node "
                 "but the last"))

    def surface_density(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        sigma = np.append(10 ** np.asarray(params['log_sigma']), 0)
        return np.interp(R, self._nodes, sigma, left=0, right=0)

    def vcirc2(
            self,
            R: npt.ArrayLike,
            vertical: DiskVertical,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        if vertical.thin(context):
            gravity = _disk_gravity.thin_disk_gravity(self._nodes, R.ravel())
            v2 = gravity.vcirc2(
                self.surface_density(gravity.radii, params, context))
        elif isinstance(vertical, VerticalLaw):
            gravity = _disk_gravity.thick_disk_gravity(self._nodes, R.ravel())
            v2 = gravity.vcirc2(
                self.surface_density(gravity.radii, params, context)[None],
                vertical.pdf(gravity.heights, params['hz']))
        else:
            gravity = _disk_gravity.thick_disk_gravity(
                np.union1d(self._nodes, vertical.knots(context)), R.ravel(),
                degree=2)
            v2 = gravity.vcirc2(
                self.surface_density(gravity.radii, params, context)[None],
                vertical.shapes(gravity.radii, gravity.heights, context)[None])
        return v2.reshape(R.shape)


class SurfaceComponent(DiskSurface):
    """
    The surface density of a disk component of the galaxy (light, the
    optical depth of dust, ...), never the data: upsilon times the
    azimuthal mean of the density of the component, as it evaluates it
    (its density polar traits, where its selection traits keep it), at
    its subnodes and linear between them (exactly its node-wise traits),
    or at those of rings of width rstep. A component with warps (tilted
    or loose rings, vertical distortions) is taken flat: its rings
    coplanar and concentric, as the rotation of the velocity traits
    'tan_mass' takes them. Its circular velocity is the pull of its rings
    (see _disk_gravity); a thin disk pulls infinitely at a jump of its
    surface density (e.g. at the edge of the component, unless the
    density is 0 there). The azimuthal means of axisymmetric traits are
    exact (one azimuth); those of the others take at least 128 azimuths,
    more until they converge: structure narrower than their spacing (e.g.
    tiny blobs) can be missed.

    upsilon is the mass per unit of the density of the component: without
    a unit of its emission, in Msun/kpc^2 per unit; with one, of HI (its
    first line hi21), the mass of gas per mass of HI, of which the density
    gives the exact mass (1.33 with helium; see _units.mass_per_unit);
    with another unit, in Msun/pc^2 per unit (e.g. alpha_CO in Msun/pc^2
    per K km/s, or a mass-to-light ratio).

    Parameters
    ----------
    component : str
        The name of the component of its galaxy that it follows.
    rstep : float, optional
        The width (in the units of the rings of the component, e.g. arcsec)
        of the rings at whose subnodes it takes the density; by
        default, the subnodes of the component.

    Raises
    ------
    ConfigError
        If the name of the component or rstep is not valid.
    """

    @staticmethod
    def type() -> str:
        return 'component'

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(
            component=self._component, rstep=self._rstep)

    def __init__(self, component: str, rstep: float | None = None):
        parseutils.check_name(component)
        if rstep is not None and not rstep > 0:
            raise ConfigError(f"rstep must be positive; it is {rstep}")
        self._component = component
        self._rstep = rstep

    def component(self) -> str:
        """Return the name of the component it follows."""
        return self._component

    def rstep(self) -> float | None:
        """Return the width of the rings of its subnodes (see above)."""
        return self._rstep

    def followed_component(self) -> str | None:
        return self._component

    def pdescs(self) -> dict[str, ParamDesc]:
        return dict(upsilon=ParamScalarDesc(
            desc="the mass per unit of the density of the component (see "
                 "SurfaceComponent)", minimum=0))

    def surface_density(
            self,
            R: npt.ArrayLike,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        raise NotImplementedError(
            "a surface density that follows a component is given on rings "
            "only")

    def vcirc2(
            self,
            R: npt.ArrayLike,
            vertical: DiskVertical,
            params: ParamValues,
            context: MassContext
    ) -> np.ndarray:
        density = context.density(self._component)
        knots = density.knots(self._rstep)
        upsilon = params['upsilon']
        if (mass := density.mass_per_unit()) is not None:
            upsilon = upsilon * mass
        R = np.asarray(R, dtype=float)
        if vertical.thin(context):
            gravity = _disk_gravity.thin_disk_gravity(knots, R.ravel())
            v2 = gravity.vcirc2(upsilon * np.sum(
                density.surface_densities(gravity.radii), axis=0))
        elif isinstance(vertical, VerticalLaw):
            gravity = _disk_gravity.thick_disk_gravity(knots, R.ravel())
            v2 = gravity.vcirc2(
                upsilon * np.sum(
                    density.surface_densities(gravity.radii), axis=0,
                    keepdims=True),
                vertical.pdf(gravity.heights, params['hz']))
        else:
            # As thick as the same component (see disks.MassDisk): its
            # density, each term of the pdf of its height
            gravity = _disk_gravity.thick_disk_gravity(
                np.union1d(knots, vertical.knots(context)), R.ravel())
            v2 = gravity.vcirc2(
                upsilon * density.surface_densities(gravity.radii),
                density.height_pdfs(gravity.radii, gravity.heights))
        return v2.reshape(R.shape)


disk_surface_parser = parseutils.TypedParser(DiskSurface, [
    SurfaceExponential,
    SurfaceNodewise,
    SurfaceGaussianRing,
    SurfaceComponent])
