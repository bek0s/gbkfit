"""
The masses of a gravitational lens (see Lens): each deflects the light
of the source plane, and the deflections of the masses of a lens add up.

The analytic masses are placed by their centre (x, y; arcsec, in the
frame of the model, as xpos and ypos) and oriented by the position angle
(pa; degrees, north through east, as that of disks) of their major axis
(u, from +y towards -x; the minor axis v is 90 degrees clockwise of it).
Their convergence is elliptical in xi = sqrt(q u^2 + v^2 / q), so the
mass within the ellipse xi of an axis ratio q is that of the circle of
radius xi of the round mass.
"""

import abc
import math
import os.path
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import scipy.ndimage

from gbkfit.dataset import FitsFile, fits_file
from gbkfit.params import ParamScalarDesc, ParamValues
from gbkfit.utils import fitsutils, gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'LensDeflectionMap',
    'LensEPL',
    'LensMass',
    'LensMultipole',
    'LensNFW',
    'LensSIE',
    'LensShear',
    'lens_mass_parser'
]


class LensMass(parseutils.TypedSerializable, abc.ABC):
    """
    A mass of a gravitational lens (see Lens): its deflection of the light
    at each point of the image plane.

    A mass may have parameters (see pdescs), named in its lens after it
    (see Lens.pdescs).

    Parameters
    ----------
    name : str, optional
        Its name, which prefixes its parameters when its lens has several
        masses.

    Raises
    ------
    ConfigError
        If the name is not valid (see parseutils.check_name).
    """

    @classmethod
    def load(cls, info: dict[str, Any], prefix: str = '') -> 'LensMass':
        """
        Load a mass from its configuration; the prefix of the filenames of
        its files, if any, is prepended to them.
        """
        return cls(**parseutils.parse_options_for_callable(info, cls.__init__))

    def dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {} if self._name is None else dict(name=self._name)

    def __init__(self, name: str | None = None):
        parseutils.check_name(name)
        self._name = name

    def name(self) -> str | None:
        """Return its name, if any."""
        return self._name

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        """Return its parameters, by name (none by default)."""
        return {}

    @abc.abstractmethod
    def deflection(
            self, grid: gridutils.Grid, params: ParamValues
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the deflection at the pixels of a grid of the image plane.

        Parameters
        ----------
        grid : Grid
            The grid; its x and y axes.
        params : ParamValues
            The values of its parameters, by their names (see pdescs).

        Returns
        -------
        tuple of np.ndarray
            The deflection (arcsec) along x and y of the frame of the
            model, each of shape (ny, nx).
        """
        pass


class LensDeflectionMap(LensMass):
    """
    A mass of a given deflection: maps of it on a grid of the image plane
    (e.g. from a lens model), placed by their RA and Dec.

    Between the pixels of the maps the deflection is interpolated
    (bilinearly), and beyond them it is that of their nearest edge (e.g.
    in the padding of the convolution).

    Parameters
    ----------
    alpha_x, alpha_y : np.ndarray
        The deflection (arcsec) along x and y of the frame of the model
        (like xpos and ypos), each of shape (ny, nx).
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the maps (see gridutils.make_grid).
    rota : float, optional
        The rotation of the maps on the sky.
    scaled : bool, optional
        Whether the deflection is scaled by a parameter, 'scale' (fixed
        or fitted; see pdescs): e.g. D_LS / D_S of the source over that
        of the maps, when they are of a source at another redshift.
    name : str, optional
        Its name (see LensMass).

    Raises
    ------
    ConfigError
        If the maps are not two finite images of one shape.
    """

    @staticmethod
    def type() -> str:
        return 'deflection_map'

    @classmethod
    def load(
            cls, info: dict[str, Any], prefix: str = ''
    ) -> 'LensDeflectionMap':
        info = dict(info)
        files = {}
        for key in ('alpha_x', 'alpha_y'):
            files[key] = parseutils.load_option(
                fits_file, info, key, required=True, prefix=prefix)
            del info[key]
        options = parseutils.parse_options_for_callable(
            info, cls.from_files, ignore_params=list(files))
        return cls.from_files(**files, **options)

    @classmethod
    def from_files(
            cls,
            alpha_x: FitsFile,
            alpha_y: FitsFile,
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None,
            scaled: bool = False,
            name: str | None = None
    ) -> 'LensDeflectionMap':
        """
        Read a mass from FITS files of its deflection maps, with the world
        coordinates of the files unless given; they must agree.

        Parameters
        ----------
        alpha_x, alpha_y : str or tuple
            The files of the maps: each a filename, or a filename and the
            HDU.
        step, rpix, rval : float or Sequence of float, optional
            The world coordinates of the maps (see gridutils.make_grid); by
            default, those of the headers. Either rpix or rval can be
            given, and the other comes from the headers (see
            fitsutils.read_data).
        rota : float, optional
            The rotation of the maps on the sky; by default, that of the
            headers.
        scaled : bool, optional
            Whether the deflection is scaled by a parameter (see the
            class).
        name : str, optional
            Its name (see LensMass).

        Returns
        -------
        LensDeflectionMap
            The mass.

        Raises
        ------
        ConfigError
            If the maps have different world coordinates.
        """
        maps = []
        for file in (alpha_x, alpha_y):
            filename, hdu = (file, 0) if isinstance(file, str) else file
            maps.append(fitsutils.read_data(filename, hdu, rpix, rval))
        (data_x, coords), (data_y, coords_y) = maps
        if coords_y != coords:
            raise ConfigError(
                "the deflection maps have different world coordinates")
        return cls(
            data_x, data_y, coords.step if step is None else step,
            coords.rpix, coords.rval, coords.rota if rota is None else rota,
            scaled, name)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        info = super().dump()
        for key, data in (
                ('alpha_x', self._alpha_x), ('alpha_y', self._alpha_y)):
            filename = f'{prefix}{key}.fits'
            fitsutils.write_data(
                filename, data, self._grid.coords, None, overwrite)
            info[key] = filename if dump_path else os.path.basename(filename)
        if self._scaled:
            info['scaled'] = True
        return info

    def __init__(
            self,
            alpha_x: np.ndarray,
            alpha_y: np.ndarray,
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None,
            scaled: bool = False,
            name: str | None = None
    ):
        super().__init__(name)
        alpha_x = np.asarray(alpha_x, dtype=float)
        alpha_y = np.asarray(alpha_y, dtype=float)
        if alpha_x.ndim != 2 or alpha_x.shape != alpha_y.shape:
            raise ConfigError(
                f"the deflection maps must be two images of one shape; they "
                f"have the shapes {alpha_x.shape} and {alpha_y.shape}")
        if not (np.all(np.isfinite(alpha_x)) and np.all(np.isfinite(alpha_y))):
            raise ConfigError("the deflection maps must be finite")
        self._alpha_x = alpha_x
        self._alpha_y = alpha_y
        self._grid = gridutils.make_grid(
            alpha_x.shape[::-1], step, rpix, rval, rota)
        self._scaled = scaled

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        if not self._scaled:
            return {}
        return dict(scale=ParamScalarDesc(
            "the scale of the deflection (e.g. D_LS / D_S)", minimum=0))

    def deflection(
            self, grid: gridutils.Grid, params: ParamValues
    ) -> tuple[np.ndarray, np.ndarray]:
        gridutils.check_overlap(grid, self._grid, "the deflection maps")
        pixel_x, pixel_y = gridutils.pixels_on(grid, self._grid)
        scale = params['scale'] if self._scaled else 1.0

        def sample(data):
            return scale * scipy.ndimage.map_coordinates(
                data, [pixel_y, pixel_x], order=1, mode='nearest')
        return sample(self._alpha_x), sample(self._alpha_y)


class LensSIE(LensMass):
    """
    A singular isothermal ellipsoid (SIE): the standard model of a galaxy
    lens, of convergence

        kappa = theta_e / (2 xi),

    whose deflection is analytic (Kormann et al. 1994): with
    f = sqrt(1 - q^2) and psi = sqrt(q^2 u^2 + v^2),

        alpha_u = theta_e sqrt(q) / f * arctan(f u / psi)
        alpha_v = theta_e sqrt(q) / f * arctanh(f v / psi)

    (theta_e u / r and theta_e v / r, a singular isothermal sphere, for
    q = 1); 0 at its centre.

    Its parameters (see pdescs): theta_e (arcsec, the Einstein radius of
    the sphere of the same mass), q (the axis ratio), pa, x and y (see
    the module).

    Parameters
    ----------
    name : str, optional
        Its name (see LensMass).
    """

    @staticmethod
    def type() -> str:
        return 'sie'

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(theta_e=ParamScalarDesc(
            "the Einstein radius (arcsec)", minimum=0)) | _ellipse_pdescs()

    def deflection(
            self, grid: gridutils.Grid, params: ParamValues
    ) -> tuple[np.ndarray, np.ndarray]:
        u, v = _on_axes(grid, params)
        alpha_u, alpha_v = _sie_deflection(
            u, v, params['theta_e'], params['q'])
        return _from_axes(alpha_u, alpha_v, params['pa'])


class LensEPL(LensMass):
    """
    An elliptical power law (EPL): a galaxy lens whose density falls as
    r^-slope (an SIE for slope 2), of convergence

        kappa = (2 - t) / 2 * (theta_e / xi)^t,   t = slope - 1,

    whose deflection is analytic (Tessore & Metcalf 2015): with
    b = theta_e sqrt(q), R e^(i phi) = q u + i v and f = (1 - q) / (1 + q),

        alpha_u + i alpha_v = 2 b / (1 + q) * (b / R)^(t - 1) * e^(i phi)
                              * 2F1(1, t / 2; 2 - t / 2; -f e^(2 i phi)),

    the hypergeometric series summed to a relative error of 1e-12; 0 at
    its centre.

    Its parameters (see pdescs): theta_e (arcsec, the Einstein radius of
    the round mass of the same profile), slope (of the density; 1 to 3),
    q (the axis ratio), pa, x and y (see the module).

    Parameters
    ----------
    name : str, optional
        Its name (see LensMass).
    """

    @staticmethod
    def type() -> str:
        return 'epl'

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(
            theta_e=ParamScalarDesc(
                "the Einstein radius (arcsec)", minimum=0),
            slope=ParamScalarDesc(
                "the slope of the density (2 = isothermal)",
                minimum=1, maximum=3)) | _ellipse_pdescs()

    def deflection(
            self, grid: gridutils.Grid, params: ParamValues
    ) -> tuple[np.ndarray, np.ndarray]:
        u, v = _on_axes(grid, params)
        alpha_u, alpha_v = _epl_deflection(
            u, v, params['theta_e'], params['slope'], params['q'])
        return _from_axes(alpha_u, alpha_v, params['pa'])


class LensNFW(LensMass):
    """
    A Navarro-Frenk-White (NFW) halo: the lens of a dark matter halo (e.g.
    of a group or a cluster), of convergence (Bartelmann 1996)

        kappa = 2 kappa_s (1 - F(x)) / (x^2 - 1),   x = xi / r_s,

    with F(x) = arccosh(1 / x) / sqrt(1 - x^2) (arccos and
    sqrt(x^2 - 1) for x > 1), and kappa_s = rho_s r_s / Sigma_cr.

    Round (q = 1), its deflection is analytic: 4 kappa_s r_s h(x) / x
    along the radius, with h(x) = ln(x / 2) + F(x).
    Elliptical, it is the integral over the ellipses within each point
    (Keeton 2001; a tanh-sinh quadrature of 52 nodes, to a relative error
    below 1e-8 from 1e-3 to 1e3 r_s). It is 0 at its centre.

    Its parameters (see pdescs): kappa_s, r_s (arcsec, the scale radius),
    q (the axis ratio), pa, x and y (see the module).

    Parameters
    ----------
    name : str, optional
        Its name (see LensMass).
    """

    @staticmethod
    def type() -> str:
        return 'nfw'

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(
            kappa_s=ParamScalarDesc(
                "the convergence scale (rho_s r_s / Sigma_cr)", minimum=0),
            r_s=ParamScalarDesc("the scale radius (arcsec)", minimum=0)
        ) | _ellipse_pdescs()

    def deflection(
            self, grid: gridutils.Grid, params: ParamValues
    ) -> tuple[np.ndarray, np.ndarray]:
        u, v = _on_axes(grid, params)
        kappa_s, r_s, q = params['kappa_s'], params['r_s'], params['q']
        if q == 1:
            alpha_u, alpha_v = _nfw_round_deflection(u, v, kappa_s, r_s)
        else:
            alpha_u, alpha_v = _elliptical_deflection(
                u, v, q, lambda xi: kappa_s * _nfw_convergence(xi / r_s))
        return _from_axes(alpha_u, alpha_v, params['pa'])


class LensMultipole(LensMass):
    """
    A multipole of a lens: the deviation of a galaxy lens from an
    ellipse (e.g. order 4 for disky or boxy isodensity contours, 3 and 1
    for lopsided ones), of convergence (e.g. Van de Vyvere et al. 2022)

        kappa = theta_e a / (2 r) * cos(m (phi - phi_m))

    at the radius r and the angle phi about its centre, with phi_m that
    of the position angle pa (a maximum of the convergence, for a > 0:
    with the pa of the major axis of the lens, disky for order 4 and
    a > 0, boxy for a < 0). Its deflection is that of the potential

        psi = theta_e a / (1 - m^2) * r * cos(m (phi - phi_m))

    for m > 1, and (theta_e a / 2) r ln(r / theta_e) cos(phi - phi_1)
    for m = 1; 0 at its centre. Its centre and theta_e are usually those
    of the lens it deviates (tied by expressions).

    Its parameters (see pdescs): theta_e (arcsec; the scale of a), a (the
    strength), pa (degrees), x and y (see the module).

    Parameters
    ----------
    order : int
        The order m (at least 1).
    name : str, optional
        Its name (see LensMass).

    Raises
    ------
    ConfigError
        If the order is not an integer of at least 1.
    """

    @staticmethod
    def type() -> str:
        return 'multipole'

    def dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return super().dump() | dict(order=self._order)

    def __init__(self, order: int, name: str | None = None):
        super().__init__(name)
        if isinstance(order, bool) or not isinstance(order, int) \
                or order < 1:
            raise ConfigError(
                f"the order of a multipole must be an integer of at least "
                f"1; it is {order!r}")
        self._order = order

    def order(self) -> int:
        """Return its order."""
        return self._order

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(
            theta_e=ParamScalarDesc(
                "the Einstein radius of its lens (arcsec)", minimum=0),
            a=ParamScalarDesc("the strength"),
            pa=ParamScalarDesc("the position angle (degrees)"),
        ) | _centre_pdescs()

    def deflection(
            self, grid: gridutils.Grid, params: ParamValues
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = _centred(grid, params)
        r = np.hypot(x, y)
        centre = r == 0
        r = np.where(centre, 1, r)
        cos_phi, sin_phi = x / r, y / r
        m = self._order
        theta_e = params['theta_e']
        strength = theta_e * params['a']
        # (the angle from +x of the position angle is 90 degrees more)
        angle = m * (np.arctan2(y, x) - math.radians(params['pa'] + 90))
        if m == 1:
            # (no mass is no deflection, whatever the scale of ln r)
            log_r = np.log(r / theta_e) if theta_e > 0 else 0
            alpha_r = strength / 2 * (log_r + 1) * np.cos(angle)
            alpha_phi = -strength / 2 * log_r * np.sin(angle)
        else:
            alpha_r = strength / (1 - m * m) * np.cos(angle)
            alpha_phi = -m * strength / (1 - m * m) * np.sin(angle)
        alpha_x = alpha_r * cos_phi - alpha_phi * sin_phi
        alpha_y = alpha_r * sin_phi + alpha_phi * cos_phi
        return np.where(centre, 0, alpha_x), np.where(centre, 0, alpha_y)


class LensShear(LensMass):
    """
    An external shear: the tidal stretch of the masses near a lens, of
    strength gamma along the position angle pa (degrees, north through
    east). Its deflection is linear, 0 at the origin of the frame of the
    model (its deflection about another point differs by a constant,
    which moves only the source).

    Its parameters (see pdescs): gamma and pa.

    Parameters
    ----------
    name : str, optional
        Its name (see LensMass).
    """

    @staticmethod
    def type() -> str:
        return 'shear'

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(
            gamma=ParamScalarDesc("the strength", minimum=0),
            pa=ParamScalarDesc("the position angle (degrees)"))

    def deflection(
            self, grid: gridutils.Grid, params: ParamValues
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = gridutils.sky_positions(grid)
        # (the angle from +x of the position angle is 90 degrees more)
        angle = 2 * math.radians(params['pa'] + 90)
        gamma1 = params['gamma'] * math.cos(angle)
        gamma2 = params['gamma'] * math.sin(angle)
        return gamma1 * x + gamma2 * y, gamma2 * x - gamma1 * y


lens_mass_parser = parseutils.TypedParser(LensMass, [
    LensDeflectionMap, LensEPL, LensMultipole, LensNFW, LensSIE, LensShear])


def _centre_pdescs() -> dict[str, ParamScalarDesc]:
    """Return the parameters of the centre of a mass."""
    return dict(
        x=ParamScalarDesc("the centre along x (arcsec)"),
        y=ParamScalarDesc("the centre along y (arcsec)"))


def _ellipse_pdescs() -> dict[str, ParamScalarDesc]:
    """Return the parameters of the shape of an elliptical mass."""
    return dict(
        # (q > 0 keeps the deflection finite)
        q=ParamScalarDesc("the axis ratio", minimum=1e-3, maximum=1),
        pa=ParamScalarDesc("the position angle (degrees)"),
    ) | _centre_pdescs()


def _centred(
        grid: gridutils.Grid, params: ParamValues
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the positions of the pixels of a grid from the centre of a
    mass, along x and y.
    """
    x, y = gridutils.sky_positions(grid)
    return x - params['x'], y - params['y']


def _on_axes(
        grid: gridutils.Grid, params: ParamValues
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the positions of the pixels of a grid from the centre of a
    mass, along its major (u) and minor (v) axes (see the module).
    """
    x, y = _centred(grid, params)
    pa = math.radians(params['pa'])
    sin_pa, cos_pa = math.sin(pa), math.cos(pa)
    return -sin_pa * x + cos_pa * y, cos_pa * x + sin_pa * y


def _from_axes(
        alpha_u: np.ndarray, alpha_v: np.ndarray, pa: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a deflection along the axes of a mass of position angle pa
    (degrees) along x and y.
    """
    pa = math.radians(pa)
    sin_pa, cos_pa = math.sin(pa), math.cos(pa)
    return (-sin_pa * alpha_u + cos_pa * alpha_v,
            cos_pa * alpha_u + sin_pa * alpha_v)


def _sie_deflection(
        u: np.ndarray,
        v: np.ndarray,
        theta_e: float,
        q: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection of an SIE along its major (u) and minor (v)
    axes (see LensSIE); 0 at its centre.
    """
    r = np.hypot(u, v)
    centre = r == 0
    if q > 1 - 1e-8:
        with np.errstate(invalid='ignore', divide='ignore'):
            scale = np.where(centre, 0, theta_e / r)
        return scale * u, scale * v
    f = math.sqrt(1 - q * q)
    psi = np.where(centre, 1, np.sqrt(q * q * u * u + v * v))
    scale = theta_e * math.sqrt(q) / f
    return (scale * np.arctan(f * u / psi),
            scale * np.arctanh(f * v / psi))


def _epl_deflection(
        u: np.ndarray,
        v: np.ndarray,
        theta_e: float,
        slope: float,
        q: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection of an EPL along its major (u) and minor (v)
    axes (see LensEPL); 0 at its centre.
    """
    t = slope - 1
    b = theta_e * math.sqrt(q)
    z = q * u + 1j * v
    radius = np.abs(z)
    centre = radius == 0
    radius = np.where(centre, 1, radius)
    # The series of e^(i phi) 2F1(...): its terms fall at least as f^n, so
    # the rest after n terms is below f^n / (1 - f)
    f = (1 - q) / (1 + q)
    term = z / radius
    ratio = -f * term * term
    series = term
    n = 1
    while f ** n > 1e-12 * (1 - f):
        term = term * ratio * (2 * n - 2 + t) / (2 * n + 2 - t)
        series = series + term
        n += 1
    alpha = 2 / (1 + q) * b ** t * radius ** (1 - t) * series
    alpha = np.where(centre, 0, alpha)
    return alpha.real, alpha.imag


def _nfw_f(x: np.ndarray) -> np.ndarray:
    """
    Return F(x) of an NFW halo (see LensNFW): arccosh(1 / x) /
    sqrt(1 - x^2), and arccos(1 / x) / sqrt(x^2 - 1) for x > 1; the sum
    of (1 - x^2)^k / (2k + 1) near x = 1.
    """
    z = 1 - x * x
    near = np.abs(z) < 0.1
    series = sum(np.where(near, z, 0) ** k / (2 * k + 1) for k in range(16))
    with np.errstate(divide='ignore'):
        inverse = 1 / x
    angle = np.where(
        x < 1, np.arccosh(np.maximum(inverse, 1)),
        np.arccos(np.minimum(inverse, 1)))
    root = np.sqrt(np.abs(np.where(near, 1, z)))
    return np.where(near, series, angle / root)


def _nfw_convergence(x: np.ndarray) -> np.ndarray:
    """
    Return the convergence of an NFW halo over its kappa_s at x times its
    scale radius (see LensNFW): 2 (F(x) - 1) / (1 - x^2), the sum of
    2 (1 - x^2)^(k - 1) / (2k + 1) from k = 1 near x = 1.
    """
    z = 1 - x * x
    near = np.abs(z) < 0.1
    series = sum(np.where(near, z, 0) ** (k - 1) / (2 * k + 1)
                 for k in range(1, 17))
    return 2 * np.where(near, series, (_nfw_f(x) - 1) / np.where(near, 1, z))


def _nfw_round_deflection(
        u: np.ndarray,
        v: np.ndarray,
        kappa_s: float,
        r_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection of a round NFW halo (see LensNFW) along u and v:
    4 kappa_s r_s h(x) / x along the radius, with x = r / r_s and
    h(x) = ln(x / 2) + F(x).
    """
    r = np.hypot(u, v)
    centre = r == 0
    x = np.where(centre, 1, r) / r_s
    h = np.log(x / 2) + _nfw_f(x)
    scale = np.where(centre, 0, 4 * kappa_s * h / (x * x))
    return scale * u, scale * v


def _tanh_sinh_nodes(
        step: float = 1 / 8, extent: float = 3.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the nodes w of the tanh-sinh quadrature over [0, 1], 1 - w
    (exact near w = 1) and their weights.
    """
    tau = np.arange(-extent, extent + step / 2, step)
    s = math.pi / 2 * np.sinh(tau)
    weights = step * math.pi / 4 * np.cosh(tau) / np.cosh(s) ** 2
    return 1 / (1 + np.exp(-2 * s)), 1 / (1 + np.exp(2 * s)), weights


_NODES, _ONE_MINUS_NODES, _WEIGHTS = _tanh_sinh_nodes()


def _elliptical_deflection(
        u: np.ndarray,
        v: np.ndarray,
        q: float,
        kappa: Callable[[np.ndarray], np.ndarray],
        chunk: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection of an elliptical convergence kappa(xi) (a
    function of xi; see the module) along its major (u) and minor (v)
    axes, by the integrals over the ellipses within each point (Keeton
    2001, with xi^2 = q xi_K^2): alpha_u = q u J_0 and alpha_v = q v J_1,

        J_n = integral over w from 0 to 1 of kappa(xi(w)) / d^(n + 1/2),
        d = 1 - (1 - q^2) w,   xi(w)^2 = q w (u^2 + v^2 / d);

    0 at its centre. The pixels are integrated in chunks, to keep the
    memory of the nodes at each pixel small.
    """
    shape = u.shape
    u, v = u.ravel(), v.ravel()
    d = q * q + (1 - q * q) * _ONE_MINUS_NODES
    alpha_u = np.empty_like(u)
    alpha_v = np.empty_like(v)
    for start in range(0, u.size, chunk):
        part = slice(start, start + chunk)
        u_part, v_part = u[part, None], v[part, None]
        xi = np.sqrt(q * _NODES * (u_part * u_part + v_part * v_part / d))
        with np.errstate(divide='ignore', invalid='ignore'):
            weighted = _WEIGHTS * kappa(xi) / np.sqrt(d)
        alpha_u[part] = q * u[part] * weighted.sum(-1)
        alpha_v[part] = q * v[part] * (weighted / d).sum(-1)
    centre = (u == 0) & (v == 0)
    alpha_u[centre] = 0
    alpha_v[centre] = 0
    return alpha_u.reshape(shape), alpha_v.reshape(shape)
