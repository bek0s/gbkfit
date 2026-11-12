"""
The masses of a gravitational lens (see Lens): each deflects the light
of the source plane, and the deflections of the masses of a lens add up.

The analytic masses are placed by their centre (x, y; arcsec, in the
frame of the model, as xpos and ypos: the frame of each observation moves
them too, see ObservationGroup) and oriented by the position angle
(pa; degrees, north through east, as that of disks) of their major axis
(u, from +y towards -x; the minor axis v is 90 degrees clockwise of it).
Their convergence is elliptical in xi = sqrt(q u^2 + v^2 / q), so the
mass within the ellipse xi of an axis ratio q is that of the circle of
radius xi of the round mass.
"""

import abc
import math
import os.path
from collections.abc import Sequence
from typing import Annotated, Any, TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.ndimage
from annotated_types import Ge

from gbkfit.config import (
    DataFile, TypedParser, TypedSerializable, Unit, check_name,
    check_options)
from gbkfit.grid import (
    Grid, check_overlap, make_grid, pixels_on, sky_positions)
from gbkfit.io import read_data, write_data
from gbkfit.params import ParamScalarDesc, ParamValues

from ._deflections import (
    elliptical_deflection, epl_deflection, nfw_convergence,
    nfw_round_deflection, sie_deflection)


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


# The rotation of a grid on the sky
_Angle: TypeAlias = Annotated[float, Unit('deg')]


class LensMass(TypedSerializable, abc.ABC):
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
    ValueError
        If the name is not valid (see check_name).
    """

    def dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {} if self._name is None else dict(name=self._name)

    def __init__(self, name: str | None = None):
        check_options(LensMass.__init__, locals())
        check_name(name)
        self._name = name

    def name(self) -> str | None:
        """Return its name, if any."""
        return self._name

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        """Return its parameters, by name (none by default)."""
        return {}

    def centred(self) -> bool:
        """
        Check whether it is placed by its centre, its parameters x and y
        (in the frame of the model, as xpos and ypos); not here.
        """
        return False

    @abc.abstractmethod
    def deflection(
            self, grid: Grid, params: ParamValues
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
        The world coordinates of the maps (see make_grid).
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
    ValueError
        If the maps are not two finite images of one shape.
    """

    config_constructors = ('from_files',)

    @staticmethod
    def type() -> str:
        return 'deflection_map'

    @staticmethod
    def type_title() -> str:
        return 'Deflection maps'

    @classmethod
    def from_files(
            cls,
            alpha_x: DataFile,
            alpha_y: DataFile,
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: _Angle | None = None,
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
            The world coordinates of the maps (see make_grid); by
            default, those of the headers. Either rpix or rval can be
            given, and the other comes from the headers (see
            read_data).
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
        ValueError
            If the maps have different world coordinates.
        """
        check_options(LensDeflectionMap.from_files, locals())
        maps = [
            read_data(file, rpix=rpix, rval=rval)
            for file in (alpha_x, alpha_y)]
        (data_x, coords), (data_y, coords_y) = maps
        if coords_y != coords:
            raise ValueError(
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
            write_data(
                filename, data, self._grid.coords, overwrite=overwrite)
            info[key] = filename if dump_path else os.path.basename(filename)
        if self._scaled:
            info['scaled'] = True
        return info

    def __init__(
            self,
            alpha_x: Annotated[npt.ArrayLike, Unit('arcsec')],
            alpha_y: Annotated[npt.ArrayLike, Unit('arcsec')],
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: _Angle | None = None,
            scaled: bool = False,
            name: str | None = None
    ):
        check_options(LensDeflectionMap.__init__, locals())
        super().__init__(name)
        alpha_x = np.asarray(alpha_x, dtype=float)
        alpha_y = np.asarray(alpha_y, dtype=float)
        if alpha_x.ndim != 2 or alpha_x.shape != alpha_y.shape:
            raise ValueError(
                f"the deflection maps must be two images of one shape; they "
                f"have the shapes {alpha_x.shape} and {alpha_y.shape}")
        if not (np.all(np.isfinite(alpha_x)) and np.all(np.isfinite(alpha_y))):
            raise ValueError("the deflection maps must be finite")
        self._alpha_x = alpha_x
        self._alpha_y = alpha_y
        self._grid = make_grid(
            alpha_x.shape[::-1], step, rpix, rval, rota)
        self._scaled = scaled

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        if not self._scaled:
            return {}
        return dict(scale=ParamScalarDesc(
            "the scale of the deflection (e.g. D_LS / D_S)",
            title='Deflection scale', min=0, unit='', default=0.5))

    def deflection(
            self, grid: Grid, params: ParamValues
    ) -> tuple[np.ndarray, np.ndarray]:
        check_overlap(grid, self._grid, "the deflection maps")
        pixel_x, pixel_y = pixels_on(grid, self._grid)
        scale = params['scale'] if self._scaled else 1.0

        def sample(data: np.ndarray) -> np.ndarray:
            return scale * scipy.ndimage.map_coordinates(
                data, [pixel_y, pixel_x], order=1, mode='nearest')
        return sample(self._alpha_x), sample(self._alpha_y)


class LensSIE(LensMass):
    """
    A singular isothermal ellipsoid (SIE): the standard model of a galaxy
    lens.

    Its convergence is

        kappa = theta_e / (2 xi),

    and its deflection is analytic (Kormann et al. 1994): with
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

    @staticmethod
    def type_title() -> str:
        return 'Singular isothermal ellipsoid (SIE)'

    def centred(self) -> bool:
        return True

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(theta_e=ParamScalarDesc(
            "the Einstein radius", title='Einstein radius', min=0,
            unit='arcsec', default=1)
        ) | _ellipse_pdescs()

    def deflection(
            self, grid: Grid, params: ParamValues
    ) -> tuple[np.ndarray, np.ndarray]:
        u, v = _on_axes(grid, params)
        alpha_u, alpha_v = sie_deflection(
            u, v, params['theta_e'], params['q'])
        return _from_axes(alpha_u, alpha_v, params['pa'])


class LensEPL(LensMass):
    """
    An elliptical power law (EPL): a galaxy lens whose density falls as
    r^-slope (an SIE for slope 2).

    Its convergence is

        kappa = (2 - t) / 2 * (theta_e / xi)^t,   t = slope - 1,

    and its deflection is analytic (Tessore & Metcalf 2015): with
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

    @staticmethod
    def type_title() -> str:
        return 'Elliptical power law (EPL)'

    def centred(self) -> bool:
        return True

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(
            theta_e=ParamScalarDesc(
                "the Einstein radius", title='Einstein radius', min=0,
                unit='arcsec', default=1),
            slope=ParamScalarDesc(
                "the slope of the density (2 = isothermal)",
                title='Density slope', min=1, max=3, unit='', default=2)
        ) | _ellipse_pdescs()

    def deflection(
            self, grid: Grid, params: ParamValues
    ) -> tuple[np.ndarray, np.ndarray]:
        u, v = _on_axes(grid, params)
        alpha_u, alpha_v = epl_deflection(
            u, v, params['theta_e'], params['slope'], params['q'])
        return _from_axes(alpha_u, alpha_v, params['pa'])


class LensNFW(LensMass):
    """
    A Navarro-Frenk-White (NFW) halo: the lens of a dark matter halo (e.g.
    of a group or a cluster).

    Its convergence is (Bartelmann 1996)

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

    @staticmethod
    def type_title() -> str:
        return 'NFW halo'

    def centred(self) -> bool:
        return True

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(
            kappa_s=ParamScalarDesc(
                "the convergence scale (rho_s r_s / Sigma_cr)",
                title='Convergence scale', min=0, unit='', default=0.1),
            r_s=ParamScalarDesc(
                "the scale radius", title='Scale radius', min=0,
                unit='arcsec', default=10)
        ) | _ellipse_pdescs()

    def deflection(
            self, grid: Grid, params: ParamValues
    ) -> tuple[np.ndarray, np.ndarray]:
        u, v = _on_axes(grid, params)
        kappa_s, r_s, q = params['kappa_s'], params['r_s'], params['q']
        if q == 1:
            alpha_u, alpha_v = nfw_round_deflection(u, v, kappa_s, r_s)
        else:
            alpha_u, alpha_v = elliptical_deflection(
                u, v, q, lambda xi: kappa_s * nfw_convergence(xi / r_s))
        return _from_axes(alpha_u, alpha_v, params['pa'])


class LensMultipole(LensMass):
    """
    A multipole of a lens: the deviation of a galaxy lens from an
    ellipse (e.g. order 4 for disky or boxy isodensity contours).

    Orders 3 and 1 make the contours lopsided. Its convergence is (e.g.
    Van de Vyvere et al. 2022)

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
    strength), pa (degrees, of a period of 360 / m: the convergence
    repeats every 360 / m degrees), x and y (see the module).

    Parameters
    ----------
    order : int
        The order m (at least 1).
    name : str, optional
        Its name (see LensMass).

    Raises
    ------
    ValueError
        If the order is not an integer of at least 1.
    """

    @staticmethod
    def type() -> str:
        return 'multipole'

    @staticmethod
    def type_title() -> str:
        return 'Multipole'

    def dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return super().dump() | dict(order=self._order)

    def __init__(self, order: Annotated[int, Ge(1)], name: str | None = None):
        check_options(LensMultipole.__init__, locals())
        super().__init__(name)
        self._order = order

    def order(self) -> int:
        """Return its order."""
        return self._order

    def centred(self) -> bool:
        return True

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(
            theta_e=ParamScalarDesc(
                "the Einstein radius of its lens", title='Einstein radius',
                min=0, unit='arcsec', default=1),
            a=ParamScalarDesc(
                "the strength", title='Strength', unit='', default=0),
            pa=ParamScalarDesc(
                "the position angle", title='Position angle', unit='deg',
                default=0, period=360 / self._order),
        ) | _centre_pdescs()

    def deflection(
            self, grid: Grid, params: ParamValues
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

    @staticmethod
    def type_title() -> str:
        return 'External shear'

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(
            gamma=ParamScalarDesc(
                "the strength", title='Shear strength', min=0, unit='',
                default=0),
            pa=ParamScalarDesc(
                "the position angle", title='Position angle', unit='deg',
                default=0, period=180))

    def deflection(
            self, grid: Grid, params: ParamValues
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = sky_positions(grid)
        # (the angle from +x of the position angle is 90 degrees more)
        angle = 2 * math.radians(params['pa'] + 90)
        gamma1 = params['gamma'] * math.cos(angle)
        gamma2 = params['gamma'] * math.sin(angle)
        return gamma1 * x + gamma2 * y, gamma2 * x - gamma1 * y


lens_mass_parser = TypedParser(LensMass, [
    LensDeflectionMap, LensEPL, LensMultipole, LensNFW, LensSIE, LensShear],
    family='lens_mass')


def _centre_pdescs() -> dict[str, ParamScalarDesc]:
    """Return the parameters of the centre of a mass."""
    return dict(
        x=ParamScalarDesc(
            "the centre along x", title='Centre (x)', unit='arcsec',
            default=0),
        y=ParamScalarDesc(
            "the centre along y", title='Centre (y)', unit='arcsec',
            default=0))


def _ellipse_pdescs() -> dict[str, ParamScalarDesc]:
    """Return the parameters of the shape of an elliptical mass."""
    return dict(
        # (q > 0 keeps the deflection finite)
        q=ParamScalarDesc(
            "the axis ratio", title='Axis ratio', min=1e-3, max=1, unit='',
            default=0.8),
        pa=ParamScalarDesc(
            "the position angle", title='Position angle', unit='deg',
            default=0, period=180),
    ) | _centre_pdescs()


def _centred(
        grid: Grid, params: ParamValues
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the positions of the pixels of a grid from the centre of a
    mass, along x and y.
    """
    x, y = sky_positions(grid)
    return x - params['x'], y - params['y']


def _on_axes(
        grid: Grid, params: ParamValues
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
