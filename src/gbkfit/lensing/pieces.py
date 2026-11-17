"""
The pieces of a lens (see Lens): each deflects the light that passes
the plane of its lens, and the deflections of the pieces of a lens add
up.

Positions and deflections are on the sky (see SkyPoints): arcsec, +x to
the west and +y to the north, from the origin of the frame of the
parameters of the galaxies (as their xpos and ypos), which is the same
for every observation of them.

The strength of each piece is that of the light of a source at the
redshift that its normalisation says (see Normalisation): quoted for the
source_redshift of its lens (SIE, EPL, NFW, shear and sheet, and
deflection maps made for one source), or at D_ls / D_s = 1 (deflection
maps of that convention); its lens converts the first to the second
(see Lens.deflection).

The shapes (SIE, EPL and NFW) are placed by their centre (xpos, ypos)
and oriented by the position angle (posa; degrees, north through east,
as that of disks) of their major axis (u, from +y towards -x; the minor
axis v is 90 degrees clockwise of it). Their convergence is elliptical
in xi = sqrt(q u^2 + v^2 / q), so the mass within the ellipse xi of an
axis ratio q is that of the circle of radius xi of the round mass.
Shear and sheets have no centre: their deflection is 0 at the origin of
the frame, and any other centre adds a constant deflection, which moves
the light of every source behind as their positions do. Deflection maps
are placed by their world coordinates (RA and Dec).
"""

import abc
import enum
import math
import os.path
from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING, Annotated, Any, Literal, NamedTuple, Protocol, TypeAlias,
    runtime_checkable)

import numpy as np
import numpy.typing as npt
import scipy.ndimage
from annotated_types import Ge

from gbkfit.config import (
    Choices, DataFile, TypedSerializable, Unit, check_name,
    check_options, split_data_file)
from gbkfit.grid import (
    Coords, Grid, frame_to_pixels, make_grid, sky_to_pixel)
from gbkfit.io import (
    Orientation, coords_from_header, model_header, read_image, write_data)
from gbkfit.model.mass import LogTable
from gbkfit.params import ModelEvaluationError, ParamScalarDesc
from ._deflections import (
    epl_deflection, multipole_deflection, nfw_convergence,
    nfw_round_deflection, sie_deflection)
from ._layers import SheetFrame, elliptical_convergence

if TYPE_CHECKING:
    from gbkfit.driver import Driver
    from gbkfit.model import Need


__all__ = [
    'Deflects',
    'GalaxyBending',
    'LensDeflectionMap',
    'LensEPL',
    'LensNFW',
    'LensPiece',
    'LensSIE',
    'LensShear',
    'LensSheet',
    'Normalisation',
    'OfGalaxy',
    'SkyPoints'
]


# The rotation of a grid on the sky
_Angle: TypeAlias = Annotated[float, Unit('deg')]

# The units of the values of the files of deflection maps
_MapUnit: TypeAlias = Annotated[
    Literal['arcsec', 'pixel'], Choices(
        arcsec='Arcseconds', pixel='Pixels of the maps')]

# The orders of the multipoles of a shape
_Orders: TypeAlias = Sequence[Annotated[int, Ge(1)]]

# The table of the convergence of an elliptical NFW halo (see LensNFW):
# from this share of its scale radius (or closer, of the points), at
# this step of ln xi (64 per decade)
_NFW_TABLE_START = 1e-8
_NFW_TABLE_STEP = math.log(10) / 64


class Normalisation(enum.Enum):
    """
    The source whose light the strength of a piece of a lens is that of
    (see LensPiece):

    - QUOTED: a source at the source_redshift of its lens (e.g. the
      Einstein radius of an SIE, as lens models quote it, or deflection
      maps made for one source, as glafic and Lenstool make them);
    - UNIT: a source of D_ls / D_s = 1, the deflection angle itself
      (alpha-hat; e.g. the deflection maps of the Hubble Frontier
      Fields).
    """
    QUOTED = 'quoted'
    UNIT = 'unit'


class SkyPoints(NamedTuple):
    """
    Points on the sky: their positions (arcsec, +x to the west and +y to
    the north, as xpos and ypos) in a frame, and the RA and Dec (degrees)
    of the origin of the frame, which places them among world
    coordinates (e.g. those of deflection maps; see frame_to_pixels).
    """
    x: np.ndarray
    y: np.ndarray
    origin: tuple[float, float]


class LensPiece(TypedSerializable, abc.ABC):
    """
    A piece of a lens (see Lens): it deflects the light at points of the
    plane of its lens, of a strength for the source that its
    normalisation says (see normalisation): by its parameters (see
    Deflects), or by the mass of a galaxy (see OfGalaxy).

    A piece may have parameters (see pdescs), named in its lens after it
    (see Lens.pdescs).

    Parameters
    ----------
    name : str, optional
        Its name, which prefixes its parameters when its lens has several
        pieces; its type by default (e.g. 'shear').

    Raises
    ------
    ValueError
        If the name is not valid (see check_name).
    """

    config_titles = dict(name='Name')

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return {} if self._name is None else dict(name=self._name)

    def __init__(self, name: str | None = None):
        check_options(LensPiece.__init__, locals())
        check_name(name)
        self._name = name

    def name(self) -> str:
        """Return its name: that given, or else its type."""
        return self.type() if self._name is None else self._name

    @abc.abstractmethod
    def normalisation(self) -> Normalisation:
        """
        Return the source whose light its strength is that of (see
        Normalisation).
        """

    @abc.abstractmethod
    def pdescs(self) -> dict[str, ParamScalarDesc]:
        """Return its parameters, by name."""


@runtime_checkable
class Deflects(Protocol):
    """
    A piece of a lens that deflects the light by its parameters (e.g. a
    shape, a shear, a sheet or deflection maps).
    """

    def deflection(
            self,
            points: SkyPoints,
            params: Mapping[str, float],
            driver: 'Driver'
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return its deflection at points of the plane of its lens.

        Parameters
        ----------
        points : SkyPoints
            The points.
        params : Mapping of str to float
            The values of its parameters, by its own names (see
            LensPiece.pdescs).
        driver : Driver
            The driver whose native kernels compute it, if it has any (the
            points and the deflection on the host).

        Returns
        -------
        tuple of np.ndarray
            The deflection (arcsec) along x and y, of the shape of the
            points.

        Raises
        ------
        ModelEvaluationError
            If it has no deflection there (e.g. points off a map).
        """
        ...


@runtime_checkable
class OfGalaxy(Protocol):
    """
    A piece of a lens that is the mass of a galaxy (e.g. LensGalaxyMass):
    it deflects as that mass, laid out on the sky by the galaxy in each
    evaluation, which those who evaluate its lens give it (see
    GalaxyBending and Lens.deflection) and the galaxy must have (see
    needs).
    """

    def galaxy(self) -> str:
        """Return the name of its galaxy."""

    def needs(self, label: str) -> tuple['Need', ...]:
        """
        Return what it needs of its galaxy (see Galaxy.check_needs), as
        the piece of the given label in messages (e.g. "the piece 'mass'
        of lens 'lens'").
        """


class GalaxyBending(Protocol):
    """
    The bending of the light by the mass of a galaxy in an evaluation (of
    a piece that is that mass, see OfGalaxy): e.g. the mass on the sky,
    with where the deflections of its layers are kept (see
    MassDeflections).
    """

    def deflection(self, points: SkyPoints) -> tuple[np.ndarray, np.ndarray]:
        """
        Return its deflection at points of the plane of the galaxy, at
        D_ls / D_s = 1 (alpha-hat, see Normalisation): along x and y
        (arcsec), of the shape of the points.
        """
        ...


class _Multipoles:
    """
    The circular multipoles of a shape, of its centre, Einstein radius
    and radial slope (see multipole_deflection): the parameters of the
    multipole of each order m, its strength 'a{m}' and its position angle
    'posa{m}' (degrees, of a period of 360 / m).

    Parameters
    ----------
    orders : Sequence of int
        The orders (at least 1, without repeats).

    Raises
    ------
    ValueError
        If an order repeats.
    """

    def __init__(self, orders: Sequence[int]):
        orders = tuple(orders)
        if len(set(orders)) != len(orders):
            raise ValueError(f"the orders of multipoles repeat: {orders}")
        self._orders = orders

    def orders(self) -> tuple[int, ...]:
        """Return the orders."""
        return self._orders

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        """Return their parameters, by name."""
        pdescs = {}
        for m in self._orders:
            pdescs[f'a{m}'] = ParamScalarDesc(
                f"the strength of the multipole of order {m}",
                title=f'Multipole {m} strength', unit='', default=0)
            pdescs[f'posa{m}'] = ParamScalarDesc(
                f"the position angle of the multipole of order {m}",
                title=f'Multipole {m} position angle', unit='deg',
                default=0, period=360 / m)
        return pdescs

    def deflection(
            self,
            x: np.ndarray,
            y: np.ndarray,
            theta_e: float,
            t: float,
            params: Mapping[str, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return their deflection along x and y at positions x and y from
        the centre of their shape, of its Einstein radius and the radial
        slope t of its convergence.
        """
        alpha_x = np.zeros_like(x)
        alpha_y = np.zeros_like(y)
        for m in self._orders:
            mx, my = multipole_deflection(
                x, y, theta_e, t, m, params[f'a{m}'], params[f'posa{m}'])
            alpha_x += mx
            alpha_y += my
        return alpha_x, alpha_y


class LensSIE(LensPiece):
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

    Its multipoles of the given orders deviate it from an ellipse (e.g.
    order 4 for disky or boxy isodensity contours; see
    multipole_deflection, of the radial slope 1 of its convergence),
    about its centre and of its Einstein radius.

    Its parameters (see pdescs): theta_e (arcsec, the Einstein radius of
    the sphere of the same mass), q (the axis ratio), posa, xpos and ypos
    (see the module), and a{m} and posa{m} of each multipole of order m.
    Its strength is quoted (see Normalisation).

    Parameters
    ----------
    multipoles : Sequence of int, optional
        The orders of its multipoles (at least 1); none by default.
    name : str, optional
        Its name (see LensPiece).

    Raises
    ------
    ValueError
        If the orders of the multipoles repeat.
    """

    config_titles = dict(multipoles='Multipoles')

    @staticmethod
    def type() -> str:
        return 'sie'

    @staticmethod
    def type_title() -> str:
        return 'Singular isothermal ellipsoid (SIE)'

    def dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return super().dump() | _dump_multipoles(self._multipoles)

    def __init__(self, multipoles: _Orders = (), name: str | None = None):
        check_options(LensSIE.__init__, locals())
        super().__init__(name)
        self._multipoles = _Multipoles(multipoles)

    def multipoles(self) -> tuple[int, ...]:
        """Return the orders of its multipoles."""
        return self._multipoles.orders()

    def normalisation(self) -> Normalisation:
        return Normalisation.QUOTED

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(theta_e=_theta_e_pdesc()) | _ellipse_pdescs() \
            | self._multipoles.pdescs()

    def deflection(
            self,
            points: SkyPoints,
            params: Mapping[str, float],
            driver: 'Driver'
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = _centred(points, params)
        u, v = _on_axes(x, y, params['posa'])
        alpha_u, alpha_v = sie_deflection(
            u, v, params['theta_e'], params['q'])
        alpha_x, alpha_y = _from_axes(alpha_u, alpha_v, params['posa'])
        mx, my = self._multipoles.deflection(
            x, y, params['theta_e'], 1, params)
        return alpha_x + mx, alpha_y + my


class LensEPL(LensPiece):
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

    Its multipoles of the given orders deviate it from an ellipse (see
    multipole_deflection), about its centre and of its Einstein radius
    and the radial slope t of its convergence (those of an SIE for
    slope 2).

    Its parameters (see pdescs): theta_e (arcsec, the Einstein radius of
    the round mass of the same profile), slope (of the density; 1 to 3),
    q (the axis ratio), posa, xpos and ypos (see the module), and a{m}
    and posa{m} of each multipole of order m. Its strength is quoted (see
    Normalisation).

    Parameters
    ----------
    multipoles : Sequence of int, optional
        The orders of its multipoles (at least 1); none by default.
    name : str, optional
        Its name (see LensPiece).

    Raises
    ------
    ValueError
        If the orders of the multipoles repeat.
    """

    config_titles = dict(multipoles='Multipoles')

    @staticmethod
    def type() -> str:
        return 'epl'

    @staticmethod
    def type_title() -> str:
        return 'Elliptical power law (EPL)'

    def dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return super().dump() | _dump_multipoles(self._multipoles)

    def __init__(self, multipoles: _Orders = (), name: str | None = None):
        check_options(LensEPL.__init__, locals())
        super().__init__(name)
        self._multipoles = _Multipoles(multipoles)

    def multipoles(self) -> tuple[int, ...]:
        """Return the orders of its multipoles."""
        return self._multipoles.orders()

    def normalisation(self) -> Normalisation:
        return Normalisation.QUOTED

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(
            theta_e=_theta_e_pdesc(),
            slope=ParamScalarDesc(
                "the slope of the density (2 = isothermal)",
                title='Density slope', min=1, max=3, unit='', default=2)
        ) | _ellipse_pdescs() | self._multipoles.pdescs()

    def deflection(
            self,
            points: SkyPoints,
            params: Mapping[str, float],
            driver: 'Driver'
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = _centred(points, params)
        u, v = _on_axes(x, y, params['posa'])
        alpha_u, alpha_v = epl_deflection(
            u, v, params['theta_e'], params['slope'], params['q'])
        alpha_x, alpha_y = _from_axes(alpha_u, alpha_v, params['posa'])
        mx, my = self._multipoles.deflection(
            x, y, params['theta_e'], params['slope'] - 1, params)
        return alpha_x + mx, alpha_y + my


class LensNFW(LensPiece):
    """
    A Navarro-Frenk-White (NFW) halo: the lens of a dark matter halo (e.g.
    of a group or a cluster).

    Its convergence is (Bartelmann 1996)

        kappa = 2 kappa_s (1 - F(x)) / (x^2 - 1),   x = xi / r_s,

    with F(x) = arccosh(1 / x) / sqrt(1 - x^2) (arccos and
    sqrt(x^2 - 1) for x > 1), and kappa_s = rho_s r_s / Sigma_cr.

    Round (q = 1), its deflection is analytic: 4 kappa_s r_s h(x) / x
    along the radius, with h(x) = ln(x / 2) + F(x). Elliptical, it is the
    integral over the ellipses within each point, by the native kernels of
    the driver (see _layers.elliptical_convergence), of its convergence in
    a table cubic in ln xi, 64 points per decade (to about 1e-7). It is 0
    at its centre.

    Its parameters (see pdescs): kappa_s, r_s (arcsec, the scale radius),
    q (the axis ratio), posa, xpos and ypos (see the module). Its
    strength is quoted (see Normalisation): Sigma_cr is that of the
    source_redshift of its lens.

    Parameters
    ----------
    name : str, optional
        Its name (see LensPiece).
    """

    @staticmethod
    def type() -> str:
        return 'nfw'

    @staticmethod
    def type_title() -> str:
        return 'NFW halo'

    def normalisation(self) -> Normalisation:
        return Normalisation.QUOTED

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
            self,
            points: SkyPoints,
            params: Mapping[str, float],
            driver: 'Driver'
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = _centred(points, params)
        kappa_s, r_s, q = params['kappa_s'], params['r_s'], params['q']
        if q == 1:
            u, v = _on_axes(x, y, params['posa'])
            alpha_u, alpha_v = nfw_round_deflection(u, v, kappa_s, r_s)
            return _from_axes(alpha_u, alpha_v, params['posa'])
        # Elliptical: its convergence as a sheet of kappa(xi), xi = sqrt(q)
        # a, stretched by 1 / sqrt(q) and sqrt(q), tabulated to the
        # farthest point (the ellipses beyond a point do not bend it)
        u, v = _on_axes(x, y, params['posa'])
        xi = np.sqrt(q * u * u + v * v / q)
        start = math.log(_NFW_TABLE_START * r_s)
        if (inside := xi[xi > 0]).size:
            start = min(start, math.log(inside.min() / 2))
        end = math.log(max(np.max(xi, initial=r_s), r_s) * 2)
        radii = np.exp(start + _NFW_TABLE_STEP * np.arange(
            max(4, math.ceil((end - start) / _NFW_TABLE_STEP) + 1)))
        frame = SheetFrame(
            (0.0, 0.0), params['posa'], (1 / math.sqrt(q), math.sqrt(q)),
            np.zeros(1), (0.0, 0.0))
        kappa = LogTable(start, _NFW_TABLE_STEP, np.log(
            kappa_s * nfw_convergence(radii / r_s)))
        return elliptical_convergence(frame, kappa, x, y, driver)


class LensShear(LensPiece):
    """
    An external shear: the tidal stretch of the masses near a lens, of
    strength gamma along the position angle posa (degrees, north through
    east). Its deflection is linear, 0 at the origin of the frame (see
    the module).

    Its parameters (see pdescs): gamma and posa. Its strength is quoted
    (see Normalisation).

    Parameters
    ----------
    name : str, optional
        Its name (see LensPiece).
    """

    @staticmethod
    def type() -> str:
        return 'shear'

    @staticmethod
    def type_title() -> str:
        return 'External shear'

    def normalisation(self) -> Normalisation:
        return Normalisation.QUOTED

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(
            gamma=ParamScalarDesc(
                "the strength", title='Shear strength', min=0, unit='',
                default=0),
            posa=ParamScalarDesc(
                "the position angle", title='Position angle', unit='deg',
                default=0, period=180))

    def deflection(
            self,
            points: SkyPoints,
            params: Mapping[str, float],
            driver: 'Driver'
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = points.x, points.y
        # (the angle from +x of the position angle is 90 degrees more)
        angle = 2 * math.radians(params['posa'] + 90)
        gamma1 = params['gamma'] * math.cos(angle)
        gamma2 = params['gamma'] * math.sin(angle)
        return gamma1 * x + gamma2 * y, gamma2 * x - gamma1 * y


class LensSheet(LensPiece):
    """
    A mass sheet: a uniform convergence kappa (e.g. that of the masses
    along the line of sight; negative, less than the mean). Its
    deflection is kappa times the position, 0 at the origin of the frame
    (see the module).

    Its parameter (see pdescs): kappa. Its strength is quoted (see
    Normalisation).

    Parameters
    ----------
    name : str, optional
        Its name (see LensPiece).
    """

    @staticmethod
    def type() -> str:
        return 'sheet'

    @staticmethod
    def type_title() -> str:
        return 'Mass sheet'

    def normalisation(self) -> Normalisation:
        return Normalisation.QUOTED

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        return dict(kappa=ParamScalarDesc(
            "the convergence", title='Convergence', unit='', default=0))

    def deflection(
            self,
            points: SkyPoints,
            params: Mapping[str, float],
            driver: 'Driver'
    ) -> tuple[np.ndarray, np.ndarray]:
        kappa = params['kappa']
        return kappa * points.x, kappa * points.y


class LensDeflectionMap(LensPiece):
    """
    A deflection given as maps of it (e.g. from a lens model of a
    cluster), placed by their world coordinates (RA and Dec): at
    D_ls / D_s = 1 (the deflection angle, as the models of the Hubble
    Frontier Fields publish it), or quoted for the source_redshift of
    its lens (made for one source, as glafic and Lenstool make them; see
    Normalisation).

    Between the pixels of the maps the deflection is interpolated
    bilinearly (it stays within the values of the maps, however noisy
    they are). The maps cover the span of the centres of their pixels:
    their deflection at points off them is an error, never a guess.

    The maps are in the gnomonic projection about their reference (their
    world coordinates), and the points they deflect in that of the frame
    of the points: the two differ, by the convergence of the meridians
    between the two references (about their separation times tan(Dec):
    e.g. 3e-3 rad for 2 arcmin at Dec 80 deg, 0.06 arcsec of a deflection
    of 20 arcsec) and more slightly by the distortions of the
    projections. A point is moved into the projection of the maps, the
    deflection there gives the point of the source in it, and that point
    is moved back: the deflection of the frame is the point less it,
    exactly.

    Shifted, the maps are moved from where their world coordinates put
    them by parameters, xpos and ypos (arcsec, along x and y; fixed or
    fitted, see pdescs): e.g. maps of a lens model made on an image whose
    astrometry is off from that of the data. Like everything else on the
    sky, they move with the offsets of each observation (see Tracer).

    Parameters
    ----------
    alpha_x, alpha_y : np.ndarray
        The deflection (arcsec) along x and y (as xpos and ypos), each of
        shape (ny, nx), whatever the rotation of the maps (files give it
        along their own axes, see from_files).
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the maps (see make_grid).
    rota : float, optional
        The rotation of the maps on the sky.
    shifted : bool, optional
        Whether the maps are shifted (see above).
    quoted : bool, optional
        Whether the maps are quoted for the source_redshift of the lens;
        by default, they are at D_ls / D_s = 1.
    name : str, optional
        Its name (see LensPiece).

    Raises
    ------
    ValueError
        If the maps are not two finite images of one shape, of at least
        two pixels along each axis.
    """

    config_constructors = ('from_files',)

    config_titles = dict(
        alpha_x='Deflection along x', alpha_y='Deflection along y',
        unit='Unit', step='Pixel size', rpix='Reference pixel',
        rval='Reference position', rota='Rotation', shifted='Shifted',
        quoted='Quoted for the source redshift')

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
            unit: _MapUnit = 'arcsec',
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: _Angle | None = None,
            shifted: bool = False,
            quoted: bool = False,
            name: str | None = None
    ) -> 'LensDeflectionMap':
        """
        Read a piece from files of its deflection maps, with the world
        coordinates of the files unless given; they must agree.

        Each file gives the deflection along one of its own pixel axes,
        towards higher pixels, as lens models write it (e.g. those of the
        Hubble Frontier Fields, of Lenstool or glafic): alpha_x along the
        first axis of its file (that of NAXIS1), and alpha_y along the
        second axis of its file, in arcsec or in pixels of the maps
        (unit). The maps are read in the orientation of the model (see
        read_data), and their vectors with them: a component along an
        axis that is reversed (east to the right of north) is negated,
        the components are swapped with the axes (Dec first), and the
        vectors are rotated by the rotation of the maps on the sky (rota)
        into x and y.

        Parameters
        ----------
        alpha_x, alpha_y : str or tuple
            The files of the maps: each a filename, or a filename and the
            HDU.
        unit : str, optional
            The unit of their values: 'arcsec' (the default) or 'pixel'
            (of the maps, along each of their axes).
        step, rpix, rval : float or Sequence of float, optional
            The world coordinates of the maps (see make_grid); by
            default, those of the headers. Either rpix or rval can be
            given, and the other comes from the headers (see
            read_data).
        rota : float, optional
            The rotation of the maps on the sky, which turns their
            vectors with them; by default, that of the headers.
        shifted, quoted : bool, optional
            Whether the maps are shifted, and whether they are quoted for
            the source_redshift of the lens (see the class).
        name : str, optional
            Its name (see LensPiece).

        Returns
        -------
        LensDeflectionMap
            The piece.

        Raises
        ------
        ValueError
            If the maps have different world coordinates.
        """
        check_options(LensDeflectionMap.from_files, locals())
        (data_x, coords, orientation_x), (data_y, coords_y, orientation_y) = (
            _read_map(file, rpix, rval) for file in (alpha_x, alpha_y))
        if coords_y != coords:
            raise ValueError(
                "the deflection maps have different world coordinates")
        step = coords.step if step is None else step
        # (the size of the pixels along each axis of the model, which
        # turns deflections in pixels into arcsec)
        scales = np.abs(np.broadcast_to(step, (2,))) if unit == 'pixel' \
            else np.ones(2)
        along = _along_model_axes(
            (data_x, data_y), (orientation_x, orientation_y), scales)
        rota = coords.rota if rota is None else rota
        return cls(
            *_rotated(*along, rota), step, coords.rpix, coords.rval, rota,
            shifted, quoted, name)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        info = super().dump()
        # (the files give the deflection along their pixel axes, which are
        # those of the maps, turned by their rotation; see from_files)
        along = _rotated(
            self._alpha_x, self._alpha_y, -self._grid.coords.rota)
        for key, data in zip(('alpha_x', 'alpha_y'), along):
            filename = f'{prefix}{key}.fits'
            write_data(
                filename, data, self._grid.coords, overwrite=overwrite)
            info[key] = filename if dump_path else os.path.basename(filename)
        if self._shifted:
            info['shifted'] = True
        if self._quoted:
            info['quoted'] = True
        return info

    def __init__(
            self,
            alpha_x: Annotated[npt.ArrayLike, Unit('arcsec')],
            alpha_y: Annotated[npt.ArrayLike, Unit('arcsec')],
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: _Angle | None = None,
            shifted: bool = False,
            quoted: bool = False,
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
        if min(alpha_x.shape) < 2:
            raise ValueError(
                f"the deflection maps need at least two pixels along each "
                f"axis, between which their deflection is interpolated; "
                f"they have the shape {alpha_x.shape}")
        if not (np.all(np.isfinite(alpha_x)) and np.all(np.isfinite(alpha_y))):
            raise ValueError("the deflection maps must be finite")
        self._alpha_x = alpha_x
        self._alpha_y = alpha_y
        self._grid = make_grid(
            alpha_x.shape[::-1], step, rpix, rval, rota)
        # (the projection of the maps, by its positions in arcsec, and
        # their pixels in it)
        self._projection = _projection(self._grid.coords.rval[:2])
        self._matrix, self._offset = sky_to_pixel(self._grid)
        self._shifted = shifted
        self._quoted = quoted

    def normalisation(self) -> Normalisation:
        return Normalisation.QUOTED if self._quoted else Normalisation.UNIT

    def grid(self) -> Grid:
        """Return the grid of the maps."""
        return self._grid

    def shifted(self) -> bool:
        """Return whether the maps are shifted (see the class)."""
        return self._shifted

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        if not self._shifted:
            return {}
        return dict(
            xpos=ParamScalarDesc(
                "the shift of the maps along x", title='Shift (x)',
                unit='arcsec', default=0),
            ypos=ParamScalarDesc(
                "the shift of the maps along y", title='Shift (y)',
                unit='arcsec', default=0))

    def deflection(
            self,
            points: SkyPoints,
            params: Mapping[str, float],
            driver: 'Driver'
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = points.x, points.y
        if self._shifted:
            # The deflection of shifted maps at a point is theirs at the
            # point less the shift
            x = x - params['xpos']
            y = y - params['ypos']
        # The points in the projection of the maps, where their deflection
        # gives the points of the source, moved back into the frame
        map_x, map_y = frame_to_pixels(x, y, points.origin, self._projection)
        matrix, offset = self._matrix, self._offset
        pixel_x = matrix[0, 0] * map_x + matrix[0, 1] * map_y + offset[0]
        pixel_y = matrix[1, 0] * map_x + matrix[1, 1] * map_y + offset[1]
        self._check_covered(points, pixel_x, pixel_y)
        alpha_x, alpha_y = (
            scipy.ndimage.map_coordinates(
                data, [pixel_y, pixel_x], order=1, mode='nearest')
            for data in (self._alpha_x, self._alpha_y))
        source_x, source_y = frame_to_pixels(
            map_x - alpha_x, map_y - alpha_y, self._grid.coords.rval[:2],
            _projection(points.origin))
        return x - source_x, y - source_y

    def _check_covered(
            self,
            points: SkyPoints,
            pixel_x: np.ndarray,
            pixel_y: np.ndarray
    ) -> None:
        """
        Raise ModelEvaluationError if points (at the given pixel
        coordinates of the maps) are off the maps: beyond the centres of
        their pixels at the edges (to a rounding).
        """
        size_x, size_y = self._grid.size
        tolerance = 1e-6
        off = ~((pixel_x >= -tolerance) & (pixel_x <= size_x - 1 + tolerance)
                & (pixel_y >= -tolerance)
                & (pixel_y <= size_y - 1 + tolerance))
        if np.any(off):
            first = np.flatnonzero(off)[0]
            x = np.ravel(points.x)[first]
            y = np.ravel(points.y)[first]
            raise ModelEvaluationError(
                f"the deflection maps {self.name()!r} do not cover "
                f"{np.sum(off)} of the {off.size} points where they deflect "
                f"light (e.g. at "
                f"x = {x:.6g}, y = {y:.6g} arcsec from the origin of the "
                f"frame); give maps of a larger field, or a smaller field "
                f"of the data")


def _projection(origin: Sequence[float]) -> Grid:
    """
    Return the grid of one pixel of 1 arcsec at an origin (RA and Dec,
    degrees), not rotated, whose pixel coordinates are the positions in
    the frame of the origin (see frame_to_pixels): its gnomonic
    projection.
    """
    return make_grid((1, 1), 1.0, (0.0, 0.0), tuple(origin), 0)


def _dump_multipoles(multipoles: _Multipoles) -> dict[str, Any]:
    """Return the option of the multipoles of a shape, if it has any."""
    orders = multipoles.orders()
    return dict(multipoles=list(orders)) if orders else {}


def _theta_e_pdesc() -> ParamScalarDesc:
    """Return the parameter of the Einstein radius of a shape."""
    return ParamScalarDesc(
        "the Einstein radius", title='Einstein radius', min=0,
        unit='arcsec', default=1)


def _ellipse_pdescs() -> dict[str, ParamScalarDesc]:
    """Return the parameters of the shape of an elliptical mass."""
    return dict(
        # (q > 0 keeps the deflection finite)
        q=ParamScalarDesc(
            "the axis ratio", title='Axis ratio', min=1e-3, max=1, unit='',
            default=0.8),
        posa=ParamScalarDesc(
            "the position angle", title='Position angle', unit='deg',
            default=0, period=180),
        xpos=ParamScalarDesc(
            "the centre along x", title='Centre (x)', unit='arcsec',
            default=0),
        ypos=ParamScalarDesc(
            "the centre along y", title='Centre (y)', unit='arcsec',
            default=0))


def _centred(
        points: SkyPoints, params: Mapping[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the positions of points from the centre of a shape, along x
    and y.
    """
    return points.x - params['xpos'], points.y - params['ypos']


def _on_axes(
        x: np.ndarray, y: np.ndarray, posa: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return positions along x and y from the centre of a shape of position
    angle posa (degrees) along its major (u) and minor (v) axes (see the
    module).
    """
    posa = math.radians(posa)
    sin_pa, cos_pa = math.sin(posa), math.cos(posa)
    return -sin_pa * x + cos_pa * y, cos_pa * x + sin_pa * y


def _from_axes(
        alpha_u: np.ndarray, alpha_v: np.ndarray, posa: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a deflection along the axes of a shape of position angle posa
    (degrees) along x and y.
    """
    posa = math.radians(posa)
    sin_pa, cos_pa = math.sin(posa), math.cos(posa)
    return (-sin_pa * alpha_u + cos_pa * alpha_v,
            cos_pa * alpha_u + sin_pa * alpha_v)


def _read_map(
        file: DataFile,
        rpix: float | Sequence[float] | None,
        rval: float | Sequence[float] | None
) -> tuple[np.ndarray, Coords, Orientation]:
    """
    Return the data of a file in the orientation of the model with their
    world coordinates (as read_data), and how the data were put in it,
    which the vectors of deflection maps follow (see
    LensDeflectionMap.from_files).
    """
    filename = split_data_file(file)[0]
    data, header = read_image(file)
    header, orientation = model_header(filename, header, data.shape)
    coords = coords_from_header(filename, header, rpix=rpix, rval=rval)
    return orientation.apply(data), coords, orientation


def _along_model_axes(
        maps: Sequence[np.ndarray],
        orientations: Sequence[Orientation],
        scales: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection (arcsec) along the pixel axes of maps put in
    the orientation of the model (x then y), of the maps of it along the
    pixel axes of their files (that of x along the first axis of its
    file, that of y along the second; see LensDeflectionMap.from_files),
    how each was put in it, and the arcsec of a unit of their values
    along each axis of the model: each goes along the axis of the model
    that the axis of its file became, negated if that axis was reversed.
    """
    along = {}
    for file_axis, (data, orientation) in enumerate(zip(maps, orientations)):
        sky = orientation.sky()
        axis = sky.order.index(file_axis)
        sign = -1 if axis in sky.flips else 1
        along[axis] = sign * scales[axis] * data
    return along[0], along[1]


def _rotated(
        along_x: np.ndarray, along_y: np.ndarray, rota: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return vectors along the pixel axes of a grid of the rotation rota
    (degrees; see Coords) along x and y of the frame (see sky_positions);
    for -rota, vectors along x and y along the pixel axes.
    """
    rota = math.radians(rota)
    cos, sin = math.cos(rota), math.sin(rota)
    return cos * along_x - sin * along_y, sin * along_x + cos * along_y
