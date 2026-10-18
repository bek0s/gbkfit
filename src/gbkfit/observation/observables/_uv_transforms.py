"""
The visibilities of the channels of model cubes: transforms of the images
of the channels to the visibilities of their baselines, by the NUFFTs of
the drivers (see UVTransform).
"""

import abc
import math
from collections.abc import Sequence

import numpy as np
import scipy.special

from gbkfit.driver import DeviceArray, Driver
from gbkfit.utils import gridutils


__all__ = [
    'UVTransform',
    'UVTransformExact',
    'UVTransformFlat',
    'UVTransformSin',
    'uv_transforms'
]


# The speed of light (m/s), and arcsec per radian
_C = 299792458.0
_ARCSEC = np.degrees(1) * 3600


class UVTransform(abc.ABC):
    """
    The visibilities of the images of the channels of a cube on a grid,
    at the baselines of each channel: of the flux a_j of each pixel, at its
    direction s_j on the sky, at a baseline b in wavelengths,

        V(b) = sum_j a_j exp(-2 pi i b . (s_j - p)),

    with p the phase centre; in its frame, of the axes east, north and p,
    b . (s_j - p) = u l_j + v m_j + w (n_j - 1). The transforms differ in
    the directions of the pixels and in the w-term (see each). They sum in
    the frame of the reference pixel of the grid, c, and then move the
    phase centre to p, which is exact: V(b) = V_c(b) exp(-2 pi i b . (c -
    p)).

    Each channel is transformed by a NUFFT made for it in each call, so
    that the memory of the device holds those of one channel only.

    Parameters
    ----------
    driver : Driver
        The driver of the cubes.
    grid : Grid
        The grid of the cubes; only its x and y axes are used, and its
        reference pixel is at the RA and Dec of its rval.
    uvw : numpy.ndarray
        The baselines (u, v, w) of the rows of the visibilities, in m, in
        the frame of the phase centre, of shape (nrows, 3).
    frequencies : Sequence of float
        The frequencies of the channels (Hz).
    phase_centre : Sequence of float
        The RA and Dec of the phase centre (degrees).
    eps : float
        The relative tolerance of the visibilities (see DriverNUFFT).

    Raises
    ------
    ValueError
        If the baselines are not of shape (nrows, 3) or not finite, the
        frequencies are not positive, the phase centre is not an RA and a
        Dec, or the tolerance is not positive.
    """

    def __init__(
            self,
            driver: Driver,
            grid: gridutils.Grid,
            uvw: np.ndarray,
            frequencies: Sequence[float],
            phase_centre: Sequence[float],
            eps: float
    ):
        if not eps > 0:
            raise ValueError(f"eps must be positive; it is {eps}")
        self._driver = driver
        self._nufft = driver.nufft()
        self._eps = eps
        size_x, size_y = grid.size[:2]
        self._shape = (size_y, size_x)
        self._baselines = _grid_baselines(grid, uvw, frequencies, phase_centre)

    def nrows(self) -> int:
        """Return the number of rows of the visibilities."""
        return len(self._baselines[0][0])

    def __call__(self, cube: DeviceArray, out: DeviceArray) -> None:
        """
        Compute the visibilities of a cube.

        Parameters
        ----------
        cube : DeviceArray
            The flux of the pixels, of shape (nchannels, ny, nx), real.
        out : DeviceArray
            The visibilities, of shape (nchannels, nrows), complex128.
        """
        driver = self._driver
        for channel, coefficients in enumerate(self._coefficients_h):
            # The data of each term: the image times the factors of its
            # pixels
            driver.math_mul(cube[channel], self._images, out=self._data.real)
            self._transform(channel, self._data, self._transforms)
            # The visibilities: the sum of the transforms of the terms,
            # times their coefficients at each baseline
            coefficients_d = driver.mem_copy_h2d(
                coefficients, self._coefficients)
            driver.math_mul(
                coefficients_d[0], self._transforms[0], out=out[channel])
            for term in range(1, len(coefficients)):
                driver.math_mul(
                    coefficients_d[term], self._transforms[term],
                    out=self._product)
                driver.math_add(out[channel], self._product, out=out[channel])

    def adjoint(self, visibilities: DeviceArray, out: DeviceArray) -> None:
        """
        Compute the adjoint of the transform: of visibilities V_k at the
        baselines of each channel, the images

            I_j = sum_k V_k conj(dV_k / da_j)

        (e.g. sum_k V_k exp(2 pi i b_k . (s_j - p)) of the exact transform;
        of weighted visibilities, a dirty image).

        Parameters
        ----------
        visibilities : DeviceArray
            The visibilities, of shape (nchannels, nrows), complex128.
        out : DeviceArray
            The images, of shape (nchannels, ny, nx), complex128.
        """
        driver = self._driver
        terms = driver.mem_alloc_d(self._data.shape, np.complex128)
        product = driver.mem_alloc_d(self._shape, np.complex128)
        for channel, coefficients in enumerate(self._coefficients_h):
            # The data of each term: the visibilities times the conjugates
            # of its coefficients at each baseline
            coefficients_d = driver.mem_copy_h2d(
                np.conj(coefficients), self._coefficients)
            driver.math_mul(
                coefficients_d, visibilities[channel], out=self._transforms)
            self._adjoint_transform(channel, self._transforms, terms)
            # The image: the sum of the transforms of the terms, times the
            # factors of their pixels
            driver.math_mul(self._images[0], terms[0], out=out[channel])
            for term in range(1, len(coefficients)):
                driver.math_mul(self._images[term], terms[term], out=product)
                driver.math_add(out[channel], product, out=out[channel])

    def _set_terms(
            self,
            images: np.ndarray,
            coefficients: list[np.ndarray]
    ) -> None:
        """
        Set the terms of the visibilities of each channel: the factors of
        the pixels of each term (of shape (nterms, ny, nx)), and the
        coefficients of the terms at each baseline of each channel (of
        shape (nterms, nrows)).
        """
        driver = self._driver
        nterms, nrows = coefficients[0].shape
        self._coefficients_h = coefficients
        self._images = driver.mem_copy_h2d(np.ascontiguousarray(images))
        self._data = driver.mem_alloc_d(images.shape, np.complex128)
        driver.mem_fill(self._data, 0)
        self._transforms = driver.mem_alloc_d((nterms, nrows), np.complex128)
        self._coefficients = driver.mem_alloc_d(
            (nterms, nrows), np.complex128)
        self._product = driver.mem_alloc_d(nrows, np.complex128)

    @abc.abstractmethod
    def _transform(
            self,
            channel: int,
            data: DeviceArray,
            out: DeviceArray
    ) -> None:
        """
        Transform the data of the terms of a channel (of shape (nterms, ny,
        nx)) into out (of shape (nterms, nrows)), in the frame of the
        reference pixel.
        """

    @abc.abstractmethod
    def _adjoint_transform(
            self,
            channel: int,
            data: DeviceArray,
            out: DeviceArray
    ) -> None:
        """
        Transform the data of the terms at the baselines of a channel (of
        shape (nterms, nrows)) into out (of shape (nterms, ny, nx)) by the
        adjoint of _transform.
        """


class UVTransformExact(UVTransform):
    """
    The exact visibilities (see UVTransform): the pixels at their
    directions on the sky, those of a gnomonic (TAN) projection of the grid
    about its reference pixel, as for all the observables (see
    gridutils.pixels_on); the w-term exact. A type-3 NUFFT from the
    (l, m, n - 1) of the pixels to the baselines of each channel.
    """

    def __init__(
            self,
            driver: Driver,
            grid: gridutils.Grid,
            uvw: np.ndarray,
            frequencies: Sequence[float],
            phase_centre: Sequence[float],
            eps: float
    ):
        super().__init__(driver, grid, uvw, frequencies, phase_centre, eps)
        # (l, m, n) = (east, north, 1) / sqrt(1 + r^2), and n - 1 without
        # cancellation
        east, north = _offsets(grid)
        r2 = east ** 2 + north ** 2
        distance = np.sqrt(1 + r2)
        self._points = np.column_stack((
            (east / distance).ravel(), (north / distance).ravel(),
            (-r2 / (distance * (1 + distance))).ravel()))
        self._set_terms(
            np.ones((1, *self._shape)),
            [rephase[None] for _, rephase in self._baselines])

    def _transform(
            self,
            channel: int,
            data: DeviceArray,
            out: DeviceArray
    ) -> None:
        baselines, _ = self._baselines[channel]
        nufft = self._nufft.type3(self._points, baselines, -1, self._eps)
        nufft(data.reshape(-1), out.reshape(-1))

    def _adjoint_transform(
            self,
            channel: int,
            data: DeviceArray,
            out: DeviceArray
    ) -> None:
        baselines, _ = self._baselines[channel]
        nufft = self._nufft.type3(baselines, self._points, 1, self._eps)
        nufft(data.reshape(-1), out.reshape(-1))


class _UVTransformGrid(UVTransform):
    """
    Visibilities of the pixels at the directions of an orthographic (SIN)
    projection of the grid about its reference pixel: uniform in (l, m), a
    type-2 NUFFT of the images of the terms of the w-term for each
    baseline.

    Attributes
    ----------
    _w_term : type of _WTerm
        The w-term (of the subclasses).
    """

    _w_term: type['_WTerm']

    def __init__(
            self,
            driver: Driver,
            grid: gridutils.Grid,
            uvw: np.ndarray,
            frequencies: Sequence[float],
            phase_centre: Sequence[float],
            eps: float
    ):
        super().__init__(driver, grid, uvw, frequencies, phase_centre, eps)
        east, north = _offsets(grid)
        r2 = east ** 2 + north ** 2
        if np.any(r2 >= 1):
            raise ValueError(
                "the pixels of the grid must be less than 90 degrees from "
                "its reference pixel")
        # The (l, m) of the pixels: those of the mode k = 0 of the type-2
        # NUFFT (the pixel shape // 2), plus the matrix of k
        origin = np.array([
            east[self._shape[0] // 2, self._shape[1] // 2],
            north[self._shape[0] // 2, self._shape[1] // 2]])
        steps = _steps(grid)
        w_max = max(np.abs(b[:, 2]).max(initial=0) for b, _ in self._baselines)
        # (n - 1 = sqrt(1 - r^2) - 1, without cancellation)
        w_term = self._w_term(-r2 / (1 + np.sqrt(1 - r2)), w_max, eps)
        # The points of each channel: the phases per unit of the modes of
        # the y and x axes; the coefficients: the phase of the origin, that
        # of the phase centre, and those of the w-term
        self._points = []
        coefficients = []
        for baselines, rephase in self._baselines:
            self._points.append((baselines[:, :2] @ steps)[:, ::-1])
            phase = np.exp(-2j * np.pi * baselines[:, :2] @ origin)
            coefficients.append(
                w_term.coefficients(baselines[:, 2]) * (phase * rephase))
        self._set_terms(w_term.images, coefficients)

    def _transform(
            self,
            channel: int,
            data: DeviceArray,
            out: DeviceArray
    ) -> None:
        nufft = self._nufft.type2(
            self._points[channel], self._shape, -1, self._eps, len(data))
        nufft(data, out)

    def _adjoint_transform(
            self,
            channel: int,
            data: DeviceArray,
            out: DeviceArray
    ) -> None:
        nufft = self._nufft.type1(
            self._points[channel], self._shape, 1, self._eps, len(data))
        nufft(data, out)


class _WTerm(abc.ABC):
    """
    The w-term of the visibilities of pixels, exp(-2 pi i w d_j), d_j =
    n_j - 1, as a sum of terms: of factors of the pixels, times
    coefficients of the w of each baseline.

    Parameters
    ----------
    d : numpy.ndarray
        The n - 1 of the pixels, of shape (ny, nx).
    w_max : float
        The largest |w| of the baselines (wavelengths).
    eps : float
        The relative tolerance of the visibilities.
    """

    @abc.abstractmethod
    def __init__(self, d: np.ndarray, w_max: float, eps: float):
        pass

    @property
    @abc.abstractmethod
    def images(self) -> np.ndarray:
        """The factors of the pixels of the terms, (nterms, ny, nx)."""

    @abc.abstractmethod
    def coefficients(self, w: np.ndarray) -> np.ndarray:
        """
        Return the coefficients of the terms at baselines of a w, of shape
        (nterms, nrows).
        """


class _ChebyshevWTerm(_WTerm):
    """
    The w-term as its Chebyshev series in the pixels (from the generating
    function of the Bessel functions, e^(-i a cos t) = sum_p eps_p (-i)^p
    J_p(a) cos(p t)): with d = d_c + h x,

        exp(-2 pi i w d) = exp(-2 pi i w d_c)
            * sum_p eps_p (-i)^p J_p(2 pi w h) T_p(x),

    eps_0 = 1 and eps_p = 2, cut where the rest is at most the tolerance:
    as |J_p(a)| <= (a / 2)^p / p! and |T_p(x)| <= 1, the rest after the
    terms below n is at most the sum of 2 (a / 2)^p / p! from p = n (of the
    flux), with a = 2 pi w_max h.
    """

    def __init__(self, d: np.ndarray, w_max: float, eps: float):
        self._centre = (d.max() + d.min()) / 2
        self._half_range = (d.max() - d.min()) / 2
        nterms = _chebyshev_terms(2 * np.pi * w_max * self._half_range, eps)
        x = (d - self._centre) / self._half_range if self._half_range \
            else np.zeros_like(d)
        self._images = np.moveaxis(
            np.polynomial.chebyshev.chebvander(x, nterms - 1), -1, 0)
        orders = np.arange(nterms)
        self._factors = np.where(orders == 0, 1, 2) * (-1j) ** orders

    @property
    def images(self) -> np.ndarray:
        return self._images

    def coefficients(self, w: np.ndarray) -> np.ndarray:
        orders = np.arange(len(self._factors))[:, None]
        bessel = scipy.special.jv(orders, 2 * np.pi * self._half_range * w)
        return self._factors[:, None] * bessel \
            * np.exp(-2j * np.pi * self._centre * w)


class _NoWTerm(_WTerm):
    """The w-term left out: exp(-2 pi i w d) = 1, one term."""

    def __init__(self, d: np.ndarray, w_max: float, eps: float):
        self._images = np.ones((1, *d.shape))

    @property
    def images(self) -> np.ndarray:
        return self._images

    def coefficients(self, w: np.ndarray) -> np.ndarray:
        return np.ones((1, len(w)), complex)


class UVTransformSin(_UVTransformGrid):
    """
    The visibilities of the pixels at the directions of an orthographic
    (SIN) projection of the grid about its reference pixel (see
    UVTransform), with the w-term exact: an approximation of the geometry
    of the other observables (a gnomonic, TAN, projection; see
    UVTransformExact), whose pixels at an angle t from the reference pixel
    are at sin t, not tan t: t^3 / 2 nearer to it (e.g. 5e-4 arcsec at
    0.1 deg, 0.07 arcsec at 0.5 deg). The w-term is its Chebyshev series in
    the pixels (see _ChebyshevWTerm), to the tolerance: a batch of type-2
    NUFFTs of the image times each term.
    """

    _w_term = _ChebyshevWTerm


class UVTransformFlat(_UVTransformGrid):
    """
    The visibilities of the pixels of an orthographic (SIN) projection of
    the grid about its reference pixel (see UVTransformSin), without the
    w-term (in the frame of the reference pixel): an approximation of a
    phase error of about pi w t^2 at an angle t from it (e.g. 2e-4 rad for
    2 arcsec at w = 7e5 wavelengths, 0.35 rad for 0.14 deg at 2e4). One
    type-2 NUFFT of the image.
    """

    _w_term = _NoWTerm


# The transforms, by their names in the options of the observables
uv_transforms = dict(
    exact=UVTransformExact, sin=UVTransformSin, flat=UVTransformFlat)


def _grid_baselines(
        grid: gridutils.Grid,
        uvw: np.ndarray,
        frequencies: Sequence[float],
        phase_centre: Sequence[float]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Return the baselines of each channel in wavelengths in the frame of the
    reference pixel of a grid, (nrows, 3), and the factors that move their
    visibilities to the phase centre, exp(-2 pi i (w_c - w)), (nrows,).
    """
    uvw = np.asarray(uvw, float)
    frequencies = np.asarray(frequencies, float)
    if uvw.ndim != 2 or uvw.shape[1] != 3 or not np.isfinite(uvw).all():
        raise ValueError(
            f"the baselines must be finite, of shape (nrows, 3); their "
            f"shape is {uvw.shape}")
    if frequencies.ndim != 1 or not np.all(frequencies > 0):
        raise ValueError(
            f"the frequencies must be positive; they are {frequencies}")
    if np.shape(phase_centre) != (2,):
        raise ValueError(
            f"the phase centre must be an RA and a Dec; it is {phase_centre}")
    rotation = _frame(grid.coords.rval[:2]) @ _frame(phase_centre).T
    baselines = []
    for frequency in frequencies:
        uvw_p = uvw * (frequency / _C)
        uvw_c = uvw_p @ rotation.T
        rephase = np.exp(-2j * np.pi * (uvw_c[:, 2] - uvw_p[:, 2]))
        baselines.append((uvw_c, rephase))
    return baselines


def _frame(position: Sequence[float]) -> np.ndarray:
    """
    Return the axes of the frame of a direction (RA and Dec, degrees):
    east, north and the direction, as the rows of a matrix, in equatorial
    coordinates (x to RA 0, z to the north pole).
    """
    ra, dec = np.radians(position)
    return np.array([
        [-np.sin(ra), np.cos(ra), 0],
        [-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)],
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]])


def _offsets(grid: gridutils.Grid) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the offsets of the pixels of a grid from its reference pixel, to
    the east and to the north (radians), of shape (ny, nx).
    """
    x, y = gridutils.sky_positions(grid)
    return -x / _ARCSEC, y / _ARCSEC


def _steps(grid: gridutils.Grid) -> np.ndarray:
    """
    Return the offsets (radians, east and north; see _offsets) of a step
    of the pixels of a grid along x and along y, as the columns of a
    matrix.
    """
    # (those of the pixels of a grid of 2 x 2 of the same world coordinates)
    east, north = _offsets(grid._replace(size=(2, 2)))
    return np.array([
        [east[0, 1] - east[0, 0], east[1, 0] - east[0, 0]],
        [north[0, 1] - north[0, 0], north[1, 0] - north[0, 0]]])


def _chebyshev_terms(a: float, eps: float) -> int:
    """
    Return the number of terms n of the Chebyshev series of exp(-i a x)
    (see _ChebyshevWTerm) whose rest, the sum of 2 (a / 2)^p / p! from p =
    n, is at most eps.
    """
    if a == 0:
        return 1
    nterms = 1
    while True:
        # (the rest is at most its first term over 1 - the ratio of the
        # terms, once that ratio is below 1)
        ratio = a / 2 / (nterms + 1)
        if ratio < 1:
            first = 2 * math.exp(
                nterms * math.log(a / 2) - math.lgamma(nterms + 1))
            if first / (1 - ratio) <= eps:
                return nterms
        nterms += 1
