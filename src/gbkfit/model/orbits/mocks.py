"""
An exact mock galaxy for Schwarzschild models: a spherical galaxy of
Hernquist light, a constant mass-to-light ratio and a central black
hole, its stars isotropic. Its distribution function follows from
Eddington's formula, its LOSVDs from that, and its observation in the
bins of an IFU field with a Gaussian PSF from those: computed by
quadrature, without particles and without the orbit code.
"""

import dataclasses
import functools

import numpy as np
import numpy.typing as npt
import scipy.integrate
import scipy.interpolate
import scipy.special

from gbkfit.model.mass import G
from .gauss_hermite import fit_gauss_hermite
from .kinematics import BinnedField, StellarKinematics
from .profiles import Hernquist


__all__ = [
    'MockGalaxy',
    'add_noise',
    'observe'
]


# The radii (in scale radii) between which the distribution function is
# tabulated, the number of its energies, and the radius to which its
# integrals extend
_DF_RADII = (1e-8, 1e5)
_DF_ENERGIES = 600
_DF_REACH = 1e7
# The energies of the integral of the distribution function (see _df)
_CUMULATIVE_ENERGIES = 20000
# The Gauss-Legendre nodes of the line-of-sight integrals (see losvd)
_LOS_NODES = 128


@dataclasses.dataclass(frozen=True)
class MockGalaxy:
    """
    A spherical galaxy of Hernquist light (Lsun, kpc), of the mass-to-
    light ratio upsilon (Msun/Lsun) and a central black hole of bh_mass
    (Msun), its stars isotropic: their distribution function f(E) is
    that of Eddington's formula, for the binding energy E = Psi - v^2 / 2
    in the relative potential Psi = -Phi.
    """
    light: Hernquist
    upsilon: float
    bh_mass: float

    def relative_potential(self, r: npt.ArrayLike) -> np.ndarray:
        """Return its relative potential Psi ((km/s)^2) at the radii r."""
        r = np.asarray(r, dtype=float)
        return G * (self.upsilon * self.light.total / (r + self.light.a)
                    + self.bh_mass / r)

    def radius(self, psi: npt.ArrayLike) -> np.ndarray:
        """Return the radius (kpc) of the relative potential psi."""
        # psi r (r + a) = A r + B (r + a): a quadratic in r, of the
        # positive root (in its forms without cancellations)
        psi = np.asarray(psi, dtype=float)
        a = self.light.a
        A = G * self.upsilon * self.light.total
        B = G * self.bh_mass
        b = psi * a - A - B
        root = np.sqrt(b ** 2 + 4 * psi * B * a)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(
                b > 0, 2 * B * a / (b + root), (root - b) / (2 * psi))

    def distribution_function(self, energy: npt.ArrayLike) -> np.ndarray:
        """
        Return its distribution function (Lsun / kpc^3 / (km/s)^3) at the
        binding energies ((km/s)^2) given (0 where they are not
        positive).
        """
        return self._df.f(energy)

    def density(self, r: npt.ArrayLike) -> np.ndarray:
        """
        Return the density of its light (Lsun/kpc^3) at the radii r, from
        its distribution function: 4 pi int_0^Psi f(E) sqrt(2 (Psi - E))
        dE (the Hernquist density, to the precision of its tables).
        """
        return np.array([
            4 * np.pi * self._velocity_moment(psi, 1)
            for psi in np.ravel(self.relative_potential(r))]).reshape(
                np.shape(r))

    def dispersion(self, r: npt.ArrayLike) -> np.ndarray:
        """
        Return the dispersion (km/s) of the velocities of its stars along
        any direction at the radii r, from its distribution function:
        (4 pi / 3) int_0^Psi f(E) (2 (Psi - E))^(3/2) dE over its density
        (the isotropic Jeans dispersion, to the precision of its tables).
        """
        psi = np.ravel(self.relative_potential(r))
        moments = np.array([
            [self._velocity_moment(p, 1), self._velocity_moment(p, 3)]
            for p in psi])
        return np.sqrt(moments[:, 1] / (3 * moments[:, 0])).reshape(
            np.shape(r))

    def losvd(self, R: npt.ArrayLike, v: npt.ArrayLike) -> np.ndarray:
        """
        Return its LOSVD (Lsun / kpc^2 / (km/s)) at the projected radii R
        (kpc) and the line-of-sight velocities v (km/s).

        Parameters
        ----------
        R : array_like
            The projected radii (kpc, positive): an array of shape (n,).
        v : array_like
            The velocities: an array of shape (m,).

        Returns
        -------
        np.ndarray
            An array of shape (n, m).
        """
        # L(R, v) = 4 pi int_0^inf F(Psi(r) - v^2 / 2) dz, with F(Q) the
        # integral of f(E) from 0 to Q, from z = 0 to where Psi(r) = v^2 /
        # 2; in t, z = R sinh(t) (smooth in log r)
        R = np.asarray(R, dtype=float)[:, None, None]
        v = np.abs(np.asarray(v, dtype=float))[None, :, None]
        reach = np.minimum(
            self.radius(v ** 2 / 2), _DF_RADII[1] * self.light.a)
        top = np.arccosh(np.maximum(reach / R, 1))
        nodes, weights = np.polynomial.legendre.leggauss(_LOS_NODES)
        t = top * (nodes + 1) / 2
        r = R * np.cosh(t)
        integrand = self._df.cumulative(
            self.relative_potential(r) - v ** 2 / 2) * R * np.cosh(t)
        return 4 * np.pi * top[..., 0] / 2 * (integrand @ weights)

    @functools.cached_property
    def _df(self) -> '_DistributionTables':
        """Return the tables of its distribution function."""
        return _DistributionTables.of(self)

    def _velocity_moment(self, psi: float, power: int) -> float:
        """Return int_0^psi f(E) (2 (psi - E))^(power / 2) dE."""
        return scipy.integrate.quad(
            lambda e: self.distribution_function(e)
            * (2 * (psi - e)) ** (power / 2),
            0, psi, epsabs=0, epsrel=1e-10, limit=400)[0]

    def _potential_slope(self, r: npt.ArrayLike) -> np.ndarray:
        """Return -dPsi/dr ((km/s)^2/kpc) at the radii r."""
        r = np.asarray(r, dtype=float)
        return G * (self.upsilon * self.light.total / (r + self.light.a) ** 2
                    + self.bh_mass / r ** 2)

    def _second_derivative(self, r: np.ndarray) -> np.ndarray:
        """Return d^2 nu / dPsi^2 of its light nu at the radii r."""
        a = self.light.a
        nu = self.light.density(r)
        # (the log-derivative of nu, and the derivatives of Psi)
        slope = -1 / r - 3 / (r + a)
        dnu = nu * slope
        d2nu = nu * (slope ** 2 + 1 / r ** 2 + 3 / (r + a) ** 2)
        dpsi = -self._potential_slope(r)
        d2psi = 2 * G * (self.upsilon * self.light.total / (r + a) ** 3
                         + self.bh_mass / r ** 3)
        return (d2nu * dpsi - dnu * d2psi) / dpsi ** 3


@dataclasses.dataclass(frozen=True)
class _DistributionTables:
    """
    The distribution function f(E) of a mock galaxy and its integral F(Q)
    from 0 to Q, as splines of their logarithms in log r(E), the log of
    the radius of the potential E (where both are near power laws, with
    or without a black hole), extended linearly beyond their ends.
    """
    galaxy: MockGalaxy
    log_f: scipy.interpolate.CubicSpline
    log_cumulative: scipy.interpolate.CubicSpline

    @classmethod
    def of(cls, galaxy: MockGalaxy) -> '_DistributionTables':
        """Return the tables of the distribution function of a galaxy."""
        x = np.linspace(*np.log(np.multiply(galaxy.light.a, _DF_RADII)),
                        _DF_ENERGIES)
        energies = galaxy.relative_potential(np.exp(x))
        f = np.array([_eddington(galaxy, e) for e in energies])
        log_f = scipy.interpolate.CubicSpline(x, np.log(f))
        # F: int_r^inf f(Psi(r')) |dPsi/dr'| dr', in log r', from the outer
        # end inwards; beyond it f is a power law of E, f ~ E^k, whose
        # integral is f E / (k + 1)
        fine = np.linspace(x[0], x[-1], _CUMULATIVE_ENERGIES)
        r = np.exp(fine)
        integrand = np.exp(log_f(fine)) * galaxy._potential_slope(r) * r
        k = log_f(fine[-1], 1) / (
            -galaxy._potential_slope(r[-1]) * r[-1] / energies[-1])
        tail = np.exp(log_f(fine[-1])) * energies[-1] / (k + 1)
        cumulative = tail + np.concatenate((
            scipy.integrate.cumulative_trapezoid(
                integrand[::-1], -fine[::-1])[::-1], [0]))
        return cls(
            galaxy=galaxy, log_f=log_f,
            log_cumulative=scipy.interpolate.CubicSpline(
                fine, np.log(cumulative)))

    def f(self, energy: npt.ArrayLike) -> np.ndarray:
        """Return f at the energies (0 where there are no stars)."""
        return self._evaluate(self.log_f, energy)

    def cumulative(self, energy: npt.ArrayLike) -> np.ndarray:
        """Return F at the energies (0 where they are not positive)."""
        return self._evaluate(self.log_cumulative, energy)

    def _evaluate(
            self,
            spline: scipy.interpolate.CubicSpline,
            energy: npt.ArrayLike
    ) -> np.ndarray:
        """
        Return exp of a spline at log r(E), extended linearly beyond its
        ends (0 where E is not positive or above the potential at the
        centre).
        """
        energy = np.asarray(energy, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = self.galaxy.radius(energy)
        valid = (energy > 0) & (r > 0)
        x = np.log(r[valid])
        ends = spline.x[[0, -1]]
        inside = np.clip(x, *ends)
        values = np.zeros_like(energy)
        values[valid] = np.exp(
            spline(inside) + spline(inside, 1) * (x - inside))
        return values


def _eddington(galaxy: MockGalaxy, energy: float) -> float:
    """
    Return the distribution function of a galaxy at a binding energy, by
    Eddington's formula: f(E) = 1 / (sqrt(8) pi^2) int_0^E d^2 nu / dPsi^2
    dPsi / sqrt(E - Psi) (the boundary term is 0: d nu / dPsi = 0 at Psi =
    0).
    """
    # In r = r_E exp(s^2), from the radius r_E of Psi = E outwards: the
    # integrand is smooth, and E - Psi(r) is computed without
    # cancellations
    a = galaxy.light.a
    A = G * galaxy.upsilon * galaxy.light.total
    B = G * galaxy.bh_mass
    r_e = float(galaxy.radius(energy))

    def integrand(s: float) -> float:
        r = r_e * np.exp(s ** 2)
        # (E - Psi(r) = r_e expm1(s^2) gap; 2 s r ds = dr, and the s of
        # sqrt(expm1(s^2)) taken out)
        gap = A / ((r_e + a) * (r + a)) + B / (r_e * r)
        ratio = np.expm1(s ** 2) / s ** 2 if s > 0 else 1.0
        return float(galaxy._second_derivative(r)
                     * galaxy._potential_slope(r) * 2 * r
                     / np.sqrt(r_e * gap * ratio))

    top = np.sqrt(np.log(_DF_REACH * a / r_e))
    value = scipy.integrate.quad(
        integrand, 0, top, epsabs=0, epsrel=1e-11, limit=400)[0]
    return value / (np.sqrt(8) * np.pi ** 2)


def observe(
        galaxy: MockGalaxy,
        field: BinnedField,
        distance: float,
        vmax: float = 2000,
        dv: float = 10,
        order: int = 4
) -> StellarKinematics:
    """
    Observe a mock galaxy in the bins of a field, without noise: its
    LOSVDs convolved with the PSF, summed over the pixels of each bin, and
    fitted by Gauss-Hermite series.

    Parameters
    ----------
    galaxy : MockGalaxy
        The galaxy, at the centre of the field.
    field : BinnedField
        The field.
    distance : float
        Its distance (Mpc).
    vmax, dv : float, optional
        The range (-vmax to vmax) and the spacing of the velocities of
        the LOSVDs (km/s).
    order : int, optional
        The order of the Gauss-Hermite series.

    Returns
    -------
    StellarKinematics
        Its kinematics, with errors of 0.
    """
    scale = StellarKinematics(
        field, distance, *[np.zeros(0)] * 8).kpc_per_arcsec()
    width = field.psf_fwhm / np.sqrt(8 * np.log(2))
    v = np.arange(-vmax, vmax + dv / 2, dv)
    # The radii of the tables (arcsec): logarithmic near the centre, then
    # finer than the PSF to the corners of the field plus 6 PSF widths
    x, y = field.pixel_centres()
    corner = np.hypot(x, y).max() + field.pixel + 6 * width
    step = width / 20
    radii = np.concatenate((
        np.geomspace(1e-4, step, 40, endpoint=False),
        np.arange(step, corner, step)))
    # The LOSVDs (symmetric in v), then convolved with the PSF: a ring
    # kernel, (1 / s^2) exp(-(R - R')^2 / (2 s^2)) I0(R R' / s^2), over
    # R' dR' (trapezoids)
    positive = v >= 0
    half = galaxy.losvd(radii * scale, v[positive])
    losvds = np.concatenate((half[:, :0:-1], half), axis=1) \
        if v[0] < 0 else half
    kernel = np.exp(-(radii[:, None] - radii) ** 2 / (2 * width ** 2)) \
        * scipy.special.i0e(radii[:, None] * radii / width ** 2) / width ** 2
    blurred = (kernel * _trapezoid_weights(radii) * radii) @ losvds
    # The LOSVD of each bin: the sum over its pixels, each sampled on a
    # grid of 4 x 4 points, interpolated linearly in the radius (Lsun /
    # (km/s))
    offsets = (np.arange(4) - 1.5) / 4 * field.pixel
    pixels = field.bin_map >= 0
    sub_x = x[pixels][:, None, None] + offsets[None, None, :]
    sub_y = y[pixels][:, None, None] + offsets[None, :, None]
    distances = np.hypot(sub_x, sub_y).ravel()
    bins = np.repeat(field.bin_map[pixels], 16)
    area = (field.pixel * scale) ** 2 / 16
    n = len(radii)
    i = np.searchsorted(radii, distances) - 1
    fraction = (distances - radii[i]) / (radii[i + 1] - radii[i])
    coverage = sum(
        np.bincount(bins * n + i + k, area * weight,
                    minlength=field.nbins() * n)
        for k, weight in ((0, 1 - fraction), (1, fraction)))
    bin_losvds = coverage.reshape(field.nbins(), n) @ blurred
    fits = [fit_gauss_hermite(v, losvd, order) for losvd in bin_losvds]
    zeros = np.zeros(field.nbins())
    return StellarKinematics(
        field=field, distance=distance,
        flux=bin_losvds.sum(axis=1) * dv, flux_error=zeros,
        mean=np.array([fit.mean for fit in fits]), mean_error=zeros,
        sigma=np.array([fit.sigma for fit in fits]), sigma_error=zeros,
        h=np.array([fit.h for fit in fits]),
        h_error=np.zeros((field.nbins(), order - 2)))


def add_noise(
        kinematics: StellarKinematics,
        error: float = 0.03,
        flux_error: float = 0.01,
        seed: int = 0
) -> StellarKinematics:
    """
    Return kinematics with errors and Gaussian noise of those errors: of
    the mean and the dispersion error times the dispersion, of each h
    error / sqrt(2) (as those of h_1 and h_2, Rix et al. 1997), and of the
    light flux_error times it.

    Parameters
    ----------
    kinematics : StellarKinematics
        The kinematics (e.g. of observe).
    error : float, optional
        The relative error of the moments.
    flux_error : float, optional
        The relative error of the light.
    seed : int, optional
        The seed of the noise.

    Returns
    -------
    StellarKinematics
        The noisy kinematics.
    """
    rng = np.random.default_rng(seed)
    errors = dict(
        flux=flux_error * kinematics.flux,
        mean=error * kinematics.sigma,
        sigma=error * kinematics.sigma,
        h=np.full_like(kinematics.h, error / np.sqrt(2)))
    noisy = {
        name: getattr(kinematics, name) + rng.normal(size=sigma.shape) * sigma
        for name, sigma in errors.items()}
    return dataclasses.replace(
        kinematics, **noisy,
        **{f'{name}_error': sigma for name, sigma in errors.items()})


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Return the weights of the trapezoid rule at the points x."""
    gaps = np.diff(x)
    return np.concatenate(([gaps[0]], gaps[1:] + gaps[:-1], [gaps[-1]])) / 2
