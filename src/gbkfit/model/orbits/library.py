"""
Orbit libraries of Schwarzschild models: orbits integrated in the
potential of a mass model, each recorded as the fractions of its time in
shells of radius (intrinsic) and in the spatial and velocity bins of an
IFU field (projected, blurred by the PSF).

Lengths are in kpc, velocities in km/s and times in kpc / (km/s) (about
0.98 Gyr); sky positions are in arcsec.
"""

import dataclasses
from collections.abc import Callable
from typing import Protocol

import numpy as np
import numpy.typing as npt
import scipy.integrate

from gbkfit.model.mass import G, MassContext, MassModel, MassSpherical
from gbkfit.params import ParamValues
from .kinematics import BinnedField


__all__ = [
    'LibraryOptions',
    'OrbitLibrary',
    'SphericalPotential',
    'Tracer',
    'build_library',
    'initial_conditions',
    'integrate'
]


# The radii (kpc) and the number of the tables of a spherical potential
_TABLE_RADII = (1e-8, 1e6)
_TABLE_SIZE = 8000
# The number of samples of orbits recorded at once (see _Recorder)
_RECORD_BATCH = 2_000_000


class Tracer(Protocol):
    """The light whose orbits a library holds: a spherical profile."""

    def density(self, r: npt.ArrayLike) -> np.ndarray:
        """Return its density (per kpc^3) at the spherical radii r."""
        ...

    def enclosed(self, r: npt.ArrayLike) -> np.ndarray:
        """Return its light within the spherical radii r."""
        ...


class SphericalPotential:
    """
    The potential of a spherical mass model, tabulated: its enclosed
    mass (exact at the nodes of its tables, which extend to 1e6 kpc) and
    its potential (from that, by quadrature), interpolated linearly in
    log r.

    Parameters
    ----------
    model : MassModel
        The mass model, of spherical components only.
    params : ParamValues
        The values of its parameters.
    context : MassContext
        Its context (see MassContext).

    Raises
    ------
    NotImplementedError
        If a component of the model is not spherical.
    """

    def __init__(
            self,
            model: MassModel,
            params: ParamValues,
            context: MassContext
    ):
        if not all(isinstance(component, MassSpherical)
                   for component in model.components()):
            raise NotImplementedError(
                "orbits are integrated in spherical mass models only")
        self._log_r = np.linspace(*np.log(_TABLE_RADII), _TABLE_SIZE)
        r = np.exp(self._log_r)
        mass = model.vcirc2(r, params, context) * r / G
        # G M(<r) / r^3 = Omega^2, the square of the circular frequency
        self._log_omega2 = np.log(G * mass / r ** 3)
        # Phi(r) = Phi(r_max) - int_r^r_max G M / r'^2 dr', in log r', with
        # Phi(r_max) that of the mass within r_max (the mass beyond the
        # tables neglected)
        outer = -G * mass[-1] / r[-1]
        inward = scipy.integrate.cumulative_simpson(
            (G * mass / r)[::-1], x=-self._log_r[::-1], initial=0)[::-1]
        self._potential = outer - inward
        self._circular_energy = self._potential + G * mass / (2 * r)

    def gravity(
            self, positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the accelerations ((km/s)^2/kpc) at positions (kpc; an
        array of shape (n, 3)), and the circular frequencies (km/s/kpc)
        of their radii.
        """
        log_r = np.log(np.einsum('ij,ij->i', positions, positions)) / 2
        omega2 = np.exp(np.interp(log_r, self._log_r, self._log_omega2))
        return -omega2[:, None] * positions, np.sqrt(omega2)

    def enclosed_mass(self, r: npt.ArrayLike) -> np.ndarray:
        """Return the mass (Msun) within the radii r (kpc)."""
        return self.circular_frequency(r) ** 2 * np.asarray(r) ** 3 / G

    def potential(self, r: npt.ArrayLike) -> np.ndarray:
        """Return the potential ((km/s)^2) at the radii r (kpc)."""
        return np.interp(np.log(r), self._log_r, self._potential)

    def circular_frequency(self, r: npt.ArrayLike) -> np.ndarray:
        """
        Return the frequency (km/s/kpc) of the circular orbits of the
        radii r (kpc).
        """
        return np.exp(np.interp(
            np.log(r), self._log_r, self._log_omega2) / 2)

    def circular_radius(self, energy: npt.ArrayLike) -> np.ndarray:
        """
        Return the radius (kpc) of the circular orbit of each energy
        ((km/s)^2).
        """
        return np.exp(np.interp(energy, self._circular_energy, self._log_r))


@dataclasses.dataclass(frozen=True)
class LibraryOptions:
    """
    The options of an orbit library: the number of its orbits, the radii
    (kpc) between which they start (uniformly in log r), the circular
    periods of their energies for which they are integrated, the accuracy
    of their steps (the radians of the circular orbit of the radius of a
    step that it lasts), the radii (kpc) of the edges of its shells (from
    0), the largest velocity (km/s) and the number of its velocity bins,
    the number of the symmetric copies of each point of an orbit (rotated
    about the z axis and reflected about its plane at random), and the
    seed of its random numbers.
    """
    orbits: int = 3000
    radii: tuple[float, float] = (1e-3, 20.0)
    periods: float = 50
    accuracy: float = 0.02
    shell_edges: tuple[float, ...] = tuple(
        np.concatenate(([0], np.geomspace(5e-3, 20, 24))))
    vmax: float = 1500
    velocity_bins: int = 120
    symmetries: int = 4
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class OrbitLibrary:
    """
    An orbit library: the initial positions (kpc) and velocities (km/s)
    of its orbits (arrays of shape (orbits, 3)), their prior masses (the
    fractions of the light of the tracer that they would hold if the
    velocities of its stars were those of its isotropic Jeans model; they
    sum to 1), the fractions of their time in each shell (an array of
    shape (orbits, shells), of the shells between shell_edges, kpc), and
    in each spatial and velocity bin (an array of shape (orbits, bins,
    velocity bins), of the velocity bins between velocity_edges, km/s:
    velocities beyond them are in the first or the last).
    """
    positions: np.ndarray
    velocities: np.ndarray
    prior: np.ndarray
    shell_edges: np.ndarray
    shells: np.ndarray
    velocity_edges: np.ndarray
    losvds: np.ndarray


def initial_conditions(
        potential: SphericalPotential,
        tracer: Tracer,
        count: int,
        radii: tuple[float, float],
        seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return initial conditions of orbits: radii stratified uniformly in log
    r between two radii, isotropic directions, and velocities of the
    isotropic Jeans model of a tracer in a potential (a Gaussian of its
    dispersion along each axis, the speeds capped at 0.95 times the
    escape speed).

    Parameters
    ----------
    potential : SphericalPotential
        The potential.
    tracer : Tracer
        The tracer.
    count : int
        The number of orbits.
    radii : tuple of float
        The smallest and the largest radius (kpc).
    seed : int
        The seed of the random numbers.

    Returns
    -------
    positions, velocities : np.ndarray
        Arrays of shape (count, 3).
    prior : np.ndarray
        The prior mass of each orbit (see OrbitLibrary).
    """
    rng = np.random.default_rng(seed)
    log_r = np.log(radii[0]) + np.log(radii[1] / radii[0]) * (
        np.arange(count) + rng.random(count)) / count
    r = np.exp(log_r)
    positions = r[:, None] * _directions(rng, count)
    # The isotropic Jeans dispersion: sigma^2 = int_r^inf nu G M / r'^2
    # dr' / nu, on the tables of the potential
    table_log_r = np.linspace(*np.log(_TABLE_RADII), _TABLE_SIZE)
    table_r = np.exp(table_log_r)
    nu = tracer.density(table_r)
    pressure = scipy.integrate.cumulative_simpson(
        (nu * G * potential.enclosed_mass(table_r) / table_r)[::-1],
        x=-table_log_r[::-1], initial=0)[::-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        log_sigma2 = np.log(pressure / nu)
    finite = np.isfinite(log_sigma2)
    sigma = np.exp(np.interp(
        log_r, table_log_r[finite], log_sigma2[finite]) / 2)
    velocities = sigma[:, None] * rng.normal(size=(count, 3))
    speed = np.linalg.norm(velocities, axis=1)
    cap = 0.95 * np.sqrt(-2 * potential.potential(r))
    velocities *= np.minimum(1, cap / speed)[:, None]
    # The light per unit log r, over the density of the radii in log r
    prior = tracer.density(r) * r ** 3
    return positions, velocities, prior / prior.sum()


def integrate(
        potential: SphericalPotential,
        positions: np.ndarray,
        velocities: np.ndarray,
        durations: np.ndarray,
        accuracy: float,
        record: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                         None] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate orbits, all at once, by kick-drift-kick leapfrog steps,
    each orbit with steps of its own, time-symmetric (Hut, Makino &
    McMillan 1995): each step lasts the mean of accuracy / Omega(r) at its
    start and at its end (Omega the circular frequency of the radius r),
    the end that of a trial step of the first.

    Parameters
    ----------
    potential : SphericalPotential
        The potential.
    positions, velocities : np.ndarray
        The initial positions (kpc) and velocities (km/s) of the orbits:
        arrays of shape (orbits, 3).
    durations : np.ndarray
        The time for which each orbit is integrated (kpc / (km/s)).
    accuracy : float
        The accuracy of the steps (the radians of a circular orbit in a
        step).
    record : callable, optional
        Called after each step with the indices of the orbits of the
        step, their positions in the middle of it (an array of shape (n,
        3)), their velocities then, and its durations: the orbits sampled
        at the middles of their steps, weighted by time.

    Returns
    -------
    positions, velocities : np.ndarray
        The final positions and velocities of the orbits.
    """
    def step(
            x: np.ndarray, v: np.ndarray, acceleration: np.ndarray,
            dt: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        """
        Return the middle positions and velocities of a step, the final
        positions and velocities, and the accelerations and circular
        frequencies there.
        """
        half = v + acceleration * (dt / 2)[:, None]
        middle = x + half * (dt / 2)[:, None]
        x = middle + half * (dt / 2)[:, None]
        acceleration, omega = potential.gravity(x)
        return middle, half, x, half + acceleration * (dt / 2)[:, None], \
            acceleration, omega

    x = np.array(positions, dtype=float)
    v = np.array(velocities, dtype=float)
    final_x, final_v = x.copy(), v.copy()
    orbits = np.arange(len(x))
    remaining = np.asarray(durations, dtype=float).copy()
    acceleration, omega = potential.gravity(x)
    while len(orbits):
        trial = np.minimum(accuracy / omega, remaining)
        *_, trial_omega = step(x, v, acceleration, trial)
        dt = np.minimum(accuracy * (1 / omega + 1 / trial_omega) / 2,
                        remaining)
        middle, half, x, v, acceleration, omega = step(
            x, v, acceleration, dt)
        if record is not None:
            record(orbits, middle, half, dt)
        remaining -= dt
        # (finished orbits leave the arrays)
        done = remaining <= 0
        if np.any(done):
            final_x[orbits[done]], final_v[orbits[done]] = x[done], v[done]
            keep = ~done
            orbits, x, v, remaining = orbits[keep], x[keep], v[keep], \
                remaining[keep]
            acceleration, omega = acceleration[keep], omega[keep]
    return final_x, final_v


def build_library(
        model: MassModel,
        params: ParamValues,
        context: MassContext,
        tracer: Tracer,
        field: BinnedField,
        kpc_per_arcsec: float,
        inclination: float = 90,
        options: LibraryOptions = LibraryOptions()
) -> OrbitLibrary:
    """
    Build the orbit library of a tracer in a mass model, observed in the
    bins of a field.

    Parameters
    ----------
    model : MassModel
        The mass model, of spherical components only.
    params : ParamValues
        The values of its parameters.
    context : MassContext
        Its context (see MassContext).
    tracer : Tracer
        The tracer.
    field : BinnedField
        The field.
    kpc_per_arcsec : float
        The kpc in an arcsec at the distance of the galaxy.
    inclination : float, optional
        The inclination of the z axis to the line of sight (degrees; 90:
        edge-on).
    options : LibraryOptions, optional
        Its options.

    Returns
    -------
    OrbitLibrary
        The library.
    """
    potential = SphericalPotential(model, params, context)
    positions, velocities, prior = initial_conditions(
        potential, tracer, options.orbits, options.radii, options.seed)
    energies = potential.potential(np.linalg.norm(positions, axis=1)) \
        + np.einsum('ij,ij->i', velocities, velocities) / 2
    durations = options.periods * 2 * np.pi / potential.circular_frequency(
        potential.circular_radius(energies))
    velocity_edges = np.linspace(
        -options.vmax, options.vmax, options.velocity_bins + 1)
    recorder = _Recorder(
        options.orbits, field, kpc_per_arcsec, np.radians(inclination),
        np.array(options.shell_edges), velocity_edges, options.symmetries,
        options.seed)
    integrate(potential, positions, velocities, durations, options.accuracy,
              recorder)
    times, shells, losvds = recorder.result()
    return OrbitLibrary(
        positions=positions, velocities=velocities, prior=prior,
        shell_edges=np.array(options.shell_edges),
        shells=shells / times[:, None],
        velocity_edges=velocity_edges,
        losvds=losvds / times[:, None, None])


def _directions(rng: np.random.Generator, count: int) -> np.ndarray:
    """Return isotropic random unit vectors: an array of shape (count, 3)."""
    vectors = rng.normal(size=(count, 3))
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


class _Recorder:
    """
    The recorder of the orbits of a library (see integrate): it adds
    their samples, weighted by time, to the shells and, in symmetric
    copies projected on the sky and displaced by the PSF at random, to
    the spatial and velocity bins.
    """

    def __init__(
            self,
            orbits: int,
            field: BinnedField,
            kpc_per_arcsec: float,
            inclination: float,
            shell_edges: np.ndarray,
            velocity_edges: np.ndarray,
            symmetries: int,
            seed: int
    ):
        self._field = field
        self._nbins = field.nbins()
        self._scale = kpc_per_arcsec
        self._inclination = inclination
        self._shell_edges = shell_edges
        self._velocity_edges = velocity_edges
        self._symmetries = symmetries
        self._rng = np.random.default_rng(seed + 1)
        self._times = np.zeros(orbits)
        self._shells = np.zeros(orbits * (len(shell_edges) - 1))
        self._losvds = np.zeros(
            orbits * self._nbins * (len(velocity_edges) - 1))
        self._pending: list[tuple[np.ndarray, ...]] = []
        self._count = 0

    def __call__(
            self,
            orbits: np.ndarray,
            positions: np.ndarray,
            velocities: np.ndarray,
            weights: np.ndarray
    ) -> None:
        self._pending.append((orbits, positions, velocities, weights))
        self._count += len(orbits)
        if self._count >= _RECORD_BATCH:
            self._flush()

    def result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the times of the orbits (an array of shape (orbits,)), in
        each shell (orbits, shells), and in each spatial and velocity bin
        (orbits, bins, velocity bins).
        """
        self._flush()
        nshells = len(self._shell_edges) - 1
        nvelocities = len(self._velocity_edges) - 1
        return (self._times, self._shells.reshape(-1, nshells),
                self._losvds.reshape(-1, self._nbins, nvelocities))

    def _flush(self) -> None:
        """Record the pending samples."""
        if not self._pending:
            return
        orbits, x, v, weights = (
            np.concatenate(column) for column in zip(*self._pending))
        self._pending, self._count = [], 0
        self._times += np.bincount(
            orbits, weights, minlength=self._times.size)
        nshells = len(self._shell_edges) - 1
        shell = np.searchsorted(
            self._shell_edges, np.linalg.norm(x, axis=1), side='right') - 1
        inside = shell < nshells
        self._shells += np.bincount(
            orbits[inside] * nshells + shell[inside], weights[inside],
            minlength=self._shells.size)
        sky_x, sky_y, sky_v = self._project(
            np.tile(x, (self._symmetries, 1)),
            np.tile(v, (self._symmetries, 1)))
        orbits = np.tile(orbits, self._symmetries)
        weights = np.tile(weights / self._symmetries, self._symmetries)
        bins = self._field.bins_at(sky_x, sky_y)
        # (the velocity bins are even)
        edges = self._velocity_edges
        nvelocities = len(edges) - 1
        velocity_bin = np.clip(
            ((sky_v - edges[0]) / (edges[1] - edges[0])).astype(int),
            0, nvelocities - 1)
        seen = bins >= 0
        self._losvds += np.bincount(
            (orbits[seen] * self._nbins + bins[seen]) * nvelocities
            + velocity_bin[seen], weights[seen],
            minlength=self._losvds.size)

    def _project(
            self, x: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the sky positions (arcsec) and the line-of-sight velocities
        (km/s, positive away from the observer) of a symmetric copy of the
        samples: rotated about the z axis by random angles, reflected
        about its plane at random, and displaced by the PSF at random.
        The line of sight is in the y-z plane, at the inclination to the z
        axis; sky x is along the x axis.
        """
        count = len(x)
        angle = 2 * np.pi * self._rng.random(count, dtype=np.float32)
        flip = np.where(self._rng.random(count, dtype=np.float32) < 0.5,
                        -1.0, 1.0)
        cos, sin = np.cos(angle), np.sin(angle)
        rotated_y = x[:, 0] * sin + x[:, 1] * cos
        rotated_vy = v[:, 0] * sin + v[:, 1] * cos
        sky_x = x[:, 0] * cos - x[:, 1] * sin
        sky_y = rotated_y * np.cos(self._inclination) \
            + flip * x[:, 2] * np.sin(self._inclination)
        sky_v = rotated_vy * np.sin(self._inclination) \
            - flip * v[:, 2] * np.cos(self._inclination)
        width = self._field.psf_fwhm / np.sqrt(8 * np.log(2))
        offsets = width * self._rng.standard_normal(
            (2, count), dtype=np.float32)
        return (sky_x / self._scale + offsets[0],
                sky_y / self._scale + offsets[1], sky_v)
