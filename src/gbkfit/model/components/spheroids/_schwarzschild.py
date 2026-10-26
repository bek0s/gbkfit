"""
The Schwarzschild kinematics of a spheroid (SchwarzschildKinematics): its
stars on a library of orbits in the gravity of its galaxy, whose weights
are solved to fit the data (by an objective, see objective.Objective);
the state of their motion in each evaluation (OrbitLibraryState, see
orbit_library): the potential of the gravity, the starts of the orbits,
the share of each in the light, and the light the orbits must hold.
"""

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.integrate

from gbkfit.driver import Driver
from gbkfit.params import ParamDesc, ParamValues, ParamVectorDesc
from gbkfit.utils.parseutils import ConfigError
from .._native import ParamBuffer
from .._needs import Need, NeedsMass, NeedsPotential
from .._stage import GalaxyMass
from ..base import ComponentState
from ._jeans import light
from ._kinematics import SpheroidKinematics, SpheroidLight, SpheroidMotion
from ._orbits import PotentialTable, potential_table


__all__ = [
    'OrbitLibraryState',
    'SchwarzschildKinematics',
    'SchwarzschildMotion',
    'orbit_library'
]


# The table of the potential (see potential_table): its radii, geometric
# between rmax times these, their number, and the number of its angles
_TABLE_SPAN = (1e-6, 1e4)
_TABLE_RADII = 300
_TABLE_ANGLES = 17
# The radii of the starts of the orbits, geometric between the radii of
# the spheres that hold these shares of the light, and their largest
# speed, this share of the escape speed
_START_LIGHT = (1e-6, 0.999)
_START_ESCAPE = 0.95
# The cells of the light: shells of the radii geometric between those of
# the spheres that hold these shares of the light (and one within), their
# number, and the number of the bands of the angle from the plane in
# each, of equal solid angles
_CELL_LIGHT = (1e-4, 0.999)
_CELL_RADII = 20
_CELL_ANGLES = 5
# The nodes of the quadrature of the light of a cell along each axis, and
# the radius from which the inner shell is integrated (times its outer
# radius)
_CELL_NODES = 8
_CELL_INNER = 1e-6
# How the orbits are integrated and recorded (see OrbitIntegrator): the
# accuracy of the steps (radians), the records of each orbit in time, and
# its turned copies on the sky
_ACCURACY = 0.02
_SAMPLES = 2000
_COPIES = 4
# The most steps of an orbit, times those of its turns on a circle
_STEPS_FACTOR = 100


class SchwarzschildKinematics(SpheroidKinematics):
    """
    The stars of a spheroid in a steady state in the gravity of its galaxy,
    as a superposition of orbits (Schwarzschild 1979): a library of
    norbits orbits, started where its light is (each with a share of it,
    see orbit_library), each integrated for turns periods of the circular
    orbit of its energy, and recorded with its twin reversed in time
    (which turns the other way). The weights of the orbits fit the data:
    the parameter 'weights' holds the factor of the share of the light of
    each (2 norbits elements: orbit k at 2k, its twin at 2k + 1), which an
    objective solves (see objective.Objective), e.g. of a solve {type:
    gaussian, mean: 1, sigma: 1, minimum: 0, likelihood: profiled}: the
    prior of the factors (mean 1) regularises the weights towards the
    shares of the light (the smaller sigma, the stronger).

    The gravity must be the gradient of a potential (Newtonian), in which
    the orbits keep their energy. The objective models the observations
    of its stars, in the bins of region_moments of the method
    gauss_hermite (their V, sigma, h3, h4 ...).

    Parameters
    ----------
    norbits : int, optional
        The number of the orbits (each with its twin).
    turns : float, optional
        How long each orbit is integrated, in periods of the circular
        orbit of its energy.
    seed : int, optional
        The seed of the random numbers of the starts (the same starts in
        every evaluation, so that the orbits change smoothly with the
        gravity).

    Raises
    ------
    ConfigError
        If norbits is not a positive integer, turns not a positive number,
        or seed not a non-negative integer.
    """

    @staticmethod
    def type() -> str:
        return 'schwarzschild'

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(
            norbits=self._norbits, turns=self._turns, seed=self._seed)

    def __init__(self, norbits: int = 3000, turns: float = 100,
                 seed: int = 0):
        def integer(value: Any) -> bool:
            return isinstance(value, int) and not isinstance(value, bool)
        if not (integer(norbits) and norbits > 0):
            raise ConfigError(
                f"norbits must be a positive integer; it is {norbits!r}")
        if isinstance(turns, bool) or not isinstance(turns, (int, float)) \
                or not (math.isfinite(turns) and turns > 0):
            raise ConfigError(
                f"turns must be a positive number; it is {turns!r}")
        if not (integer(seed) and seed >= 0):
            raise ConfigError(
                f"seed must be a non-negative integer; it is {seed!r}")
        self._norbits = norbits
        self._turns = float(turns)
        self._seed = seed

    def norbits(self) -> int:
        """Return the number of the orbits (each with its twin)."""
        return self._norbits

    def turns(self) -> float:
        """Return how long each orbit is integrated (see the class)."""
        return self._turns

    def seed(self) -> int:
        """Return the seed of the random numbers of the starts."""
        return self._seed

    def pdescs(self) -> dict[str, ParamDesc]:
        return dict(weights=ParamVectorDesc(
            2 * self._norbits,
            desc="the factor of the share of the light of each orbit "
                 "(orbit k at 2k, its twin at 2k + 1), solved",
            minimum=0))

    def uses_mass(self) -> bool:
        return True

    def needs(self, label: str) -> tuple[Need, ...]:
        return (
            NeedsMass(f"{label} need the mass of its galaxy"),
            NeedsPotential(
                f"the orbits of {label} take forces that are the gradient "
                f"of a potential"))

    def motion(self, rmax: float) -> 'SchwarzschildMotion':
        return SchwarzschildMotion(self, rmax)


@dataclasses.dataclass(frozen=True)
class OrbitLibraryState(ComponentState):
    """
    The orbits of the stars of a spheroid of Schwarzschild kinematics in
    one evaluation (see orbit_library): the potential of the gravity of
    its galaxy, the start of each orbit (an array of shape (norbits, 6):
    x, y, z (arcsec) and vx, vy, vz (km/s), in the frame of the spheroid)
    and its share of the light of the spheroid (they sum to 1; its twin
    has the same), the edges of the cells of the spheroid (radii, arcsec,
    and angles from the plane, radians) and the light in each cell (of
    both sides of the plane; an array of shape (ncell_r, ncell_b)), how
    long the orbits are integrated (see SchwarzschildKinematics), the kpc
    of an arcsec at the galaxy, and how (see OrbitIntegrator): the
    accuracy of the steps, the records of each orbit in time and its
    turned copies on the sky.
    """
    potential: PotentialTable
    starts: np.ndarray
    shares: np.ndarray
    cell_edges: tuple[np.ndarray, np.ndarray]
    cell_light: np.ndarray
    turns: float
    kpc_per_arcsec: float
    accuracy: float = _ACCURACY
    samples: int = _SAMPLES
    copies: int = _COPIES

    def max_steps(self) -> int:
        """
        Return the most steps of an orbit: 100 times those of its turns on
        a circle.
        """
        return int(_STEPS_FACTOR * self.turns * 2 * math.pi / self.accuracy)

    def orbit_light(self, factors: npt.ArrayLike) -> np.ndarray:
        """
        Return the light of each orbit and of its twin (2k and 2k + 1), of
        the factors of their shares of the light (their weights, see
        SchwarzschildKinematics): the light of the cells times its share
        times its factor, half of it each.
        """
        return np.repeat(self.cell_light.sum() * self.shares / 2, 2) \
            * np.asarray(factors, dtype=float)


def orbit_library(
        kinematics: SchwarzschildKinematics,
        light_terms: SpheroidLight,
        rmax: float,
        mass: GalaxyMass
) -> OrbitLibraryState:
    """
    Return the orbits of the stars of a spheroid of Schwarzschild
    kinematics in one evaluation (see OrbitLibraryState), of the light of
    the spheroid (see Spheroid.light) and its rmax (arcsec), in the
    gravity of its galaxy.

    The potential is tabled from rmax times 1e-6 to 1e4. The orbits start
    at radii stratified evenly in ln r between those of the spheres that
    hold 1e-6 and 0.999 of the light, in random directions, each with the
    share of the light of its point (the density times r^3: the starts are
    spread evenly in ln r and in solid angle), and velocities in random
    directions of the isotropic Jeans dispersion of the light averaged
    over angles (a Gaussian along each axis), their speed at most 0.95
    times the escape speed of the radius. The cells are between the radii
    of the spheres that hold 1e-4 and 0.999 of the light (and one within:
    beyond, the orbits hold too little light for its share in a cell to
    count); their light is their integral, by Gauss-Legendre quadrature in
    ln r and sin(b).
    """
    rng = np.random.default_rng(kinematics.seed())
    count = kinematics.norbits()
    table = potential_table(
        mass.field, mass.kpc_per_arcsec,
        (rmax * _TABLE_SPAN[0], rmax * _TABLE_SPAN[1]),
        _TABLE_RADII, _TABLE_ANGLES)
    nodes = table.lnr_zero + table.lnr_step * np.arange(len(table.values))
    nu = _averaged_light(table, light_terms)
    lnr_light = _light_radii(nodes, nu)
    first, last = lnr_light(_START_LIGHT)
    lnr = first + (last - first) * (
        np.arange(count) + rng.random(count)) / count
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    normals = rng.normal(size=(count, 3))
    r = np.exp(lnr)
    positions = r[:, None] * directions
    density = light(
        *light_terms, np.hypot(positions[:, 0], positions[:, 1]),
        positions[:, 2])[0]
    shares = density * r ** 3
    sigma, escape = _start_speeds(table, nu, lnr)
    velocities = sigma[:, None] * normals
    speeds = np.linalg.norm(velocities, axis=1)
    velocities *= np.minimum(1, _START_ESCAPE * escape / speeds)[:, None]
    cell_edges = (
        np.concatenate(([0], np.exp(np.linspace(
            *lnr_light(_CELL_LIGHT), _CELL_RADII)))),
        np.arcsin(np.linspace(0, 1, _CELL_ANGLES + 1)))
    return OrbitLibraryState(
        table, np.hstack((positions, velocities)), shares / shares.sum(),
        cell_edges, _cell_light(light_terms, *cell_edges),
        kinematics.turns(), mass.kpc_per_arcsec)


def _angle_weights(table: PotentialTable) -> np.ndarray:
    """
    Return the weights of the averages over the angles of the nodes of a
    potential table, in solid angle.
    """
    weights = np.cos(np.linspace(0, np.pi / 2, table.values.shape[1]))
    return weights / weights.sum()


def _averaged_light(
        table: PotentialTable, light_terms: SpheroidLight
) -> np.ndarray:
    """
    Return the density of the light of a spheroid averaged over angles at
    the radii of a potential table.
    """
    nodes = table.lnr_zero + table.lnr_step * np.arange(len(table.values))
    b = np.linspace(0, np.pi / 2, table.values.shape[1])
    r = np.exp(nodes)[:, None]
    return light(*light_terms, r * np.cos(b), r * np.sin(b))[0] \
        @ _angle_weights(table)


def _light_radii(
        nodes: np.ndarray, nu: np.ndarray
) -> Callable[[Sequence[float]], np.ndarray]:
    """
    Return the function of the ln of the radii of the spheres that hold
    given shares of the light, of its density averaged over angles nu at
    the radii exp(nodes) (the light within the first left out).
    """
    enclosed = scipy.integrate.cumulative_trapezoid(
        4 * np.pi * nu * np.exp(3 * nodes), nodes, initial=0)
    shares, unique = np.unique(enclosed / enclosed[-1], return_index=True)

    def radii(fractions: Sequence[float]) -> np.ndarray:
        return np.interp(fractions, shares, nodes[unique])
    return radii


def _start_speeds(
        table: PotentialTable, nu: np.ndarray, lnr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the isotropic Jeans dispersion of a density of light averaged
    over angles nu at the radii of a potential table (sigma^2 nu =
    int_r^inf nu dPhi/d ln r d ln r), and the escape speed in the plane,
    at the radii exp(lnr).
    """
    nodes = table.lnr_zero + table.lnr_step * np.arange(len(table.values))
    slope = table.values[:, :, 1] @ _angle_weights(table)
    pressure = scipy.integrate.cumulative_trapezoid(
        (nu * slope)[::-1], -nodes[::-1], initial=0)[::-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        log_sigma2 = np.log(pressure / nu)
    finite = np.isfinite(log_sigma2)
    sigma = np.exp(np.interp(lnr, nodes[finite], log_sigma2[finite]) / 2)
    # (the potential in the plane, and its limit beyond the table)
    plane = table.values[:, 0, :2]
    limit = plane[-1, 0] + plane[-1, 1]
    escape = np.sqrt(2 * np.maximum(
        limit - np.interp(lnr, nodes, plane[:, 0]), 0))
    return sigma, escape


def _cell_light(
        light_terms: SpheroidLight, r_edges: np.ndarray, b_edges: np.ndarray
) -> np.ndarray:
    """
    Return the light of a spheroid in the cells between the radii r_edges
    (the first 0) and the angles from the plane b_edges, of both sides of
    the plane: 4 pi int int nu r^3 d(ln r) d(sin b) over each.
    """
    nodes, node_weights = np.polynomial.legendre.leggauss(_CELL_NODES)
    lnr_edges = np.log(np.concatenate(
        ([r_edges[1] * _CELL_INNER], r_edges[1:])))
    sin_edges = np.sin(b_edges)

    def points(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # (the nodes of each interval, and their weights)
        half = np.diff(edges)[:, None] / 2
        centre = edges[:-1, None] + half
        return centre + half * nodes, half * node_weights
    lnr, lnr_weights = points(lnr_edges)
    sin_b, sin_weights = points(sin_edges)
    r = np.exp(lnr)[:, :, None, None]
    density = light(
        *light_terms, r * np.sqrt(1 - sin_b ** 2), r * sin_b)[0]
    return 4 * np.pi * np.einsum(
        'ijkl,ij,kl->ik', density * r ** 3, lnr_weights, sin_weights)


class SchwarzschildMotion(SpheroidMotion):
    """
    The motion of the stars of a spheroid of SchwarzschildKinematics, of
    the given rmax: its orbits in each evaluation (its state, see
    orbit_library). It gives the native spheroid no kinematics: an
    objective models the observations of its stars (see
    SchwarzschildKinematics).
    """

    def __init__(self, kinematics: SchwarzschildKinematics, rmax: float):
        self._kinematics = kinematics
        self._rmax = rmax

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._kinematics.pdescs()

    def takes_light(self) -> bool:
        return True

    def orbit_weights(self) -> str | None:
        return 'weights'

    def state(
            self,
            values: ParamValues,
            light: SpheroidLight,
            mass: GalaxyMass | None
    ) -> OrbitLibraryState:
        if mass is None:
            raise ValueError(
                "the Schwarzschild kinematics of a spheroid need the mass of "
                "its galaxy")
        return orbit_library(self._kinematics, light, self._rmax, mass)

    def sizes(self) -> dict[str, int]:
        return {}

    def native(
            self, driver: Driver, dtype: np.dtype, buffer: ParamBuffer
    ) -> dict[str, Any]:
        return {}

    def write(
            self,
            buffer: ParamBuffer,
            params: ParamValues,
            state: ComponentState | None
    ) -> None:
        pass
