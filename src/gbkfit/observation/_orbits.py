"""
The recordings of orbits of a group plan (OrbitRecording): the orbits of
the stars of a spheroid whose weights are solved (see
Component.orbit_weights), recorded in the bins of the observations with
data that see them, whose models are linear in the weights there (see
OrbitSeries).
"""

import dataclasses
import logging
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from gbkfit.driver import Driver
from gbkfit.model import GalaxyState
from gbkfit.model.components.spheroids import (
    OrbitIntegrator, OrbitLibraryState, SkyGrid, Spheroid)
from gbkfit.params import ModelEvaluationError, ParamValues
from gbkfit.utils.parseutils import ConfigError
from .observables import ModelData

if TYPE_CHECKING:
    from .group import ObservationGroup
    from .sky import ObservationSky


_log = logging.getLogger(__name__)


__all__ = [
    'LinearMoment',
    'OrbitRecording',
    'OrbitSeries'
]


# The parameters of the frame of a spheroid whose orbits are recorded
# (see OrbitRecording.reads)
_FRAME_PARAMS = ('xpos', 'ypos', 'vsys', 'posa', 'incl')

# The relative error of the light of the spheroid in the bins and in the
# cells, which the orbits must hold (the constraints)
_LIGHT_ERROR = 0.01

# The integral of the light over each cell of a sky grid: Gauss-Legendre
# nodes along each axis of a square, and a square is split in four while
# that changes its light by more than the tolerance (relative to the
# brightest cell), at most to the depth
_CELL_NODES = 4
_CELL_TOLERANCE = 1e-5
_CELL_DEPTH = 12


class LinearMoment(NamedTuple):
    """
    A moment of the bins of an observation, linear in the weights f of
    the orbits that it sees (see OrbitSeries): base + slope (moments @ f),
    of the base and the slope of each bin, and the moments of the orbits
    in the series of each bin at weights 1 (an array of shape (nbins,
    norbits)).
    """
    base: np.ndarray
    slope: np.ndarray
    moments: np.ndarray


@dataclasses.dataclass(frozen=True)
class OrbitSeries:
    """
    The orbits of a recording in the series of the bins of its
    observations (see OrbitRecording.record), where their models are
    linear in the weights f of the orbits, and the constraints of the
    light of the orbits.

    Attributes
    ----------
    moments : dict
        Each moment of each observation (see LinearMoment), by key, by
        the index of the observation.
    light_rows, light_targets : np.ndarray
        The constraints of the light of the orbits in the bins and in the
        cells of the spheroid, of relative errors of 1%: their weighted
        residuals light_rows @ f - light_targets.
    """
    moments: dict[int, dict[str, LinearMoment]]
    light_rows: np.ndarray
    light_targets: np.ndarray


class OrbitRecording:
    """
    The orbits of the stars of a spheroid whose weights are solved (see
    Component.orbit_weights), recorded in the bins of the observations
    with data that see them (see GroupPlan), whose models are linear in
    the weights there (see OrbitSeries).

    At each recording (see record), the orbits of the state of the
    spheroid (see OrbitLibraryState) are integrated once, on the driver of
    its galaxy, and recorded in the bins of every observation through its
    PSF (see RegionMoments.sky_grid), in the Gauss-Hermite series of the
    data of each bin (its V and sigma; Rix et al. 1997): gamma h_m of
    each orbit, g_m. The model of a bin is then linear in the light of the
    orbits w_k (see OrbitLibraryState.orbit_light: their shares of the
    light times their weights f_k):

        V = V_d + sqrt(2) sigma_d g_1 / gamma
        sigma = sigma_d (1 + sqrt(2) g_2 / gamma)
        h_m = g_m / gamma    (m >= 3)

    of g_m = sum_k w_k g_m,k, gamma = L_b / N_b (L_b the light of the
    spheroid in the bin, its surface density through the shares of the
    bin, and N_b = 1 + sum over even m >= 4 of h_m sqrt(m!) / m!! of the
    data): to first order in g_1 and g_2, the series fitted to the model
    (van der Marel & Franx 1993), whose residuals are the constraints of
    Rix et al. It is the model of the observations (see model_d), which
    an objective fits (see objective.Objective). The light of the
    spheroid in the bins and in the cells (see OrbitLibraryState), with
    errors of 1%, constrains the weights.

    Parameters
    ----------
    group : ObservationGroup
        The observations.
    galaxy, component : int
        The index of the galaxy of the spheroid in the group, and its own.
    observations : Sequence of int
        The observations with data that see its stars.
    galaxy_driver : Driver
        The driver of the stage of its galaxy.
    observation_drivers : Sequence of Driver
        The driver of each of its observations.

    Raises
    ------
    ConfigError
        If an observation of it has a bin with some of its moments masked
        and not others, or with a sigma that is not positive (what its
        instrument and its scene may have is checked by its plan, see
        RegionMoments).
    """

    def __init__(
            self,
            group: 'ObservationGroup',
            galaxy: int,
            component: int,
            observations: Sequence[int],
            galaxy_driver: Driver,
            observation_drivers: Sequence[Driver]
    ):
        galaxy_ = group.galaxies()[galaxy]
        spheroid = galaxy_.components()[component]
        mapping = group.mappings()[galaxy]
        self._group = group
        self._galaxy = galaxy
        self._component = component
        self._spheroid = spheroid
        # (the names of the parameters of the spheroid that its records
        # read, its frame and its light, by its own names)
        self._names = {
            param: mapping[galaxy_.component_params(component)[param]]
            for param in _FRAME_PARAMS + spheroid.light_params()}
        self._weights = mapping[
            galaxy_.component_params(component)[spheroid.orbit_weights()]]
        self._observations = tuple(observations)
        self._galaxy_driver = galaxy_driver
        self._observation_drivers = tuple(observation_drivers)
        observed = [group.observations()[i] for i in observations]
        self._dtypes = tuple(observation.dtype() for observation in observed)
        self._data = tuple(
            _BinsData(observation, i)
            for observation, i in zip(observed, observations))
        self._order = max(data.order for data in self._data)
        self._grids = tuple(
            observation.observable().sky_grid(
                observation.instrument(), observation.oversampling())
            for observation in observed)
        # (and those of the offsets of the frames of its observations)
        self._reads = tuple(dict.fromkeys((*self._names.values(), *(
            name for observation in observed
            for name in observation.offset_params() if name is not None))))
        self._integrator: OrbitIntegrator | None = None
        # (whether it warned of orbits cut by the most steps: once)
        self._cut_warned = False

    def spheroid(self) -> Spheroid:
        """Return the spheroid."""
        return self._spheroid

    def weights(self) -> str:
        """Return the name of the parameter of the weights of its orbits."""
        return self._weights

    def observations(self) -> tuple[int, ...]:
        """Return the observations it models, by index."""
        return self._observations

    def component(self) -> tuple[int, int]:
        """
        Return the index of the galaxy of the spheroid in the group, and
        its own.
        """
        return self._galaxy, self._component

    def reads(self) -> tuple[str, ...]:
        """
        Return the names of the parameters that its records read (see
        record): those of the frame and the light of the spheroid (see
        Spheroid.light_params), and the offsets of the frames of its
        observations. The rest of the spheroid and the mass of its galaxy
        set its orbits, in the state of the galaxy; the weights of the
        orbits only weigh its records.
        """
        return self._reads

    def close(self) -> None:
        """
        Release the memory of its integrator; the next recording allocates
        it again.
        """
        self._integrator = None

    def record(
            self,
            params: ParamValues,
            states: Sequence[GalaxyState]
    ) -> OrbitSeries:
        """
        Integrate the orbits at the values of the parameters (those it
        reads among them, see reads), in the states of the galaxies (see
        GroupPlan.galaxy_states), and return them in the series of the
        bins of its observations.

        Raises
        ------
        ModelEvaluationError
            If bins see no light of the spheroid.
        """
        state = states[self._galaxy].states[self._component]
        if not isinstance(state, OrbitLibraryState):
            raise ValueError(
                f"the spheroid {self._spheroid.name()!r} has no orbits in "
                f"the state of its galaxy")
        values = {param: params[name] for param, name in self._names.items()}
        shifts = [
            self._group.frame_shifts(i, params) for i in self._observations]
        if self._integrator is None:
            self._integrator = OrbitIntegrator(
                self._galaxy_driver, self._dtypes[0], len(state.starts),
                accuracy=state.accuracy, max_steps=state.max_steps(),
                nsamples=state.samples, nphi=state.copies, twins=True,
                cells=state.cell_light.shape, sky=self._grids,
                order=self._order)

        # The orbits, in the series of the data of the bins, of the centre
        # and the systemic velocity of the spheroid in the frame of each
        centres = np.array([
            (values['xpos'] + x, values['ypos'] + y) for x, y, _ in shifts])
        series = tuple(np.concatenate(columns) for columns in zip(*(
            data.series(values['vsys'] + v)
            for data, (_, _, v) in zip(self._data, shifts))))
        records = self._integrator.integrate(
            state.potential, state.starts, state.turns,
            (values['posa'], values['incl']), state.cell_edges, centres,
            series)
        if not self._cut_warned \
                and (cut := int(np.sum(records.steps >= state.max_steps()))):
            # (an orbit cut by the most steps records its samples only up
            # to there, and loses the light of the rest of its time)
            _log.warning(
                f"{cut} orbits of the spheroid {self._spheroid.name()!r} "
                f"took the most steps of their integration "
                f"({state.max_steps()}) before the end of their time: the "
                f"light of the rest of it is left out")
            self._cut_warned = True

        # The moments of each observation, linear in the weights, and the
        # constraints of the light of the bins and the cells
        scales = state.orbit_light(1.0)
        moments = records.moments.astype(np.float64) * scales[:, None, None]
        bins_light = np.concatenate([
            _bins_light(self._spheroid, values, grid, centre)
            for grid, centre in zip(self._grids, centres)])
        if not np.all(bins_light > 0):
            raise ModelEvaluationError(
                f"bins see no light of the spheroid "
                f"{self._spheroid.name()!r}: the series of their stars has no "
                f"strength")
        linear = {}
        first = 0
        for data, i in zip(self._data, self._observations):
            bins = slice(first, first + data.nbins)
            linear[i] = data.moments(
                moments[:, bins], bins_light[bins] / data.norms)
            first += data.nbins
        cells = records.cells.reshape(len(scales), -1) * scales[:, None]
        cell_light = state.cell_light.ravel()
        held = cell_light > 1e-12 * cell_light.sum()
        rows = np.vstack((
            moments[:, :, 0].T / bins_light[:, None],
            cells[:, held].T / cell_light[held, None])) / _LIGHT_ERROR
        return OrbitSeries(linear, rows, np.full(len(rows), 1 / _LIGHT_ERROR))

    def model_d(
            self,
            series: OrbitSeries,
            index: int,
            params: ParamValues
    ) -> ModelData:
        """
        Return the model data of its observation of the given index, of
        its orbits in the series of the bins (see record) at their weights
        in the parameters, on the driver of the observation: its moments
        (NaN in the bins without a series, whose data are masked), without
        a mask or weights.
        """
        weights = np.asarray(params[self._weights], dtype=float)
        k = self._observations.index(index)
        driver, dtype = self._observation_drivers[k], self._dtypes[k]
        seen = self._data[k].seen
        return {
            key: dict(
                d=driver.mem_copy_h2d(np.ascontiguousarray(np.where(
                    seen,
                    moment.base + moment.slope * (moment.moments @ weights),
                    np.nan), dtype)),
                m=None, w=None)
            for key, moment in series.moments[index].items()}


class _BinsData:
    """
    The moments of the bins of an observation that sees stars on orbits
    (see OrbitRecording): their number, the orders 1 to M, the data of each
    order (by its key, 'moment{m}'; NaN where masked), the bins whose V
    and sigma are not masked (the others have no series: none of their
    moments), and the normalisation N_b of each.
    """

    def __init__(self, observation: 'ObservationSky', index: int):
        dataset = observation.data()
        orders = observation.observable().orders()
        self.order = len(orders)
        self.values = {
            f'moment{m}': np.asarray(dataset[f'moment{m}'].data(), float)
            for m in orders}
        masks = [np.asarray(dataset[f'moment{m}'].mask(), bool)
                 for m in orders]
        if not all(np.array_equal(mask, masks[0]) for mask in masks):
            raise ConfigError(
                f"the bins of observation {index}, of stars on orbits, have "
                f"all their moments or none (their series is of their V and "
                f"sigma)")
        self.seen = masks[0]
        self.nbins = len(self.seen)
        # (the series of a bin is about its sigma, which scales it: a sigma
        # of 0 made every evaluation NaN, with a message of no use)
        if not np.all(self.values['moment2'][self.seen] > 0):
            raise ConfigError(
                f"the bins of observation {index}, of stars on orbits, have "
                f"a positive sigma (the series of each bin is about its V "
                f"and sigma); mask those that do not")
        # (N_b = 1 + sum over even m >= 4 of h_m sqrt(m!) / m!!)
        self.norms = np.ones(self.nbins)
        for m in range(4, self.order + 1, 2):
            double = math.prod(range(m, 0, -2))
            self.norms += np.nan_to_num(self.values[f'moment{m}']) \
                * math.sqrt(math.factorial(m)) / double

    def series(self, vsys: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the mean (relative to the systemic velocity vsys) and the
        dispersion of the series of each bin (0 and 1 where they are
        masked).
        """
        return (np.where(self.seen, self.values['moment1'] - vsys, 0.0),
                np.where(self.seen, self.values['moment2'], 1.0))

    def moments(
            self, moments: np.ndarray, gamma: np.ndarray
    ) -> dict[str, LinearMoment]:
        """
        Return each moment of its bins, linear in the weights of the
        orbits (by key), of the moments of the orbits in its bins at
        weights 1 (of shape (norbits, nbins, order + 2)) and the strength
        gamma of the line of each bin: the order m is m0 + a g_m, of gamma
        h_m of the orbits g_m.
        """
        mean = np.where(self.seen, self.values['moment1'], 0.0)
        sigma = np.where(self.seen, self.values['moment2'], 0.0)
        parts = {1: (mean, np.sqrt(2) * sigma / gamma),
                 2: (sigma, np.sqrt(2) * sigma / gamma)}
        for m in range(3, self.order + 1):
            parts[m] = (np.zeros(self.nbins), 1 / gamma)
        return {
            f'moment{m}': LinearMoment(base, slope, moments[:, :, m + 1].T)
            for m, (base, slope) in parts.items()}


def _bins_light(
        spheroid: Spheroid,
        values: ParamValues,
        grid: SkyGrid,
        centre: np.ndarray
) -> np.ndarray:
    """
    Return the light of a spheroid at the values of its parameters in the
    targets of a sky grid (its surface density integrated over each cell,
    as the orbits record it, through the shares), of its centre in the
    sky of the grid.
    """
    cells = np.flatnonzero(np.diff(grid.shares.indptr))
    ny, nx = grid.shape
    inverse = np.linalg.inv(grid.cells[:, :2])
    area = abs(np.linalg.det(inverse))

    def density(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        x, y = inverse @ (np.stack((u, v)) - grid.cells[:, 2:]) \
            - centre[:, None]
        return spheroid.surface_density(values, x, y)

    light = _cells_integral(density, cells % nx, cells // nx) * area
    return grid.shares[cells].T @ light


def _cells_integral(
        function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        x0: np.ndarray,
        y0: np.ndarray
) -> np.ndarray:
    """
    Return the integral of a function of (x, y) over each unit square
    [x0, x0 + 1] x [y0, y0 + 1], adaptively (see _CELL_NODES).
    """
    nodes, weights = np.polynomial.legendre.leggauss(_CELL_NODES)
    nodes, weights = (nodes + 1) / 2, weights / 2

    def rule(x: np.ndarray, y: np.ndarray, size: float) -> np.ndarray:
        # (the integral over each square of the given corners and size)
        u = x[:, None, None] + size * nodes[None, :, None]
        v = y[:, None, None] + size * nodes[None, None, :]
        u, v = np.broadcast_arrays(u, v)
        f = function(u.ravel(), v.ravel()).reshape(u.shape)
        return size ** 2 * np.einsum('i,j,sij->s', weights, weights, f)

    integral = np.zeros(len(x0))
    squares = np.arange(len(x0))
    x, y = np.asarray(x0, dtype=float), np.asarray(y0, dtype=float)
    whole = rule(x, y, 1.0)
    brightest = np.max(np.abs(whole), initial=0.0)
    size = 1.0
    for depth in range(_CELL_DEPTH):
        size /= 2
        # (each square in four: the corners of its quarters)
        x = (x[:, None] + size * np.array([0, 1, 0, 1])).ravel()
        y = (y[:, None] + size * np.array([0, 0, 1, 1])).ravel()
        quarters = rule(x, y, size)
        split = quarters.reshape(-1, 4).sum(axis=1)
        # (a profile narrower than the nodes shows its peak only once
        # split, so the brightest light found so far sets the tolerance)
        brightest = max(brightest, np.max(np.abs(split), initial=0.0))
        done = np.abs(split - whole) <= _CELL_TOLERANCE * brightest
        if depth == _CELL_DEPTH - 1:
            done[:] = True
        np.add.at(integral, squares[done], split[done])
        kept = np.repeat(~done, 4)
        squares = np.repeat(squares, 4)[kept]
        x, y, whole = x[kept], y[kept], quarters[kept]
        if not len(squares):
            break
    return integral


def orbit_recordings(
        group: 'ObservationGroup',
        galaxy_drivers: Sequence[Driver],
        observation_drivers: Sequence[Driver]
) -> tuple[OrbitRecording, ...]:
    """
    Return the recordings of orbits of the observations of a group (see
    OrbitRecording), on the drivers of its galaxies and its observations:
    one for each component whose stars observations with data see on
    orbits (see Component.orbit_weights).

    Raises
    ------
    ConfigError
        If a recording cannot be made (see OrbitRecording).
    """
    seen: dict[tuple[int, int], list[int]] = {}
    for i, observation in enumerate(group.observations()):
        if observation.data() is not None:
            for key in group.orbit_components(i):
                seen.setdefault(key, []).append(i)
    return tuple(
        OrbitRecording(
            group, g, c, observations, galaxy_drivers[g],
            [observation_drivers[i] for i in observations])
        for (g, c), observations in seen.items())
