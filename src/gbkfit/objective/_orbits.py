"""
The orbit groups of an objective (OrbitGroup): the weights of the orbits
of the stars of a spheroid (see Component.orbit_weights), solved to fit
the observations that see them, which the groups model.
"""

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from gbkfit.driver import DeviceArray
from gbkfit.model import GalaxyState
from gbkfit.model.components.spheroids import (
    OrbitIntegrator, OrbitLibraryState, SkyGrid, Spheroid)
from gbkfit.observation import GroupPlan, ObservationGroup, ObservationSky
from gbkfit.params import ParamValues, Solution, Solve, solve_together_rows
from gbkfit.params._keys import element_name
from gbkfit.utils.parseutils import ConfigError
from .comparisons import Comparison


__all__ = [
    'OrbitGroup',
    'orbit_groups'
]


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


class OrbitGroup:
    """
    The weights of the orbits of the stars of a spheroid (see
    Component.orbit_weights), solved together to fit the moments of the
    bins of the observations that see them (see OrbitMomentsPlan), which
    it models (see Objective).

    At each evaluation, the orbits of the state of the spheroid (see
    OrbitLibraryState) are integrated once, on the driver of its galaxy,
    and recorded in the bins of every observation through its PSF (see
    RegionMoments.sky_grid), in the Gauss-Hermite series of the data of
    each bin (its V
    and sigma; Rix et al. 1997): gamma h_m of each orbit, g_m. The model
    of a bin is then linear in the light of the orbits w_k (see
    OrbitLibraryState.orbit_light: their shares of the light times the
    solved factors f_k, its weights):

        V = V_d + sqrt(2) sigma_d g_1 / gamma
        sigma = sigma_d (1 + sqrt(2) g_2 / gamma)
        h_m = g_m / gamma    (m >= 3)

    of g_m = sum_k w_k g_m,k, gamma = L_b / N_b (L_b the light of the
    spheroid in the bin, its surface density through the shares of the
    bin, and N_b = 1 + sum over even m >= 4 of h_m sqrt(m!) / m!! of the
    data): to first order in g_1 and g_2, the series fitted to the model
    (van der Marel & Franx 1993), whose residuals are the constraints of
    Rix et al. Its comparison gives the rows of the data (its residuals
    are c (model - data), see Comparison.check_pointwise); the light of
    the spheroid in the bins and in the cells (see OrbitLibraryState),
    with errors of 1%, gives the rows of constraints, which count with the
    prior of the weights (see solve_together_rows).

    Parameters
    ----------
    group : ObservationGroup
        The observations.
    galaxy, component : int
        The index of the galaxy of the spheroid in the group, and its own.
    observations : Sequence of int
        The observations that see its stars.
    comparisons : Sequence of Comparison
        The comparison of each observation of the group.
    elements : Sequence of str
        The names of the elements of its weights, in their order.
    solve : Solve
        Their solve.

    Raises
    ------
    ConfigError
        If an observation of it does not compare each value of its data
        with its model alone (see Comparison.check_pointwise), has an LSF
        (the kinematics of stars of pPXF have it taken out) or a PSF that
        varies, or a bin with some of its moments masked and not others.
    """

    def __init__(
            self,
            group: ObservationGroup,
            galaxy: int,
            component: int,
            observations: Sequence[int],
            comparisons: Sequence[Comparison],
            elements: Sequence[str],
            solve: Solve
    ):
        galaxy_ = group.galaxies()[galaxy]
        spheroid = galaxy_.components()[component]
        mapping = group.mappings()[galaxy]
        self._group = group
        self._galaxy = galaxy
        self._component = component
        self._spheroid = spheroid
        self._names = {
            param: mapping[name]
            for param, name in galaxy_.component_params(component).items()}
        self._observations = tuple(observations)
        self._comparisons = tuple(comparisons[i] for i in observations)
        self._elements = tuple(elements)
        self._solve = solve
        self._data = []
        for i, comparison in zip(observations, self._comparisons):
            observation = group.observations()[i]
            comparison.check_pointwise(observation.keys())
            instrument = observation.instrument()
            if instrument.lsf() is not None:
                raise ConfigError(
                    f"observation {i} sees the stars of the spheroid "
                    f"{spheroid.name()!r} on orbits: its instrument has no "
                    f"LSF (pPXF takes it out of their kinematics)")
            if instrument.psf() is not None and instrument.psf().varies():
                raise ConfigError(
                    f"observation {i} sees the stars of the spheroid "
                    f"{spheroid.name()!r} on orbits: its PSF does not vary")
            self._data.append(_BinsData(observation, i))
        self._order = max(data.order for data in self._data)
        self._grids = tuple(
            observation.observable().sky_grid(
                observation.instrument(), observation.oversampling())
            for observation in (group.observations()[i] for i in observations))
        self._integrator: OrbitIntegrator | None = None
        self._ones: list[dict[str, dict[str, DeviceArray | None]]] = []
        # The values of the parameters of its last evaluation, and its
        # solution and models then
        self._last: tuple[ParamValues, Solution, dict] | None = None

    def observations(self) -> tuple[int, ...]:
        """Return the observations it models."""
        return self._observations

    def elements(self) -> tuple[str, ...]:
        """Return the names of the elements of its weights."""
        return self._elements

    def close(self) -> None:
        """Release its memory; the next evaluation allocates it again."""
        self._integrator = None
        self._ones = []
        self._last = None

    def evaluate(
            self,
            params: ParamValues,
            states: Sequence[GalaxyState],
            group_plan: GroupPlan
    ) -> tuple[Solution, dict[int, dict[str, dict[str, DeviceArray | None]]]]:
        """
        Solve the weights at the values of the parameters, in the states of
        the galaxies (see GroupPlan.galaxy_states), and return them and the
        model data of each of its observations, by index, on its driver:
        those of its last evaluation, if the parameters have the same
        values.
        """
        if self._last is not None and _same(self._last[0], params):
            return self._last[1], self._last[2]
        state = states[self._galaxy].states[self._component]
        if not isinstance(state, OrbitLibraryState):
            raise ValueError(
                f"the spheroid {self._spheroid.name()!r} has no orbits in "
                f"the state of its galaxy")
        names = self._names
        values = {param: params[name] for param, name in names.items()}
        moved = [self._group.moved(i, params) for i in self._observations]
        if self._integrator is None:
            self._prepare(state, group_plan)
        integrator = self._integrator

        # The orbits, in the series of the data of the bins
        centres = np.array([
            (m[names['xpos']], m[names['ypos']]) for m in moved])
        series = tuple(np.concatenate(columns) for columns in zip(*(
            data.series(m[names['vsys']])
            for data, m in zip(self._data, moved))))
        records = integrator.integrate(
            state.potential, state.starts, state.turns,
            (values['posa'], values['incl']), state.cell_edges, centres,
            series)

        # The rows of the factors of the orbits: of the data (the
        # residuals of the comparisons), and of the light of the bins and
        # the cells (constraints)
        scales = state.orbit_light(1.0)
        moments = records.moments.astype(np.float64) * scales[:, None, None]
        bins_light = np.concatenate([
            _bins_light(self._spheroid, values, grid, centre)
            for grid, centre in zip(self._grids, centres)])
        if not np.all(bins_light > 0):
            raise ValueError(
                f"bins see no light of the spheroid "
                f"{self._spheroid.name()!r}: the series of their stars has no "
                f"strength")
        data_rows, data_targets, slices = [], [], []
        first = 0
        for data, comparison, ones, i in zip(
                self._data, self._comparisons, self._ones,
                self._observations):
            comparison.evaluate(ones, params, True, {})
            factors = {
                key: group_plan.observation_drivers()[i].mem_copy_d2h(
                    residual).astype(np.float64)
                for key, residual in comparison.residual_nddata_d().items()}
            gamma = bins_light[first:first + data.nbins] / data.norms
            slices.append((slice(first, first + data.nbins), gamma))
            rows, targets = data.rows(
                moments[:, first:first + data.nbins], gamma, factors)
            data_rows.append(rows)
            data_targets.append(targets)
            first += data.nbins
        cells = records.cells.reshape(len(scales), -1) * scales[:, None]
        cell_light = state.cell_light.ravel()
        held = cell_light > 1e-12 * cell_light.sum()
        constraints = np.vstack((
            moments[:, :, 0].T / bins_light[:, None],
            cells[:, held].T / cell_light[held, None])) / _LIGHT_ERROR
        rows = np.vstack((*data_rows, constraints))
        targets = np.concatenate((
            *data_targets, np.full(len(constraints), 1 / _LIGHT_ERROR)))
        ndata = len(rows) - len(constraints)
        solution = solve_together_rows(
            [self._solve], [len(scales)], rows, targets, ndata)
        residuals = rows[ndata:] @ solution.values - targets[ndata:]
        solution = Solution(
            solution.values,
            np.concatenate((residuals, solution.residuals)),
            solution.log_term, solution.m_eff)

        # The model data of the observations
        sums = np.einsum('ktm,k->tm', moments, solution.values)
        models = {}
        for data, (slice_, gamma), i in zip(
                self._data, slices, self._observations):
            observation_driver = group_plan.observation_drivers()[i]
            models[i] = {
                key: dict(
                    d=observation_driver.mem_copy_h2d(np.ascontiguousarray(
                        model, self._group.observations()[i].dtype())),
                    m=None, w=None)
                for key, model in data.model(sums[slice_], gamma).items()}
        self._last = (dict(params), solution, models)
        return solution, models

    def _prepare(
            self, state: OrbitLibraryState, group_plan: GroupPlan
    ) -> None:
        """
        Allocate its memory: its integrator, on the driver of the galaxy,
        and the models of the data plus 1 of each observation (whose
        residuals are the factors of its comparison).
        """
        dtype = self._group.observations()[self._observations[0]].dtype()
        self._integrator = OrbitIntegrator(
            group_plan.galaxy_drivers()[self._galaxy], dtype,
            len(state.starts), accuracy=state.accuracy,
            max_steps=state.max_steps(), nsamples=state.samples,
            nphi=state.copies, twins=True, cells=state.cell_light.shape,
            sky=self._grids, order=self._order)
        self._ones = []
        for data, i in zip(self._data, self._observations):
            driver = group_plan.observation_drivers()[i]
            dtype = self._group.observations()[i].dtype()
            self._ones.append({
                key: dict(d=driver.mem_copy_h2d(
                    (np.nan_to_num(value) + 1).astype(dtype)), m=None, w=None)
                for key, value in data.values.items()})


class _BinsData:
    """
    The moments of the bins of an observation that sees stars on orbits
    (see OrbitGroup): their number, the orders 1 to M, the data of each
    order (by its key, 'moment{m}'; NaN where masked), the bins whose V
    and sigma are not masked (the others have no series: none of their
    moments), and the normalisation N_b of each.
    """

    def __init__(self, observation: ObservationSky, index: int):
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

    def _parts(
            self, gamma: np.ndarray
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """
        Return the model of each order m as m0 + a g_m, of gamma h_m of the
        orbits g_m: m0 and a, by m.
        """
        mean = np.where(self.seen, self.values['moment1'], 0.0)
        sigma = np.where(self.seen, self.values['moment2'], 0.0)
        parts = {1: (mean, np.sqrt(2) * sigma / gamma),
                 2: (sigma, np.sqrt(2) * sigma / gamma)}
        for m in range(3, self.order + 1):
            parts[m] = (np.zeros(self.nbins), 1 / gamma)
        return parts

    def rows(
            self,
            moments: np.ndarray,
            gamma: np.ndarray,
            factors: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the rows of its residuals in the weights of the orbits and
        their targets, of the moments of the orbits in its bins (scaled
        by their light: of shape (norbits, nbins, order + 2)), the
        strength gamma of the line of each bin, and the factors of the
        residuals of each key (see Comparison.check_pointwise).
        """
        rows, targets = [], []
        for m, (base, slope) in self._parts(gamma).items():
            key = f'moment{m}'
            factor = factors[key]
            rows.append((factor * slope)[:, None] * moments[:, :, m + 1].T)
            targets.append(factor * (np.nan_to_num(self.values[key]) - base))
        return np.vstack(rows), np.concatenate(targets)

    def model(
            self, sums: np.ndarray, gamma: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Return the model of each key, of the sums over the weighted orbits
        of their moments in its bins (of shape (nbins, order + 2)) and the
        strength gamma of the line of each bin.
        """
        return {
            f'moment{m}': base + slope * sums[:, m + 1]
            for m, (base, slope) in self._parts(gamma).items()}


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


def _same(first: ParamValues, second: ParamValues) -> bool:
    """Check whether two sets of values of parameters are the same."""
    return first.keys() == second.keys() and all(
        np.array_equal(first[name], second[name]) for name in first)


def orbit_groups(
        group: ObservationGroup,
        comparisons: Sequence[Comparison],
        solves: Mapping[str, Solve]
) -> tuple[OrbitGroup, ...]:
    """
    Return the orbit groups of the observations of a group (see
    OrbitGroup), of the comparison of each, and the solve of each solved
    element (by element name): one for each component whose stars the
    observations see on orbits (see Component.orbit_weights).

    Raises
    ------
    ConfigError
        If the weights of the orbits of a component are not solved, or
        not by one solve, or an orbit group cannot be made (see
        OrbitGroup).
    """
    seen: dict[tuple[int, int], list[int]] = {}
    for i in range(group.nobservations()):
        for key in group.orbit_components(i):
            seen.setdefault(key, []).append(i)
    groups = []
    for (g, c), observations in seen.items():
        galaxy = group.galaxies()[g]
        component = galaxy.components()[c]
        param = group.mappings()[g][
            galaxy.component_params(c)[component.orbit_weights()]]
        elements = [
            element_name(param, k)
            for k in range(group.pdescs()[param].size())]
        if not all(element in solves for element in elements):
            raise ConfigError(
                f"the weights of the orbits of the stars of "
                f"{component.name()!r} ({param!r}) are solved to fit the "
                f"data: give them a solve, e.g. {param}: {{solve: {{type: "
                f"gaussian, mean: 1, sigma: 1, minimum: 0, likelihood: "
                f"profiled}}}}")
        solve_set = {id(solves[element]) for element in elements}
        if len(solve_set) > 1:
            raise ConfigError(
                f"the weights of the orbits {param!r} are solved by one "
                f"solve")
        groups.append(OrbitGroup(
            group, g, c, observations, comparisons, elements,
            solves[elements[0]]))
    return tuple(groups)
