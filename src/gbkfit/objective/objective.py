from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Self

import numpy as np

from gbkfit.dataset import Dataset
from gbkfit.driver import DeviceArray, Driver
from gbkfit.model import Extras
from gbkfit.objective.comparisons import Comparison, comparison
from gbkfit.observation import FluxScale, GroupPlan, ObservationGroup
from gbkfit.params import (
    ParamDesc, ParamValues, Solution, Solve, check_together, solve_together)
from gbkfit.params._keys import element_name
from gbkfit.utils import threadutils
from gbkfit.utils.parseutils import ConfigError
from ._orbits import orbit_groups


__all__ = [
    'Merits',
    'Objective'
]


# The most threads of BLAS in the solves of the weights of orbits (see
# Objective._solve_orbits): more lose to their overheads (an AICp solve
# of 20000 weights on 1, 4, 8 and 22 threads of a 16-core laptop: 0.84,
# 0.48, 0.52 and 2.1 s for 305 rows; 29, 12.4, 12.8 and 18.7 s for 2495)
_ORBIT_BLAS_THREADS = 4


class Merits(NamedTuple):
    """
    The numbers of an evaluation of an objective by which fitters judge
    the models (see Objective.merits).

    Attributes
    ----------
    chi2 : float
        The chi-square (see Objective.chi2).
    log_likelihood : float
        The log-likelihood (see Objective.log_likelihood).
    aicp : float
        The AICp (see Objective.aicp).
    """
    chi2: float
    log_likelihood: float
    aicp: float


class _Solved(NamedTuple):
    """
    Solved elements of the flux scale of an observation (see Solve): the
    observation (its index), the parameter of the scale, the indices of
    the elements (None for a scalar), how the scale scales the model of
    the observation, and how the elements are solved.
    """
    observation: int
    param: str
    indices: np.ndarray | None
    scale: FluxScale
    solve: Solve


class _Group(NamedTuple):
    """
    Solved amplitudes of the light of the models (see
    Component.light_amplitudes), solved together (see solve_together):
    the elements (each the name of its parameter and its index, None for
    a scalar), in runs of one solve (the solve and the number of its
    elements), and the observations whose models take them.
    """
    elements: tuple[tuple[str, int | None], ...]
    runs: tuple[tuple[Solve, int], ...]
    observations: tuple[int, ...]


class Objective:
    """
    The comparison of the models of a group of observations with their
    data, each under its likelihood, on the drivers of a plan of the
    group (see ObservationGroup.plan).

    Each observation has a comparison of its kind of likelihood (see
    Comparison): its residuals, of each value of its data, and the terms
    of the chi-square beyond them (e.g. the noise residuals of a fitted
    noise; see LikelihoodGaussian). The objective sums them.

    The elements of parameters can be solved at each evaluation, rather
    than fitted (see Solve and ParamSpace): those of the flux scales of
    observations (see ObservationSky), and the amplitudes of the light of
    components (see Component.light_amplitudes), which the model is linear
    in. A flux scale: the model is evaluated with it at 1, the comparison
    of its observation gives its sums, its solve its values, and the
    comparison then compares the model times them (see
    Comparison.evaluate). Amplitudes of the light, solved together where
    their parts of the model fall on the same data (in one or several
    observations; see solve_together): the models are evaluated with them
    at 0 and with each at 1 alone; the weighted residuals of the
    comparisons, linear in the model, give the sums of their parts
    (r(m_k) - r(m_0) and -r(m_0)), and the models are evaluated at their
    values. They do not go with a solved flux scale of the same
    observation (the model is then not linear in both). The weights of the
    orbits of the stars of a spheroid (see Component.orbit_weights): the
    objective models the observations that see the stars (they have no
    model of their own), from the orbits, linear in their weights, which
    it solves together (see OrbitGroup). The priors of the solved values
    add residuals after those of the data; the log-likelihood has the
    values integrated out, or at their best values (see Solve). They are
    nuisance parameters (see nnuisance); solved gives their values.

    Its memory on the devices (and the host) is allocated on the first
    evaluation, and released by close, or when Python collects it. An
    objective is a context manager that closes on exit: fitters can keep
    it alive after a fit (e.g. lmfit, in reference cycles), with its
    memory, until Python collects it.

    The methods for fitters return totals over all observations and new
    arrays (chi2, log_likelihood, aicp, merits, residuals). The others
    return a value or the arrays of each observation, which are
    overwritten by the next evaluation.

    Parameters
    ----------
    group_plan : GroupPlan
        The plan of the observations, each with data.
    solves : Mapping, optional
        The solve of each solved element of the parameters, by element
        name (see ParamSpace.solves); none by default.

    Raises
    ------
    ConfigError
        If an observation has no data, its comparison cannot be made (see
        Comparison), or a solved element is neither of the flux scale of an
        observation whose comparison can solve it (see
        Comparison.check_scales), nor an amplitude of the light that the
        models are linear in (see Component.light_amplitudes,
        Observable.linear_in_light and Comparison.check_linear), nor a
        weight of orbits; or the weights of the orbits of stars that
        observations see are not solved (see OrbitGroup).
    """

    def __init__(
            self,
            group_plan: GroupPlan,
            solves: Mapping[str, Solve] | None = None
    ):
        group = group_plan.group()
        observations = group.observations()
        if missing := [
                i for i, obs in enumerate(observations) if obs.data() is None]:
            raise ConfigError(
                f"every observation of an objective needs data; these have "
                f"none: {missing}")
        self._group_plan = group_plan
        self._group = group
        self._datasets = tuple(obs.data() for obs in observations)
        self._comparisons = tuple(
            comparison(obs, driver) for obs, driver in zip(
                observations, group_plan.observation_drivers()))
        self._solves = dict(solves or {})
        self._orbits = orbit_groups(group, self._comparisons, self._solves)
        orbit_elements = {
            element for orbit in self._orbits for element in orbit.elements()}
        self._solved, amplitudes = _solved(
            group, self._comparisons, {
                name: solve for name, solve in self._solves.items()
                if name not in orbit_elements})
        # The observations with solved scales
        self._scaled = sorted({solved.observation for solved in self._solved})
        self._groups = _groups(
            group, self._comparisons, amplitudes, self._scaled)
        self._reset_memory()

    def _reset_memory(self) -> None:
        """
        Drop its memory (see prepare and close): the next evaluation
        allocates it again.
        """
        for comparison_ in self._comparisons:
            comparison_.close()
        for orbit in self._orbits:
            orbit.close()
        # The values of the solved elements of the last evaluation (of the
        # flux scales, the groups of amplitudes and the orbits), and
        # the scales of the observations with some, on the host (h) and
        # the device (d), by index
        self._solutions: list[Solution | None] = [None] * (
            len(self._solved) + len(self._groups) + len(self._orbits))
        self._h_scales: dict[int, np.ndarray] = {}
        self._d_scales: dict[int, DeviceArray] = {}
        # The weighted residuals of the models of each group, at 0 and with
        # each of its elements at 1 alone (and then their parts, see
        # _solve_groups), of each of its observations, in float64 on its
        # device, and their sums (host and device), by group and
        # observation
        self._d_rows: dict[tuple[int, int], DeviceArray] = {}
        self._sums: dict[tuple[int, int], tuple[
            np.ndarray, DeviceArray, np.ndarray, DeviceArray]] = {}
        # The memory is allocated on the first evaluation (see prepare)
        self._prepared = False

    def nobservations(self) -> int:
        """Return the number of observations."""
        return self._group.nobservations()

    def datasets(self) -> tuple[Dataset, ...]:
        """Return the data of each observation."""
        return self._datasets

    def group(self) -> ObservationGroup:
        """Return the observations."""
        return self._group

    def group_plan(self) -> GroupPlan:
        """Return the plan of the observations."""
        return self._group_plan

    def with_driver(
            self,
            driver: Driver,
            drivers: Mapping[str, Driver] | None = None
    ) -> 'Objective':
        """
        Return a copy of it with its work on other drivers: its models
        (see ObservationGroup.plan, which takes the drivers) and their
        residuals. The copy shares the data; its memory is its own.
        Copies on several devices evaluate several points at once (see
        EvaluationPool).
        """
        return Objective(self._group.plan(driver, drivers), self._solves)

    def with_data(self, datasets: Sequence[Dataset]) -> 'Objective':
        """
        Return a copy of it with other data of its observations (e.g. those
        of a bootstrap), whose plan shares what it can with its own (see
        GroupPlan.with_data). The comparisons of the copy share what they
        can with its own (see Comparison.share): e.g. the refits of a
        bootstrap work out the whitening of correlated noise once.
        """
        objective = Objective(
            self._group_plan.with_data(datasets), self._solves)
        for new, old in zip(objective._comparisons, self._comparisons):
            new.share(old)
        return objective

    def close(self) -> None:
        """
        Release its memory on the devices and the host now, rather than
        when Python collects it (e.g. hundreds of megabytes of the whitening
        of correlated noise; see LikelihoodGaussian). The next evaluation
        allocates it again.
        """
        self._reset_memory()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def pdescs(self) -> dict[str, ParamDesc]:
        """Return the parameters of the models, by name (prefixed)."""
        return self._group.pdescs()

    def constants(self) -> dict[str, Any]:
        """Return the constants of the models, by name (prefixed)."""
        return self._group.constants()

    def prepare(self) -> None:
        """
        Allocate its memory, and prepare the comparisons (see
        Comparison.prepare; e.g. they work out the whitening of correlated
        noise). The first evaluation calls it; it does nothing once done
        (until close).
        """
        if self._prepared:
            return
        for comparison_ in self._comparisons:
            comparison_.prepare()
        for i in self._scaled:
            observation = self._group.observations()[i]
            _, scale = observation.flux_scale()
            shape = tuple(
                1 if scale.axis in (None, k) else n
                for k, n in enumerate(self._datasets[i].shape()))
            driver = self._group_plan.observation_drivers()[i]
            (self._h_scales[i],
             self._d_scales[i]) = driver.mem_alloc_s(
                shape, observation.dtype())
        drivers = self._group_plan.observation_drivers()
        for g, group in enumerate(self._groups):
            size = len(group.elements)
            for i in group.observations:
                driver = drivers[i]
                nvalues = self._comparisons[i].residual_vector_d().size
                self._d_rows[g, i] = driver.mem_alloc_d(
                    (size + 1, nvalues), np.float64)
                self._sums[g, i] = (
                    *driver.mem_alloc_s((size, size), np.float64),
                    *driver.mem_alloc_s(size, np.float64))
        self._prepared = True

    def chi2(
            self,
            params: ParamValues,
            out_extra: Extras | None = None
    ) -> float:
        """
        Return the chi-square of all observations: the sum of their squared
        weighted residuals (with those of the priors of solved values and
        the noise residuals, if any; see residuals).

        Parameters
        ----------
        params : ParamValues
            The parameters of the models, prefixed (see pdescs).
        out_extra : Extras, optional
            Where the extra outputs of the models go, if wanted.

        Returns
        -------
        float
            The chi-square.
        """
        return self.merits(params, out_extra).chi2

    def log_likelihood(
            self,
            params: ParamValues,
            out_extra: Extras | None = None
    ) -> float:
        """
        Return the log-likelihood of all observations: -chi2 / 2, and
        with solved values, their integral (see Solve). The
        part of the Gaussian normalisation that depends only on the errors
        of the data is left out (with a fitted noise, the rest is in the
        chi-square, as its noise residuals).

        Parameters
        ----------
        params : ParamValues
            The parameters of the models, prefixed (see pdescs).
        out_extra : Extras, optional
            Where the extra outputs of the models go, if wanted.

        Returns
        -------
        float
            The log-likelihood.
        """
        return self.merits(params, out_extra).log_likelihood

    def aicp(
            self,
            params: ParamValues,
            out_extra: Extras | None = None
    ) -> float:
        """
        Return the AICp of all observations (Lipka & Thomas 2021): the
        chi-square of the data plus twice the effective number of the
        solved values (see nnuisance), without the residuals of their
        priors, which are not of the data (with the noise residuals of a
        fitted noise, which are). Of models whose solved values are
        regularised by their priors (e.g. the weights of orbits), the
        lower, the better: the chi-square alone prefers the weakest
        regularisation, which fits the noise. Without solved values, it is
        the chi-square.

        Parameters
        ----------
        params : ParamValues
            The parameters of the models, prefixed (see pdescs).
        out_extra : Extras, optional
            Where the extra outputs of the models go, if wanted.

        Returns
        -------
        float
            The AICp.
        """
        return self.merits(params, out_extra).aicp

    def merits(
            self,
            params: ParamValues,
            out_extra: Extras | None = None
    ) -> Merits:
        """
        Return the chi-square, the log-likelihood and the AICp of all
        observations, of one evaluation (see chi2, log_likelihood and
        aicp).

        Parameters
        ----------
        params : ParamValues
            The parameters of the models, prefixed (see pdescs).
        out_extra : Extras, optional
            Where the extra outputs of the models go, if wanted.

        Returns
        -------
        Merits
            The numbers.
        """
        self._update_residual_d(params, True, out_extra)
        with self._group_plan.timers().measure('objective_residual_sum_eval'):
            data_chi2 = float(sum(
                comparison_.chi2() for comparison_ in self._comparisons))
        chi2 = data_chi2 + float(sum(
            np.sum(residuals ** 2) for residuals in self._solve_residuals()))
        log_likelihood = -0.5 * chi2 + float(sum(
            solution.log_term for solution in self._solutions))
        return Merits(chi2, log_likelihood, data_chi2 + 2 * self.nnuisance())

    def residuals(
            self,
            params: ParamValues,
            out_extra: Extras | None = None
    ) -> np.ndarray:
        """
        Return the weighted residuals of all observations, one after the
        other, in a new array (in float64; see residual_vector_h for each
        observation), after them the residuals of the priors of the solved
        values (see Solve), and then the noise residuals of each
        observation (see LikelihoodGaussian), if any.

        Parameters
        ----------
        params : ParamValues
            The parameters of the models, prefixed (see pdescs).
        out_extra : Extras, optional
            Where the extra outputs of the models go, if wanted.

        Returns
        -------
        np.ndarray
            The residuals (a new array on every call: solvers that keep
            the residuals of earlier calls, e.g. scipy's least_squares,
            need one).
        """
        self._check_residuals()
        vectors = self.residual_vector_h(params, True, out_extra)
        noise = [c.noise_residuals() for c in self._comparisons]
        return np.concatenate(
            [*vectors, *self._solve_residuals(), *noise], dtype=np.float64)

    def pointwise_log_likelihood(self, params: ParamValues) -> np.ndarray:
        """
        Return the log-likelihood of each data point of all observations
        that counts (see Comparison.counted; not the values the data mask),
        in the order of residuals: -r^2 / 2 of its weighted residual r, so
        that they sum to the log-likelihood (see log_likelihood); for
        comparisons of models that leave points out (e.g. analysis.loo).
        With correlated noise, the points are the components of the
        whitening, not the pixels.

        Parameters
        ----------
        params : ParamValues
            The parameters of the models, prefixed (see pdescs).

        Returns
        -------
        np.ndarray
            The log-likelihood of each point.

        Raises
        ------
        NotImplementedError
            If an observation fits its noise, or values are solved (see
            Solve), whose terms are sums over many points.
        """
        if self._solutions or any(
                c.noise_residuals().size for c in self._comparisons):
            raise NotImplementedError(
                "the log-likelihood of each point is not given with a fitted "
                "noise or solved values (e.g. flux scales): their terms are "
                "sums over many points")
        counted = np.concatenate([c.counted() for c in self._comparisons])
        return -0.5 * self.residuals(params)[counted] ** 2

    def noise_chi2(self) -> float:
        """
        Return the sum of the squared noise residuals of all observations
        (see LikelihoodGaussian), of the last evaluation: the part of the
        chi-square that is not of the data (e.g. for a reduced
        chi-square); 0 without a fitted noise.
        """
        return float(sum(
            np.sum(c.noise_residuals() ** 2) for c in self._comparisons))

    def solves(self) -> dict[str, Solve]:
        """Return the solve of each solved element, by element name."""
        return dict(self._solves)

    def solved(self) -> dict[str, float]:
        """
        Return the values of the solved elements of the last evaluation,
        by element name (see ParamSpace.evaluate); none before one.
        """
        values = {}
        for solved, solution in zip(self._solved, self._solutions):
            if solution is None:
                continue
            names = [solved.param] if solved.indices is None else [
                element_name(solved.param, int(k)) for k in solved.indices]
            values.update(zip(names, solution.values.tolist()))
        for group, solution in zip(
                self._groups, self._solutions[len(self._solved):]):
            if solution is None:
                continue
            values.update(zip(
                (element_name(*element) for element in group.elements),
                solution.values.tolist()))
        first = len(self._solved) + len(self._groups)
        for orbit, solution in zip(self._orbits, self._solutions[first:]):
            if solution is not None:
                values.update(zip(orbit.elements(), solution.values.tolist()))
        return values

    def nnuisance(self) -> float:
        """
        Return the effective number of the nuisance parameters that the
        objective solves itself (see Solve), of the last evaluation: how
        much the data decide the solved elements. The degrees of freedom
        of a fit exclude them.

        Raises
        ------
        RuntimeError
            If it has solved elements and no evaluation yet.
        """
        if any(solution is None for solution in self._solutions):
            raise RuntimeError(
                "the effective number of the solved elements is of an "
                "evaluation: evaluate the objective first")
        # Each solved value counts by how much its data decide it (its
        # m_eff: 1 decided by the data, 0 by its prior or at its minimum),
        # not as 1: a value its data decide uses up a degree of freedom,
        # and the residual of its prior, about 0 then, is not a residual
        # of the data (see nresiduals). Counting it as 1 with its prior
        # residual as a residual (net 0) left the degrees of freedom too
        # large by about 1 for each value the data decide.
        return float(sum(solution.m_eff for solution in self._solutions))

    def nresiduals(self) -> int:
        """
        Return the number of residuals of the data that count: those of the
        comparisons (see Comparison.nresiduals); not the residuals of the
        priors of solved values (see nnuisance), nor the noise residuals
        (see LikelihoodGaussian), which are not of the data.
        """
        return sum(c.nresiduals() for c in self._comparisons)

    def nindependent(self) -> float:
        """
        Return the number of independent residuals that the residuals are
        worth (see nresiduals): the chi-square that the noise of the data
        gives them on average, one for each that counts, but those of the
        data items whose noise is correlated, which are worth fewer (see
        Whitening). The degrees of freedom of a fit are of them. It works
        out the whitening of the residuals, if not yet (see prepare).
        """
        self.prepare()
        return sum(c.nindependent() for c in self._comparisons)

    def _solve_residuals(self) -> list[np.ndarray]:
        """
        Return the residuals of the priors of the solved values of the last
        evaluation, of each solve.
        """
        return [solution.residuals for solution in self._solutions]

    def gives_residuals(self) -> bool:
        """
        Check whether the comparisons of all its observations give
        residuals (see Comparison.gives_residuals), which least squares
        and covariances need.
        """
        return all(c.gives_residuals() for c in self._comparisons)

    def _check_residuals(self) -> None:
        """
        Raise ConfigError if the comparison of an observation gives no
        residuals (e.g. a user likelihood of the log-likelihood; see
        LikelihoodUser).
        """
        for i, comparison_ in enumerate(self._comparisons):
            if not comparison_.gives_residuals():
                raise ConfigError(
                    f"the likelihood of observation {i} is a function of "
                    f"the log-likelihood, which has no residuals (e.g. for "
                    f"least squares); give the residuals instead "
                    f"(form 'residuals'), or use another fitter")

    def residual_vector_h(
            self,
            params: ParamValues,
            weighted: bool,
            out_extra: Extras | None = None
    ) -> list[np.ndarray]:
        """
        Return the residuals of each observation, on the host: its data
        items one after the other, in one array that the next evaluation
        overwrites (without its noise residuals; see residuals).

        Parameters
        ----------
        params : ParamValues
            The parameters of the models, prefixed (see pdescs).
        weighted : bool
            Whether to apply the weights of the data items (from the
            likelihoods).
        out_extra : Extras, optional
            Where the extra outputs of the models go, if wanted.

        Returns
        -------
        list of np.ndarray
            The residuals of each observation.
        """
        self._update_residual_h(params, weighted, out_extra)
        return [c.residual_vector_h() for c in self._comparisons]

    def residual_vector_d(
            self,
            params: ParamValues,
            weighted: bool,
            out_extra: Extras | None = None
    ) -> list[DeviceArray]:
        """
        Return the residuals of each observation, on its device (see
        residual_vector_h).
        """
        self._update_residual_d(params, weighted, out_extra)
        return [c.residual_vector_d() for c in self._comparisons]

    def residual_nddata_h(
            self,
            params: ParamValues,
            weighted: bool,
            out_extra: Extras | None = None
    ) -> list[dict[str, np.ndarray]]:
        """
        Return the residuals of each data item of each observation, on the
        host, as arrays of the shape of the data, which the next
        evaluation overwrites (see residual_vector_h).
        """
        self._update_residual_h(params, weighted, out_extra)
        return [c.residual_nddata_h() for c in self._comparisons]

    def residual_nddata_d(
            self,
            params: ParamValues,
            weighted: bool,
            out_extra: Extras | None = None
    ) -> list[dict[str, DeviceArray]]:
        """
        Return the residuals of each data item of each observation, on its
        device (see residual_nddata_h).
        """
        self._update_residual_d(params, weighted, out_extra)
        return [c.residual_nddata_d() for c in self._comparisons]

    def _update_residual_h(
            self,
            params: ParamValues,
            weighted: bool,
            out_extra: Extras | None = None
    ) -> None:
        """Evaluate the residuals, and copy them to the host."""
        self._update_residual_d(params, weighted, out_extra)
        with self._group_plan.timers().measure('objective_residual_d2h'):
            for comparison_ in self._comparisons:
                comparison_.copy_to_host()

    def _update_residual_d(
            self,
            params: ParamValues,
            weighted: bool,
            out_extra: Extras | None = None
    ) -> None:
        """
        Evaluate the models (with the solved elements at 1), the solved
        values and the residuals, under the lock of the plan of the models
        (see GroupPlan.lock).
        """
        self.prepare()
        group_plan = self._group_plan
        with group_plan.lock():
            if self._groups:
                params = self._solve_groups(params)
            skip = [i for orbit in self._orbits for i in orbit.observations()]
            models = list(group_plan.model_d(
                self._unscaled(params), out_extra, skip))
            if self._orbits:
                with group_plan.timers().measure('objective_orbits'):
                    self._solve_orbits(params, models)
            with group_plan.timers().measure('objective_residual_eval'):
                scales = self._solve(models, params)
                for comparison_, model, scales_ in zip(
                        self._comparisons, models, scales):
                    comparison_.evaluate(model, params, weighted, scales_)

    def _solve_orbits(
            self,
            params: ParamValues,
            models: list[dict[str, dict[str, DeviceArray | None]]]
    ) -> None:
        """
        Solve the weights of the orbits of each orbit group (see
        OrbitGroup), and put the models of its observations in models,
        with BLAS on at most _ORBIT_BLAS_THREADS threads.
        """
        group_plan = self._group_plan
        states = group_plan.galaxy_states(params)
        first = len(self._solved) + len(self._groups)
        with threadutils.blas_limit(_ORBIT_BLAS_THREADS):
            for j, orbit in enumerate(self._orbits):
                solution, orbit_models = orbit.evaluate(
                    params, states, group_plan)
                self._solutions[first + j] = solution
                for i, model in orbit_models.items():
                    models[i] = model

    def _solve_groups(self, params: ParamValues) -> ParamValues:
        """
        Solve the groups of amplitudes of the light (see _Group), and
        return the values of the parameters with their solved values.
        """
        group_plan = self._group_plan
        drivers = group_plan.observation_drivers()
        unscaled = self._unscaled(params)
        # (every solved amplitude at 0: those of the other groups take no
        # part in the models of a group)
        values = _with_elements(
            unscaled, [e for g in self._groups for e in g.elements], 0.0)
        for g, group in enumerate(self._groups):
            size = len(group.elements)
            # The weighted residuals of the models at 0 (row 0) and with
            # each element at 1 alone (row k + 1); the residuals are linear
            # in the model (see Comparison.check_linear)
            for k in range(size + 1):
                ones = [group.elements[k - 1]] if k else []
                models = group_plan.model_d(_with_elements(values, ones, 1.0))
                for i in group.observations:
                    comparison_ = self._comparisons[i]
                    comparison_.evaluate(models[i], values, True, {})
                    drivers[i].mem_copy_d2d(
                        comparison_.residual_vector_d(),
                        self._d_rows[g, i][k])
            # The sums of the parts of the model: of R = r(m_k) - r(m_0) and
            # r(m_0), N = R R^T and b = -R r(m_0)
            normal = np.zeros((size, size))
            rhs = np.zeros(size)
            for i in group.observations:
                driver, rows = drivers[i], self._d_rows[g, i]
                h_normal, d_normal, h_rhs, d_rhs = self._sums[g, i]
                negative = -rows[0]
                for k in range(1, size + 1):
                    driver.math_add(rows[k], negative, out=rows[k])
                driver.math_matmul(rows[1:], rows[1:].T, out=d_normal)
                driver.math_matmul(rows[1:], negative, out=d_rhs)
                driver.mem_copy_d2h(d_normal, h_normal)
                driver.mem_copy_d2h(d_rhs, h_rhs)
                normal += h_normal
                rhs += h_rhs
            solution = solve_together(
                [solve for solve, _ in group.runs],
                [count for _, count in group.runs], normal, rhs)
            self._solutions[len(self._solved) + g] = solution
            values = _with_elements(values, group.elements, solution.values)
        return values

    def _unscaled(self, params: ParamValues) -> ParamValues:
        """
        Return the values of the parameters with the solved elements at 1
        (copies of those of their parameters).
        """
        if not self._solved:
            return params
        values = dict(params)
        for solved in self._solved:
            if solved.indices is None:
                values[solved.param] = 1.0
            else:
                values[solved.param] = np.array(values[solved.param])
                values[solved.param][solved.indices] = 1.0
        return values

    def _solve(
            self,
            models: list[dict[str, dict[str, DeviceArray | None]]],
            params: ParamValues
    ) -> list[dict[str, DeviceArray]]:
        """
        Solve the solved elements of the models (see Solve), and return
        the scales of the data items of each observation, on its device, by
        name.
        """
        scales: list[dict[str, DeviceArray]] = [
            {} for _ in range(self.nobservations())]
        for i in self._scaled:
            self._h_scales[i][...] = 1
        for j, solved in enumerate(self._solved):
            i = solved.observation
            a, b = _scale_sums(
                self._comparisons[i], models[i], params, solved.scale)
            if solved.indices is not None:
                a, b = a[solved.indices], b[solved.indices]
            solution = solved.solve.solve(a, b)
            self._solutions[j] = solution
            flat = self._h_scales[i].reshape(-1)
            flat[slice(None) if solved.indices is None
                 else solved.indices] = solution.values
        for i in self._scaled:
            observation = self._group.observations()[i]
            self._group_plan.observation_drivers()[i].mem_copy_h2d(
                self._h_scales[i], self._d_scales[i])
            _, scale = observation.flux_scale()
            scales[i] = dict.fromkeys(scale.items, self._d_scales[i])
        return scales


def _solved(
        group: ObservationGroup,
        comparisons: Sequence[Comparison],
        solves: Mapping[str, Solve]
) -> tuple[tuple[_Solved, ...], dict[str, Solve]]:
    """
    Return the solved elements of the flux scales of the observations of a
    group, of the solve of each element (by element name), those of one
    parameter and one solve together; and the solves of the other solved
    elements, by element name (see _groups).

    Raises
    ------
    ConfigError
        If the comparison of an observation cannot solve its scale (see
        Comparison.check_scales).
    """
    pdescs = group.pdescs()
    solved = []
    found = set()
    for i, observation in enumerate(group.observations()):
        if (flux_scale := observation.flux_scale()) is None:
            continue
        param, scale = flux_scale
        pdesc = pdescs[param]
        names = [param] if pdesc.is_scalar() else [
            element_name(param, k) for k in range(pdesc.size())]
        indices_of: dict[int, list[int]] = {}
        for k, name in enumerate(names):
            if name in solves:
                indices_of.setdefault(id(solves[name]), []).append(k)
                found.add(name)
        if indices_of:
            comparisons[i].check_scales(scale.items)
        for indices in indices_of.values():
            solve = solves[names[indices[0]]]
            solved.append(_Solved(
                i, param, None if pdesc.is_scalar() else np.array(indices),
                scale, solve))
    others = {name: solve for name, solve in solves.items()
              if name not in found}
    return tuple(solved), others


def _groups(
        group: ObservationGroup,
        comparisons: Sequence[Comparison],
        solves: Mapping[str, Solve],
        scaled: Sequence[int]
) -> tuple[_Group, ...]:
    """
    Return the groups of the solved elements of the parameters of a group
    that are amplitudes of the light (see Component.light_amplitudes), of
    the solve of each element (by element name): those of parameters
    whose models fall on the same observations solved together.

    Raises
    ------
    ConfigError
        If an element is not an amplitude of the light, or one that a model
        is not linear in (with why), its observation has a solved flux
        scale (the model is not linear in both), its comparison cannot
        solve it (see Comparison.check_linear), or its solves cannot be
        solved together (see check_together).
    """
    if not solves:
        return ()
    pdescs = group.pdescs()
    # The observations that take each amplitude of the light, and why
    # their models are not linear in it, if not
    takers: dict[str, list[int]] = {}
    reasons: dict[str, str] = {}
    for i in range(group.nobservations()):
        for param, reason in group.light_amplitudes(i).items():
            takers.setdefault(param, []).append(i)
            if reason is not None:
                reasons.setdefault(param, reason)
    elements: dict[str, list[tuple[str, int | None]]] = {}
    for param, pdesc in pdescs.items():
        names = [param] if pdesc.is_scalar() else [
            element_name(param, k) for k in range(pdesc.size())]
        for k, name in enumerate(names):
            if name in solves:
                elements.setdefault(param, []).append(
                    (param, None if pdesc.is_scalar() else k))
    if unknown := sorted(set(solves) - {
            element_name(*e) for es in elements.values() for e in es}):
        raise ConfigError(f"these elements are not parameters: {unknown}")
    for param in elements:
        if param not in takers:
            raise ConfigError(
                f"{param} cannot be solved: only the flux scales of "
                f"observations and the amplitudes of the light of components "
                f"seen by observations can be solved (the model is linear "
                f"in them)")
        if param in reasons:
            raise ConfigError(
                f"{param} cannot be solved: {reasons[param]}, so the model "
                f"is not linear in it; fit it instead")
        for i in takers[param]:
            if i in scaled:
                raise ConfigError(
                    f"{param} cannot be solved with the solved flux scale of "
                    f"observation {i}: the model is not linear in both")
            comparisons[i].check_linear(group.observations()[i].keys())
    # (the parameters whose models fall on the same observations, together)
    groups: list[tuple[list[str], set[int]]] = []
    for param in elements:
        observations = set(takers[param])
        joined = [g for g in groups if g[1] & observations]
        merged = ([param], observations)
        for g in joined:
            merged[0][:0] = g[0]
            merged[1].update(g[1])
            groups.remove(g)
        groups.append(merged)
    result = []
    for params, observations in groups:
        members = [e for param in params for e in elements[param]]
        runs: list[tuple[Solve, int]] = []
        for element in members:
            solve = solves[element_name(*element)]
            if runs and runs[-1][0] is solve:
                runs[-1] = (solve, runs[-1][1] + 1)
            else:
                runs.append((solve, 1))
        check_together(
            [solve for solve, _ in runs], [count for _, count in runs])
        result.append(_Group(
            tuple(members), tuple(runs), tuple(sorted(observations))))
    return tuple(result)


def _with_elements(
        params: ParamValues,
        elements: Sequence[tuple[str, int | None]],
        values: float | Sequence[float]
) -> ParamValues:
    """
    Return the values of the parameters with the given elements (each the
    name of its parameter and its index, None for a scalar) at the given
    values (one for all, or one for each); copies of those it changes.
    """
    if not elements:
        return params
    result = dict(params)
    values = np.broadcast_to(np.asarray(values, dtype=float), len(elements))
    copied = set()
    for (param, index), value in zip(elements, values):
        if index is None:
            result[param] = float(value)
            continue
        if param not in copied:
            result[param] = np.array(result[param], dtype=float)
            copied.add(param)
        result[param][index] = value
    return result


def _scale_sums(
        comparison_: Comparison,
        model: dict[str, dict[str, DeviceArray | None]],
        params: ParamValues,
        scale: FluxScale
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the sums of the values of a flux scale (see
    Comparison.scale_sums) of the model of an observation, over the items
    it scales, in the order of the elements of its parameter (one for a
    scale of one value).
    """
    total_a = total_b = 0.0
    for key in scale.items:
        a, b = comparison_.scale_sums(
            model, params, key, 0 if scale.axis is None else scale.axis)
        if scale.axis is None:
            a, b = np.sum(a), np.sum(b)
        total_a, total_b = total_a + a, total_b + b
    return np.ravel(total_a), np.ravel(total_b)
