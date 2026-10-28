import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple, Self, TypeAlias

import numpy as np

from gbkfit.dataset import Dataset
from gbkfit.driver import DeviceArray, Driver
from gbkfit.model import Extras
from gbkfit.objective.comparisons import comparison
from gbkfit.observation import GroupPlan, ModelData, ObservationGroup
from gbkfit.params import ParamDesc, ParamValues, Solution, Solve
from gbkfit.params._keys import with_elements
from gbkfit.utils import threadutils
from gbkfit.utils.parseutils import ConfigError
from ._amplitudes import AmplitudeSolve, amplitude_solves
from ._orbits import OrbitSolve, orbit_solves
from ._scales import ScaleSolve, scale_solves


__all__ = [
    'Evaluation',
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


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """
    An evaluation of an objective at the values of its parameters (see
    Objective.evaluate): its merits, the parts of its chi-square, and the
    values of its solved elements there.

    Attributes
    ----------
    merits : Merits
        The numbers by which fitters judge the models.
    data_chi2 : float
        The chi-square of the comparisons of the data with the models, with
        the noise residuals of a fitted noise (see noise_chi2).
    noise_chi2 : float
        The sum of the squared noise residuals (see LikelihoodGaussian):
        the part of data_chi2 that is not of the data (0 without a fitted
        noise).
    prior_chi2 : float
        The sum of the squared residuals of the priors of the solved
        values (and of the constraints of the light of orbits): the rest
        of the chi-square, not of the data.
    nnuisance : float
        The effective number of the solved values (the sum of their m_eff;
        see Solve): how much the data decide them.
    profiled_prior_chi2 : float
        The part of prior_chi2 of the values solved 'profiled' (see
        Solve.likelihood).
    profiled_nnuisance : float
        The part of nnuisance of the values solved 'profiled'.
    solved : dict
        The values of the solved elements, by element name (see
        ParamSpace.evaluate).
    """
    merits: Merits
    data_chi2: float
    noise_chi2: float
    prior_chi2: float
    nnuisance: float
    profiled_prior_chi2: float
    profiled_nnuisance: float
    solved: dict[str, float]


# Solved elements solved together (a block): elements of a flux scale,
# amplitudes of the light, or the weights of orbits
_Block: TypeAlias = ScaleSolve | AmplitudeSolve | OrbitSolve


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
    than fitted (see Solve and ParamSpace): those that the models are
    linear in, in blocks solved together. The plan of the group gives the
    models as linear in them, and the comparisons weigh their pieces (see
    Comparison.linear_form). A flux scale of an observation (see
    ObservationSky; ScaleSolve): the model is evaluated with it at 1, the
    comparison of the observation gives its sums, its solve its values,
    and the comparison then compares the model times them (see
    Comparison.evaluate). Amplitudes of the light of components (see
    Component.light_amplitudes; AmplitudeSolve), solved together where
    their parts of the model fall on the same data (in one or several
    observations): the plan gives the models with them at 0 and the piece
    of each (see GroupPlan.amplitude_pieces), and the models are evaluated
    at their values. They do not go with a solved flux scale of the same
    observation (the model is then not linear in both). The weights of the
    orbits of the stars of a spheroid (see Component.orbit_weights;
    OrbitSolve): the models of the observations that see the stars are
    linear in them (the orbits in the series of their data, see
    OrbitRecording), and the models are evaluated at their values. The
    priors of the solved values add residuals after those of the data;
    the log-likelihood has the values integrated out, or at their best
    values (see Solve). They are nuisance parameters (see
    Evaluation.nnuisance); an evaluation gives their values (see
    evaluate).

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
        Comparison.pointwise_form), nor an amplitude of the light that the
        models are linear in (see Component.light_amplitudes,
        Observable.linear_in_light and Comparison.linear_form), nor a
        weight of orbits; or the weights of the orbits of stars that
        observations see are not solved (see OrbitSolve).
    """

    def __init__(
            self,
            group_plan: GroupPlan,
            solves: Mapping[str, Solve] | None = None
    ):
        group = group_plan.group()
        observations = group.observations()
        group.require_data()
        self._group_plan = group_plan
        self._group = group
        self._datasets = tuple(obs.data() for obs in observations)
        self._comparisons = tuple(
            comparison(obs, driver) for obs, driver in zip(
                observations, group_plan.observation_drivers()))
        self._solves = dict(solves or {})
        # The blocks of each kind, each of the solves left by the others
        self._orbits = orbit_solves(
            group_plan, self._comparisons, self._solves)
        self._scales = scale_solves(
            group, self._comparisons, _unclaimed(self._solves, self._orbits))
        # The observations with solved scales
        self._scaled = sorted({block.observation() for block in self._scales})
        self._amplitudes = amplitude_solves(
            group, self._comparisons, group_plan.observation_drivers(),
            _unclaimed(self._solves, (*self._orbits, *self._scales)),
            self._scaled)
        # Every block, in the order of their values (see evaluate)
        self._blocks: tuple[_Block, ...] = (
            *self._scales, *self._amplitudes, *self._orbits)
        self._reset_memory()

    def _reset_memory(self) -> None:
        """
        Drop its memory (see prepare and close): the next evaluation
        allocates it again.
        """
        for comparison_ in self._comparisons:
            comparison_.close()
        for block in (*self._amplitudes, *self._orbits):
            block.close()
        # The solution of each block of the last evaluation, and the scales
        # of the observations with solved ones, on the host (h) and the
        # device (d), by index
        self._solutions: dict[_Block, Solution] = {}
        self._h_scales: dict[int, np.ndarray] = {}
        self._d_scales: dict[int, DeviceArray] = {}
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
        of correlated noise; see LikelihoodGaussian), with that of the
        recordings of orbits of its plan (see GroupPlan.close). The next
        evaluation allocates it again.
        """
        self._reset_memory()
        self._group_plan.close()

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
        solved values (see Evaluation.nnuisance), without the residuals of their
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
        observations, of one evaluation (see evaluate).
        """
        return self.evaluate(params, out_extra).merits

    def evaluate(
            self,
            params: ParamValues,
            out_extra: Extras | None = None
    ) -> Evaluation:
        """
        Evaluate the models and compare them with the data: return the
        merits (see chi2, log_likelihood and aicp), the parts of the
        chi-square and the solved values there.

        Parameters
        ----------
        params : ParamValues
            The parameters of the models, prefixed (see pdescs).
        out_extra : Extras, optional
            Where the extra outputs of the models go, if wanted.

        Returns
        -------
        Evaluation
            The evaluation.
        """
        self._update_residual_d(params, True, out_extra)
        with self._group_plan.timers().measure('objective_residual_sum_eval'):
            data_chi2 = float(sum(
                comparison_.chi2() for comparison_ in self._comparisons))
        prior_chi2 = float(sum(
            np.sum(residuals ** 2) for residuals in self._solve_residuals()))
        chi2 = data_chi2 + prior_chi2
        # (the noise residuals of fitted scales hold a constant of their
        # scale_min, which the log-likelihood and the AICp leave out)
        constant = float(sum(c.noise_constant() for c in self._comparisons))
        log_likelihood = -0.5 * (chi2 + constant) + float(sum(
            solution.log_term for solution in self._solutions.values()))
        nnuisance = self._nnuisance()
        profiled = [
            self._solutions[block] for block in self._blocks
            if block.likelihood() == 'profiled']
        return Evaluation(
            Merits(chi2, log_likelihood,
                   data_chi2 + constant + 2 * nnuisance),
            data_chi2=data_chi2, noise_chi2=self._noise_chi2(),
            prior_chi2=prior_chi2, nnuisance=nnuisance,
            profiled_prior_chi2=float(sum(
                np.sum(solution.residuals ** 2) for solution in profiled)),
            profiled_nnuisance=float(sum(
                solution.m_eff for solution in profiled)),
            solved=self._solved_values())

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
        if self._blocks or any(
                c.noise_residuals().size for c in self._comparisons):
            raise NotImplementedError(
                "the log-likelihood of each point is not given with a fitted "
                "noise or solved values (e.g. flux scales): their terms are "
                "sums over many points")
        counted = np.concatenate([c.counted() for c in self._comparisons])
        return -0.5 * self.residuals(params)[counted] ** 2

    def _noise_chi2(self) -> float:
        """
        Return the sum of the squared noise residuals of all observations
        (see LikelihoodGaussian), of the last evaluation.
        """
        return float(sum(
            np.sum(c.noise_residuals() ** 2) for c in self._comparisons))

    def solves(self) -> dict[str, Solve]:
        """Return the solve of each solved element, by element name."""
        return dict(self._solves)

    def _solved_values(self) -> dict[str, float]:
        """
        Return the values of the solved elements of the last evaluation,
        by element name (see ParamSpace.evaluate).
        """
        values = {}
        for block in self._blocks:
            values.update(zip(
                block.elements(), self._solutions[block].values.tolist()))
        return values

    def _nnuisance(self) -> float:
        """
        Return the effective number of the solved values of the last
        evaluation (see Evaluation.nnuisance).
        """
        # Each solved value counts by how much its data decide it (its
        # m_eff: 1 decided by the data, 0 by its prior or at its minimum),
        # not as 1: a value its data decide uses up a degree of freedom,
        # and the residual of its prior, about 0 then, is not a residual
        # of the data (see nresiduals). Counting it as 1 with its prior
        # residual as a residual (net 0) left the degrees of freedom too
        # large by about 1 for each value the data decide.
        return float(sum(
            solution.m_eff for solution in self._solutions.values()))

    def nresiduals(self) -> int:
        """
        Return the number of residuals of the data that count: those of the
        comparisons (see Comparison.nresiduals); not the residuals of the
        priors of solved values (see Evaluation.nnuisance), nor the noise residuals
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
        return [self._solutions[block].residuals for block in self._blocks]

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
        Evaluate the models, the solved values and the residuals, under the
        lock of the plan of the models (see GroupPlan.lock): the amplitudes
        of the light and the weights of orbits from the pieces of the
        models, the models at their values (with solved scales at 1), and
        the scales from those models.
        """
        self.prepare()
        group_plan = self._group_plan
        solutions: dict[_Block, Solution] = {}
        with group_plan.lock():
            values = self._unscaled(params)
            for block in self._amplitudes:
                pieces = group_plan.amplitude_pieces(
                    values, block.amplitudes(), block.observations())
                solutions[block] = block.solve(pieces, values)
                values = with_elements(
                    values, block.amplitudes(), solutions[block].values)
            if self._orbits:
                with group_plan.timers().measure('objective_orbits'):
                    values = self._solve_orbits(values, solutions)
            models = group_plan.model_d(values, out_extra)
            with group_plan.timers().measure('objective_residual_eval'):
                scales = self._solve_scales(models, values, solutions)
                for comparison_, model, scales_ in zip(
                        self._comparisons, models, scales):
                    comparison_.evaluate(model, values, weighted, scales_)
        self._solutions = solutions

    def _solve_orbits(
            self,
            params: ParamValues,
            solutions: dict[_Block, Solution]
    ) -> ParamValues:
        """
        Solve the weights of the orbits of each recording of the plan (see
        OrbitSolve), with BLAS on at most _ORBIT_BLAS_THREADS threads, into
        the solutions, and return the values of the parameters with their
        solved values.
        """
        with threadutils.blas_limit(_ORBIT_BLAS_THREADS):
            series = self._group_plan.orbit_series(params)
            for orbit, series_j in zip(self._orbits, series):
                solution = orbit.solve(series_j, params)
                solutions[orbit] = solution
                params = params | {
                    orbit.recording().weights(): solution.values}
        return params

    def _unscaled(self, params: ParamValues) -> ParamValues:
        """
        Return the values of the parameters with the solved elements of
        flux scales at 1 (copies of those of their parameters).
        """
        values = params
        for block in self._scales:
            values = block.neutral(values)
        return values

    def _solve_scales(
            self,
            models: Sequence[ModelData],
            params: ParamValues,
            solutions: dict[_Block, Solution]
    ) -> list[dict[str, DeviceArray]]:
        """
        Solve the elements of the flux scales from the models (see
        ScaleSolve) into the solutions, and return the scales of the data
        items of each observation, on its device, by name.
        """
        scales: list[dict[str, DeviceArray]] = [
            {} for _ in range(self.nobservations())]
        for i in self._scaled:
            self._h_scales[i][...] = 1
        for block in self._scales:
            i = block.observation()
            solution = block.solve(models[i], params)
            solutions[block] = solution
            block.place(solution.values, self._h_scales[i])
        for i in self._scaled:
            observation = self._group.observations()[i]
            self._group_plan.observation_drivers()[i].mem_copy_h2d(
                self._h_scales[i], self._d_scales[i])
            _, scale = observation.flux_scale()
            scales[i] = dict.fromkeys(scale.items, self._d_scales[i])
        return scales


def _unclaimed(
        solves: Mapping[str, Solve], blocks: Iterable[_Block]
) -> dict[str, Solve]:
    """
    Return the solves of the solved elements that none of the blocks
    solves, by element name.
    """
    claimed = {element for block in blocks for element in block.elements()}
    return {name: solve for name, solve in solves.items()
            if name not in claimed}
