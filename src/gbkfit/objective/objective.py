from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Self

import numpy as np

from gbkfit.dataset import Dataset
from gbkfit.driver import DeviceArray, Driver
from gbkfit.model import Extras
from gbkfit.objective.comparisons import Comparison, comparison
from gbkfit.observation import FluxScale, ObservationGroup
from gbkfit.params import ParamDesc, ParamValues, Solution, Solve
from gbkfit.params._keys import element_name
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Objective'
]


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


class Objective:
    """
    The comparison of the models of a group of observations with their
    data, each under its likelihood.

    Each observation has a comparison of its kind of likelihood (see
    Comparison): its residuals, of each value of its data, and the terms
    of the chi-square beyond them (e.g. the noise residuals of a fitted
    noise; see LikelihoodGaussian). The objective sums them.

    The elements of parameters can be solved at each evaluation, rather
    than fitted (see Solve and ParamSpace): for now, those of the flux
    scales of observations (see ObservationSky), which the model is linear
    in. The model is evaluated with them at 1, the comparison of their
    observation gives their sums, their solve their values, and the
    comparison then compares the model times them (see
    Comparison.evaluate). The priors of the solved values add residuals
    after those of the data; the log-likelihood has the values integrated
    out. They are nuisance parameters (see nnuisance); solved gives their
    values.

    Its memory on the devices (and the host) is allocated on the first
    evaluation, and released by close, or when Python collects it. An
    objective is a context manager that closes on exit: fitters can keep
    it alive after a fit (e.g. lmfit, in reference cycles), with its
    memory, until Python collects it.

    The methods for fitters return totals over all observations and new
    arrays (chi2, log_likelihood, residuals). The others return a value
    or the arrays of each observation, which are overwritten by the next
    evaluation.

    Parameters
    ----------
    group : ObservationGroup
        The observations, each with data.
    solves : Mapping, optional
        The solve of each solved element of the parameters, by element
        name (see ParamSpace.solves); none by default.

    Raises
    ------
    ConfigError
        If an observation has no data, its comparison cannot be made (see
        Comparison), or a solved element is not of the flux scale of an
        observation whose comparison can solve it (see
        Comparison.check_scales).
    """

    def __init__(
            self,
            group: ObservationGroup,
            solves: Mapping[str, Solve] | None = None
    ):
        observations = group.observations()
        if missing := [
                i for i, obs in enumerate(observations) if obs.data() is None]:
            raise ConfigError(
                f"every observation of an objective needs data; these have "
                f"none: {missing}")
        self._group = group
        self._datasets = tuple(obs.data() for obs in observations)
        self._comparisons = tuple(comparison(obs) for obs in observations)
        self._solves = dict(solves or {})
        self._solved = _solved(group, self._comparisons, self._solves)
        # The observations with solved elements
        self._scaled = sorted({solved.observation for solved in self._solved})
        self._reset_memory()

    def _reset_memory(self) -> None:
        """
        Drop its memory (see prepare and close): the next evaluation
        allocates it again.
        """
        for comparison_ in self._comparisons:
            comparison_.close()
        # The values of the solved elements of the last evaluation, and
        # the scales of the observations with some, on the host (h) and
        # the device (d), by index
        self._solutions: list[Solution | None] = [None] * len(self._solved)
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

    def with_driver(self, driver: Driver) -> 'Objective':
        """
        Return a copy of it with all its work on another driver: its
        models (see ObservationGroup.with_driver) and its residuals. The
        copy shares the data; its memory is its own. Copies on several
        devices evaluate several points at once.
        """
        return Objective(self._group.with_driver(driver), self._solves)

    def with_data(self, datasets: Sequence[Dataset]) -> 'Objective':
        """
        Return a copy of it with other data of its observations (e.g. those
        of a bootstrap; see ObservationGroup.with_data). The comparisons of
        the copy share what they can with its own (see Comparison.share):
        e.g. the refits of a bootstrap work out the whitening of correlated
        noise once.
        """
        objective = Objective(self._group.with_data(datasets), self._solves)
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
            (self._h_scales[i],
             self._d_scales[i]) = observation.driver().mem_alloc_s(
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
        self._update_residual_d(params, True, out_extra)
        with self._group.timers().measure('objective_residual_sum_eval'):
            chi2 = sum(comparison_.chi2() for comparison_ in self._comparisons)
        return float(chi2 + sum(
            np.sum(residuals ** 2) for residuals in self._solve_residuals()))

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
        return self.chi2_and_log_likelihood(params, out_extra)[1]

    def chi2_and_log_likelihood(
            self,
            params: ParamValues,
            out_extra: Extras | None = None
    ) -> tuple[float, float]:
        """
        Return the chi-square and the log-likelihood of all observations,
        of one evaluation (see chi2 and log_likelihood).

        Parameters
        ----------
        params : ParamValues
            The parameters of the models, prefixed (see pdescs).
        out_extra : Extras, optional
            Where the extra outputs of the models go, if wanted.

        Returns
        -------
        tuple of float
            The chi-square and the log-likelihood.
        """
        chi2 = self.chi2(params, out_extra)
        return chi2, -0.5 * chi2 + sum(
            solution.log_term for solution in self._solutions)

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
        Return the log-likelihood of each data point of all observations,
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
        if self._solved or any(
                c.noise_residuals().size for c in self._comparisons):
            raise NotImplementedError(
                "the log-likelihood of each point is not given with a fitted "
                "noise or solved values (e.g. flux scales): their terms are "
                "sums over many points")
        return -0.5 * self.residuals(params) ** 2

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
        return values

    def nnuisance(self) -> int:
        """
        Return the number of nuisance parameters that the objective solves
        itself: the solved elements (see Solve). The degrees of freedom of
        a fit exclude them.
        """
        return len(self._solves)

    def nresiduals(self) -> int:
        """
        Return the number of residuals that count: those of the comparisons
        (see Comparison.nresiduals), and that of the prior of each solved
        value; not the noise residuals, which are not of the data (see
        LikelihoodGaussian).
        """
        return sum(c.nresiduals() for c in self._comparisons) \
            + self.nnuisance()

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
        return sum(c.nindependent() for c in self._comparisons) \
            + self.nnuisance()

    def _solve_residuals(self) -> list[np.ndarray]:
        """
        Return the residuals of the priors of the solved values of the last
        evaluation, of each solve.
        """
        return [solution.residuals for solution in self._solutions]

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
        with self._group.timers().measure('objective_residual_d2h'):
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
        values and the residuals.
        """
        self.prepare()
        models = self._group.model_d(self._unscaled(params), out_extra)
        with self._group.timers().measure('objective_residual_eval'):
            scales = self._solve(models, params)
            for comparison_, model, scales_ in zip(
                    self._comparisons, models, scales):
                comparison_.evaluate(model, params, weighted, scales_)

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
            observation.driver().mem_copy_h2d(
                self._h_scales[i], self._d_scales[i])
            _, scale = observation.flux_scale()
            scales[i] = dict.fromkeys(scale.items, self._d_scales[i])
        return scales


def _solved(
        group: ObservationGroup,
        comparisons: Sequence[Comparison],
        solves: Mapping[str, Solve]
) -> tuple[_Solved, ...]:
    """
    Return the solved elements of the flux scales of the observations of a
    group, of the solve of each element (by element name), those of one
    parameter and one solve together.

    Raises
    ------
    ConfigError
        If an element is not of the flux scale of an observation, or its
        comparison cannot solve it (see Comparison.check_scales).
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
    if unknown := [name for name in solves if name not in found]:
        raise ConfigError(
            f"only the flux scales of observations can be solved (the "
            f"model is linear in them); these cannot: {unknown}")
    return tuple(solved)


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
