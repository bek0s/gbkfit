from collections.abc import Sequence
from typing import Any, NamedTuple, Self

import numpy as np

from gbkfit.dataset import Dataset
from gbkfit.driver import DeviceArray, Driver
from gbkfit.model import Extras
from gbkfit.objective.comparisons import comparison
from gbkfit.observation import ObservationGroup
from gbkfit.params import (
    ParamDesc, ParamValues, Solution, Solve, SolveGaussian)
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Objective'
]


class _Solved(NamedTuple):
    """
    Solved values (see Solve): the scales of a data item of an observation
    (its index), one at each point of the axes of the item but one, and
    how they are solved.
    """
    observation: int
    key: str
    axis: int
    solve: Solve


class Objective:
    """
    The comparison of the models of a group of observations with their
    data, each under its likelihood.

    Each observation has a comparison of its kind of likelihood (see
    Comparison): its residuals, of each value of its data, and the terms
    of the chi-square beyond them (e.g. the noise residuals of a fitted
    noise; see LikelihoodGaussian). The objective sums them.

    Some values of parameters are solved at each evaluation, rather than
    fitted (see Solve): for now, the amplitudes of the spectra of an
    observation whose likelihood has them (see LikelihoodGaussian). The
    comparison of the observation gives their sums, the solve their
    values, and the comparison then compares the model times them. Their
    priors add residuals after those of the data; the log-likelihood has
    them integrated out. They are nuisance parameters (see nnuisance), and
    an extra output of their observation, 'amplitudes'.

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

    Raises
    ------
    ConfigError
        If an observation has no data, or its comparison cannot be made
        (see Comparison).
    """

    def __init__(self, group: ObservationGroup):
        observations = group.observations()
        if missing := [
                i for i, obs in enumerate(observations) if obs.data() is None]:
            raise ConfigError(
                f"every observation of an objective needs data; these have "
                f"none: {missing}")
        self._group = group
        self._datasets = tuple(obs.data() for obs in observations)
        self._comparisons = tuple(comparison(obs) for obs in observations)
        # (the amplitudes of spectra, of their likelihoods: of a prior of
        # mean 1, cut at 0)
        self._solved = tuple(
            _Solved(
                i, obs.spectra_item(), obs.spectra_axis(),
                SolveGaussian(1.0, obs.likelihood().amplitude_sigma(), 0.0))
            for i, obs in enumerate(observations)
            if obs.likelihood().amplitudes() is not None)
        self._reset_memory()

    def _reset_memory(self) -> None:
        """
        Drop its memory (see prepare and close): the next evaluation
        allocates it again.
        """
        for comparison_ in self._comparisons:
            comparison_.close()
        # The values of each solve of the last evaluation, and its scales
        # on the host (h) and the device (d) of its observation
        n = len(self._solved)
        self._solutions: list[Solution | None] = [None] * n
        self._h_scales: list[np.ndarray | None] = [None] * n
        self._d_scales: list[DeviceArray | None] = [None] * n
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
        return Objective(self._group.with_driver(driver))

    def with_data(self, datasets: Sequence[Dataset]) -> 'Objective':
        """
        Return a copy of it with other data of its observations (e.g. those
        of a bootstrap; see ObservationGroup.with_data). The comparisons of
        the copy share what they can with its own (see Comparison.share):
        e.g. the refits of a bootstrap work out the whitening of correlated
        noise once.
        """
        objective = Objective(self._group.with_data(datasets))
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
        for j, solved in enumerate(self._solved):
            observation = self._group.observations()[solved.observation]
            (self._h_scales[j],
             self._d_scales[j]) = observation.driver().mem_alloc_s(
                self._scales_shape(solved), observation.dtype())
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
                "noise or solved values (e.g. amplitudes of spectra): their "
                "terms are sums over many points")
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

    def amplitudes(self) -> list[np.ndarray | None]:
        """
        Return the amplitudes of the spectra of each observation (see
        LikelihoodGaussian), of the last evaluation, as new arrays of the
        shape of the spectra but their axis; None for the observations
        without them, or before an evaluation.
        """
        amplitudes: list[np.ndarray | None] = [None] * self.nobservations()
        for solved, solution in zip(self._solved, self._solutions):
            if solution is not None:
                amplitudes[solved.observation] = np.squeeze(
                    solution.values, solved.axis).copy()
        return amplitudes

    def nnuisance(self) -> int:
        """
        Return the number of nuisance parameters that the objective solves
        itself (see Solve). The degrees of freedom of a fit exclude them.
        """
        return sum(
            int(np.prod(self._scales_shape(solved)))
            for solved in self._solved)

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

    def _scales_shape(self, solved: _Solved) -> tuple[int, ...]:
        """
        Return the shape of the scales of solved values: that of their data
        item, of 1 along its axis.
        """
        shape = self._datasets[solved.observation].shape()
        return tuple(1 if k == solved.axis else n for k, n in enumerate(shape))

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
        """Evaluate the models, the solved values and the residuals."""
        self.prepare()
        models = self._group.model_d(params, out_extra)
        with self._group.timers().measure('objective_residual_eval'):
            scales = self._solve(models, params, out_extra)
            for comparison_, model, scales_ in zip(
                    self._comparisons, models, scales):
                comparison_.evaluate(model, params, weighted, scales_)

    def _solve(
            self,
            models: list[dict[str, dict[str, DeviceArray | None]]],
            params: ParamValues,
            out_extra: Extras | None
    ) -> list[dict[str, DeviceArray]]:
        """
        Solve the solved values of the models (see Solve), and return the
        scales of the data items of each observation, on its device, by
        name; with extras, the amplitudes of spectra go to them
        ('{prefix}amplitudes', of the shape of the spectra but their axis).
        """
        scales: list[dict[str, DeviceArray]] = [
            {} for _ in range(self.nobservations())]
        for j, solved in enumerate(self._solved):
            i = solved.observation
            sums = self._comparisons[i].scale_sums(
                models[i], params, solved.key, solved.axis)
            solution = solved.solve.solve(*sums)
            self._solutions[j] = solution
            self._h_scales[j][...] = solution.values
            self._group.observations()[i].driver().mem_copy_h2d(
                self._h_scales[j], self._d_scales[j])
            scales[i][solved.key] = self._d_scales[j]
            if out_extra is not None:
                out_extra.add(
                    'amplitudes',
                    np.squeeze(solution.values, solved.axis).copy(),
                    "Amplitudes of the spectra",
                    "The amplitude of each spectrum of the model that fits "
                    "the data best (see LikelihoodGaussian): the factor of "
                    "its model in the residuals.",
                    observation=self._group.observation_label(i))
        return scales
