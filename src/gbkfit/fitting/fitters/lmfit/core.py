"""
The lmfit least-squares fitter.
"""

import typing
from typing import Annotated, Any, Literal

import lmfit
import numpy as np
from annotated_types import Gt

from gbkfit.fitting.base import FitRun, Fitter
from gbkfit.fitting.fitters._detail import (
    LeastSquaresFunctions, inside_bounds, relative_step)
from gbkfit.fitting.fitparams import (
    FitParam, FitParamBounded, FitParamClass, FitParamLeastSquares)
from gbkfit.fitting.problem import FitProblem
from gbkfit.fitting.uncertainty import (
    UncertaintyOptions, finite_difference_steps)
from gbkfit.utils.parseutils import check_options


__all__ = [
    'FitterLMFitLeastSquares'
]


class FitterLMFitLeastSquares(Fitter):
    """
    Least squares with lmfit: Levenberg-Marquardt (MINPACK, method
    'leastsq'; lmfit maps bounded values to unbounded ones) or scipy's
    trust region reflective (method 'least_squares'). GBKFIT computes the
    Jacobian by forward differences (see LeastSquaresFunctions).

    The free elements take a value (where to start), and optional min,
    max and a Gaussian or uniform prior (see FitParamLeastSquares).

    Parameters
    ----------
    method : {'leastsq', 'least_squares'}, optional
        The method.
    max_nfev : int, optional
        The most evaluations of the residuals by the method, not counting
        those of the Jacobians (by default, lmfit's: 2000 times (the
        number of free elements + 1)).
    ftol, xtol, gtol : float, optional
        The tolerances of the relative change of the chi-square, of the
        relative change of the values, and of the gradient (by default,
        the method's).
    step : float, optional
        The relative step of the finite differences (by default, the
        square root of the precision of the models, e.g. 3.5e-4 for
        float32).
    seed : int, optional
        The seed of the Laplace samples.
    uncertainty : UncertaintyOptions, optional
        How to work out the uncertainty of the solution.

    Raises
    ------
    ConfigError
        If max_nfev or a tolerance or the step is not positive.
    """

    @staticmethod
    def type() -> str:
        return 'lmfit.least_squares'

    @staticmethod
    def fit_param_class() -> FitParamClass:
        return FitParamLeastSquares

    def libraries(self) -> tuple[str, ...]:
        return ('lmfit',)

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), method=self._method, max_nfev=self._max_nfev,
            ftol=self._ftol, xtol=self._xtol, gtol=self._gtol,
            step=self._step) | self.dump_common()

    def __init__(
            self,
            method: Literal['leastsq', 'least_squares'] = 'leastsq',
            max_nfev: Annotated[int, Gt(0)] | None = None,
            ftol: Annotated[float, Gt(0)] | None = None,
            xtol: Annotated[float, Gt(0)] | None = None,
            gtol: Annotated[float, Gt(0)] | None = None,
            step: Annotated[float, Gt(0)] | None = None,
            seed: int = 0,
            uncertainty: UncertaintyOptions | None = None
    ):
        check_options(FitterLMFitLeastSquares.__init__, locals())
        super().__init__(seed, uncertainty)
        self._method = method
        self._max_nfev = max_nfev
        self._ftol = ftol
        self._xtol = xtol
        self._gtol = gtol
        self._step = step

    def _run(
            self,
            problem: FitProblem,
            fit_params: list[FitParam],
            rng: np.random.Generator
    ) -> FitRun:
        params = typing.cast(list[FitParamBounded], fit_params)
        x0 = np.array([p.value() for p in params])
        lower = np.array([p.minimum() for p in params])
        upper = np.array([p.maximum() for p in params])
        scales = np.array([p.scale() for p in params])
        step = self._step or relative_step(problem.precision(), 'forward')
        # MINPACK's own steps (sqrt(epsfcn) times the value) are too small
        # for float32 models at small values, and scipy's at 0: GBKFIT
        # computes the Jacobian (see LeastSquaresFunctions)
        functions = LeastSquaresFunctions(
            problem.residuals_batch, lower, upper, scales, 'forward', step)
        # A start on a bound does not move: lmfit's map of bounded values
        # (leastsq) is flat there, and TRF's first steps are tiny
        x0 = inside_bounds(
            x0, lower, upper, finite_difference_steps(x0, scales, step))
        # lmfit names its parameters; the names of elements (e.g. 'v[3]')
        # are not valid ones
        lmfit_params = lmfit.Parameters()
        for i, value in enumerate(x0):
            lmfit_params.add(
                f'p{i}', value=value, min=lower[i], max=upper[i])

        def values(lmfit_values: lmfit.Parameters) -> np.ndarray:
            return np.array(list(lmfit_values.valuesdict().values()))

        options: dict[str, Any] = dict(
            ftol=self._ftol, xtol=self._xtol, gtol=self._gtol)
        options = {k: v for k, v in options.items() if v is not None}
        jac_option = 'Dfun' if self._method == 'leastsq' else 'jac'
        options[jac_option] = lambda v: functions.jacobian(values(v))
        minimizer = lmfit.Minimizer(
            lambda v: functions.residuals(values(v)), lmfit_params,
            nan_policy='raise', calc_covar=False, max_nfev=self._max_nfev)
        result = minimizer.minimize(method=self._method, **options)
        # A run stopped by max_nfev ends at its last evaluation (e.g. a
        # rejected step): take its best one instead
        best = functions.best()
        x = best if result.aborted and best is not None \
            else values(result.params)
        return FitRun(
            solutions=[x], success=bool(result.success),
            message=self._message(result, minimizer.max_nfev),
            diagnostics=dict(lmfit_nfev=int(result.nfev)), raw=result)

    def _message(
            self,
            result: lmfit.minimizer.MinimizerResult,
            max_nfev: int
    ) -> str:
        """
        Return why the method stopped. For 'leastsq', lmfit reports every
        convergence of MINPACK as 'Fit succeeded.', whichever test stopped
        it (also the values not changing because no step was accepted),
        and lmfit 1.3.4 replaces its message for a run stopped by max_nfev
        with 'Tolerance seems to be too small.': MINPACK's message says
        which test stopped it. 'least_squares' keeps scipy's message, and
        lmfit's own when stopped by max_nfev.
        """
        if self._method != 'leastsq':
            return str(result.message)
        if result.aborted:
            return f"Fit aborted: number of function evaluations > {max_nfev}"
        return str(result.lmdif_message)
