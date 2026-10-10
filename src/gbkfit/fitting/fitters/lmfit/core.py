"""
The lmfit least-squares fitter.
"""

import typing
from typing import Any, Literal

import lmfit
import numpy as np

from gbkfit.fitting.base import FitRun, Fitter
from gbkfit.fitting.fitters._detail import (
    LeastSquaresFunctions, relative_step)
from gbkfit.fitting.fitparams import (
    FitParam, FitParamBounded, FitParamClass)
from gbkfit.fitting.problem import FitProblem
from gbkfit.fitting.uncertainty import UncertaintyOptions
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'FitterLMFitLeastSquares'
]


class FitterLMFitLeastSquares(Fitter):
    """
    Least squares with lmfit: Levenberg-Marquardt (MINPACK's lmdif,
    method 'leastsq'; lmfit maps bounded values to unbounded ones) or
    scipy's trust region reflective (method 'least_squares'). The
    Jacobian is by forward differences.

    The free elements take a value (where to start), and optional min
    and max (see FitParamBounded).

    Parameters
    ----------
    method : {'leastsq', 'least_squares'}, optional
        The method.
    max_nfev : int, optional
        The most model evaluations (by default, lmfit's: 2000 times the
        number of free elements plus 1).
    ftol, xtol, gtol : float, optional
        The tolerances of the relative change of the chi-square, of the
        relative change of the values, and of the gradient (by default,
        the method's).
    step : float, optional
        The relative step of the finite differences (by default, the
        square root of the precision of the models, e.g. 3.5e-4 for
        float32; MINPACK's and scipy's default, for float64, is too small
        for float32 models).
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
        return FitParamBounded

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
            max_nfev: int | None = None,
            ftol: float | None = None,
            xtol: float | None = None,
            gtol: float | None = None,
            step: float | None = None,
            seed: int = 0,
            uncertainty: UncertaintyOptions | None = None
    ):
        super().__init__(seed, uncertainty)
        for name, value in dict(
                max_nfev=max_nfev, ftol=ftol, xtol=xtol, gtol=gtol,
                step=step).items():
            if value is not None and not value > 0:
                raise ConfigError(
                    f"option '{name}' must be positive; it is {value}")
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
        # lmfit names its parameters; the names of elements (e.g. 'v[3]')
        # are not valid ones
        lmfit_params = lmfit.Parameters()
        for i, p in enumerate(params):
            lmfit_params.add(
                f'p{i}', value=p.value(), min=p.minimum(), max=p.maximum())
        step = self._step or relative_step(problem.precision(), 'forward')
        functions = LeastSquaresFunctions(
            problem.residuals,
            np.array([p.minimum() for p in params]),
            np.array([p.maximum() for p in params]), 'forward', step)

        def values(lmfit_values: lmfit.Parameters) -> np.ndarray:
            return np.array(list(lmfit_values.valuesdict().values()))

        options: dict[str, Any] = dict(
            ftol=self._ftol, xtol=self._xtol, gtol=self._gtol)
        options = {k: v for k, v in options.items() if v is not None}
        if self._method == 'leastsq':
            # MINPACK's own differences: its step, sqrt(epsfcn) times |x|
            # (or sqrt(epsfcn) at 0), suits GBKFIT's models
            options['epsfcn'] = step ** 2
        else:
            # scipy's do not (see LeastSquaresFunctions)
            options['jac'] = lambda v: functions.jacobian(values(v))
        minimizer = lmfit.Minimizer(
            lambda v: functions.residuals(values(v)), lmfit_params,
            nan_policy='raise', calc_covar=False, max_nfev=self._max_nfev)
        result = minimizer.minimize(method=self._method, **options)
        return FitRun(
            solutions=[values(result.params)], success=bool(result.success),
            message=str(result.message),
            diagnostics=dict(lmfit_nfev=int(result.nfev)), raw=result)
