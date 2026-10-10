"""
The lmfit least-squares fitter.
"""

import math
import typing
from typing import Any, Literal

import lmfit
import numpy as np

from gbkfit.fitting.base import FitRun, Fitter
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

    @staticmethod
    def libraries() -> tuple[str, ...]:
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
        # lmfit names its parameters; the names of elements (e.g. 'v[3]')
        # are not valid ones
        lmfit_params = lmfit.Parameters()
        for i, p in enumerate(typing.cast(list[FitParamBounded], fit_params)):
            lmfit_params.add(
                f'p{i}', value=p.value(), min=p.minimum(), max=p.maximum())

        def residuals(params: lmfit.Parameters) -> np.ndarray:
            x = np.array(list(params.valuesdict().values()))
            return problem.residuals(x)

        step = self._step or math.sqrt(problem.precision())
        options: dict[str, Any] = dict(
            ftol=self._ftol, xtol=self._xtol, gtol=self._gtol)
        options = {k: v for k, v in options.items() if v is not None}
        if self._method == 'leastsq':
            # MINPACK's step is sqrt(epsfcn) times the value
            options['epsfcn'] = step ** 2
        else:
            options['diff_step'] = step
        minimizer = lmfit.Minimizer(
            residuals, lmfit_params, nan_policy='raise', calc_covar=False,
            max_nfev=self._max_nfev)
        result = minimizer.minimize(method=self._method, **options)
        x = np.array(list(result.params.valuesdict().values()))
        return FitRun(
            solutions=[x], success=bool(result.success),
            message=str(result.message),
            diagnostics=dict(lmfit_nfev=int(result.nfev)), raw=result)
