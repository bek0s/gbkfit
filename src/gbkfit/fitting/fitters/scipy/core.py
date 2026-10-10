"""
The scipy least-squares fitter.
"""

import typing
from typing import Any, Literal

import numpy as np
import scipy.optimize

from gbkfit.fitting.base import FitRun, Fitter
from gbkfit.fitting.fitters._detail import (
    LeastSquaresFunctions, relative_step)
from gbkfit.fitting.fitparams import (
    FitParam, FitParamBounded, FitParamClass)
from gbkfit.fitting.problem import FitProblem
from gbkfit.fitting.uncertainty import UncertaintyOptions
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'FitterScipyLeastSquares'
]


class _BoundsTransform:
    """
    The map from unbounded internal values to bounded ones (as MINUIT
    and lmfit do), for the 'lm' method, which takes no bounds:
    x = lo + (sin(p) + 1) (hi - lo) / 2 with both bounds, and
    x = lo - 1 + sqrt(p^2 + 1) (or hi + 1 - sqrt(p^2 + 1)) with one.
    """

    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        self._lower = lower
        self._upper = upper
        self._both = np.isfinite(lower) & np.isfinite(upper)
        self._low = np.isfinite(lower) & ~self._both
        self._high = np.isfinite(upper) & ~self._both

    def to_bounded(self, p: np.ndarray) -> np.ndarray:
        """Return the bounded values of internal ones."""
        lo, hi = self._lower, self._upper
        x = p.copy()
        x[self._both] = (lo + (np.sin(p) + 1) * (hi - lo) / 2)[self._both]
        x[self._low] = (lo - 1 + np.sqrt(p * p + 1))[self._low]
        x[self._high] = (hi + 1 - np.sqrt(p * p + 1))[self._high]
        return x

    def to_internal(self, x: np.ndarray) -> np.ndarray:
        """Return the internal values of bounded ones."""
        lo, hi = self._lower, self._upper
        with np.errstate(invalid='ignore', divide='ignore'):
            p = x.copy()
            p[self._both] = np.arcsin(
                np.clip(2 * (x - lo) / (hi - lo) - 1, -1, 1))[self._both]
            p[self._low] = np.sqrt((x - lo + 1) ** 2 - 1)[self._low]
            p[self._high] = np.sqrt((hi - x + 1) ** 2 - 1)[self._high]
        return p


class FitterScipyLeastSquares(Fitter):
    """
    Least squares with scipy.optimize.least_squares: trust region
    reflective ('trf', with bounds), dogbox, or Levenberg-Marquardt
    ('lm': MINPACK, whose bounds are applied by a map of the values, as
    lmfit does). The Jacobian is by finite differences.

    The free elements take a value (where to start), and optional min
    and max (see FitParamBounded).

    Parameters
    ----------
    method : {'trf', 'dogbox', 'lm'}, optional
        The method.
    jac : {'2-point', '3-point'}, optional
        The finite differences: forward, or central (twice the cost).
    x_scale : float or 'jac', optional
        The scale of the values: a number, or 'jac' to scale each by the
        norm of its Jacobian column, as MINPACK does (by default, scipy's:
        1 for 'trf' and 'dogbox', 'jac' for 'lm').
    loss : {'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'}, optional
        The loss of the residuals ('linear' for least squares; the others
        are robust to outliers, with the scale f_scale; not with 'lm').
    f_scale : float, optional
        The residual at which robust losses start to differ.
    max_nfev : int, optional
        The most model evaluations (by default, scipy's).
    ftol, xtol, gtol : float, optional
        The tolerances of the relative change of the cost, of the values,
        and of the gradient.
    step : float, optional
        The relative step of the finite differences (by default, the
        square root, or the cube root for '3-point', of the precision of
        the models). GBKFIT computes the Jacobian: scipy's steps fall
        back to a step for float64 models at 0 (see
        LeastSquaresFunctions).
    seed : int, optional
        The seed of the Laplace samples.
    uncertainty : UncertaintyOptions, optional
        How to work out the uncertainty of the solution.

    Raises
    ------
    ConfigError
        If a number is not positive, or 'lm' is given a robust loss.
    """

    @staticmethod
    def type() -> str:
        return 'scipy.least_squares'

    @staticmethod
    def fit_param_class() -> FitParamClass:
        return FitParamBounded

    @staticmethod
    def libraries() -> tuple[str, ...]:
        return ()

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), method=self._method, jac=self._jac,
            x_scale=self._x_scale, loss=self._loss, f_scale=self._f_scale,
            max_nfev=self._max_nfev, ftol=self._ftol, xtol=self._xtol,
            gtol=self._gtol, step=self._step) | self.dump_common()

    def __init__(
            self,
            method: Literal['trf', 'dogbox', 'lm'] = 'trf',
            jac: Literal['2-point', '3-point'] = '2-point',
            x_scale: float | Literal['jac'] | None = None,
            loss: Literal[
                'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'] = 'linear',
            f_scale: float = 1.0,
            max_nfev: int | None = None,
            ftol: float = 1e-8,
            xtol: float = 1e-8,
            gtol: float = 1e-8,
            step: float | None = None,
            seed: int = 0,
            uncertainty: UncertaintyOptions | None = None
    ):
        super().__init__(seed, uncertainty)
        for name, value in dict(
                f_scale=f_scale, max_nfev=max_nfev, ftol=ftol, xtol=xtol,
                gtol=gtol, step=step).items():
            if value is not None and not value > 0:
                raise ConfigError(
                    f"option '{name}' must be positive; it is {value}")
        if x_scale not in (None, 'jac') and not x_scale > 0:
            raise ConfigError(
                f"option 'x_scale' must be 'jac' or positive; it is "
                f"{x_scale}")
        if method == 'lm' and loss != 'linear':
            raise ConfigError("method 'lm' takes only the 'linear' loss")
        self._method = method
        self._jac = jac
        self._x_scale = x_scale
        self._loss = loss
        self._f_scale = f_scale
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
        scheme: Literal['forward', 'central'] = \
            'central' if self._jac == '3-point' else 'forward'
        step = self._step or relative_step(problem.precision(), scheme)
        options: dict[str, Any] = dict(
            method=self._method, x_scale=self._x_scale, loss=self._loss,
            f_scale=self._f_scale, max_nfev=self._max_nfev, ftol=self._ftol,
            xtol=self._xtol, gtol=self._gtol)
        # The residuals are new arrays each call, which scipy needs (it
        # keeps those of earlier calls; scipy issue #11493)
        if self._method == 'lm':
            # 'lm' takes no bounds: it works on unbounded internal values
            transform = _BoundsTransform(lower, upper)
            functions = LeastSquaresFunctions(
                lambda p: problem.residuals(transform.to_bounded(p)),
                np.full_like(x0, -np.inf), np.full_like(x0, np.inf),
                scheme, step)
            result = scipy.optimize.least_squares(
                functions.residuals, transform.to_internal(x0),
                jac=functions.jacobian, **options)
            x = transform.to_bounded(result.x)
        else:
            functions = LeastSquaresFunctions(
                problem.residuals, lower, upper, scheme, step)
            result = scipy.optimize.least_squares(
                functions.residuals, x0, jac=functions.jacobian,
                bounds=(lower, upper), **options)
            x = result.x
        return FitRun(
            solutions=[x], success=bool(result.success),
            message=str(result.message),
            diagnostics=dict(status=int(result.status)), raw=result)
