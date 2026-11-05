"""
Priors on several parameters at once (see JointPrior): the options
'priors' of the parameters (see ParamSpace), beside the prior of each
free element (see Prior).
"""

import abc
import numbers
from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
import scipy.linalg
from annotated_types import Gt
from pydantic import FiniteFloat

from ._errors import ParamEvaluationError
from .space import ParamFunctions, ParamSpace, ParamValues
from gbkfit.config import (
    Choices, TypedParser, TypedSerializable, check_options)


__all__ = [
    'JointPrior',
    'JointPriorBound',
    'JointPriorGauss',
    'JointPriorMVN',
    'JointPriorSmooth',
    'JointPriorSmooth2D',
    'joint_prior_parser',
    'joint_prior_residuals'
]


# The residuals of a prior on several parameters, of the values of all
# the parameters
Residuals = Callable[[ParamValues], np.ndarray]

# The option of the order of the differences of a prior on smoothness
_Order: TypeAlias = Annotated[Literal[1, 2], Choices(
    {1: 'First differences', 2: 'Second differences'})]


def _check_std(std: float | Sequence[float]) -> None:
    """Check that standard deviations are finite and positive."""
    values = np.asarray(std, dtype=float)
    if not (np.all(np.isfinite(values)) and np.all(values > 0)):
        raise ValueError(
            f"std must be finite and positive; it is {std}")


class JointPrior(TypedSerializable, abc.ABC):
    """
    A prior on several parameters at once: on values computed from all
    the parameter values (after ties, modes and fixed values) and the
    constants (e.g. the radial nodes of disks) by expressions, as those of
    the properties (see ParamSpace.expression). For example, a prior on a
    rotation speed derived from several parameters, or the smoothness of
    a parameter from ring to ring.

    It is given by residuals r, of a number that does not depend on the
    values: its chi-square, -2 ln p up to a constant, is the sum of their
    squares. The fitters use them as they use the residuals of the prior
    of each element (see Prior and FitProblem): least squares append
    them to the residuals, the other optimizers add the chi-square to
    theirs, and the samplers add -chi2 / 2 to the log-prior.
    """

    @abc.abstractmethod
    def residuals(
            self,
            params: ParamSpace,
            functions: ParamFunctions | None = None
    ) -> Residuals:
        """
        Return the function of the residuals of the prior, of the values
        of the parameters of a space (whose expressions it compiles), with
        the functions of the models that the derived values they read
        call (see ParamSpace.derived_values).

        Raises
        ------
        ValueError
            If an expression is invalid.
        """


class JointPriorGauss(JointPrior):
    """
    A Gaussian prior on an expression (e.g. on the rotation at 2.2 disk
    scale lengths, from the Tully-Fisher relation). Its residuals are
    (g - mean) / std, one per element of its value.

    Parameters
    ----------
    expr : str
        The expression.
    mean, std : float or Sequence of float
        The mean and standard deviation, of each element of its value or
        of all.

    Raises
    ------
    ValueError
        If std is not finite and positive.
    """

    @staticmethod
    def type() -> str:
        return 'gauss'

    @staticmethod
    def type_title() -> str:
        return 'Gaussian'

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), expr=self._expr, mean=self._mean,
            std=self._std)

    def __init__(
            self,
            expr: str,
            mean: float | Sequence[float],
            std: float | Sequence[float]
    ):
        check_options(JointPriorGauss.__init__, locals())
        _check_std(std)
        self._expr = expr
        self._mean = mean
        self._std = std

    def residuals(
            self,
            params: ParamSpace,
            functions: ParamFunctions | None = None
    ) -> Residuals:
        expression = params.expression(self._expr, functions)
        mean = np.asarray(self._mean, dtype=float)
        std = np.asarray(self._std, dtype=float)
        return lambda values: ((expression(values) - mean) / std).ravel()


class JointPriorSmooth(JointPrior):
    """
    A prior on the smoothness of a vector, e.g. a parameter of each ring
    of a disk. It is on its first differences (order 1, a random walk:
    penalises jumps) or its second differences (order 2: penalises bends,
    not steady trends), with standard deviation std from element to
    element.

    With the positions of the elements (e.g. the radial nodes), the
    differences are those of the derivatives over the positions, in units
    of their mean spacing: the residuals are the terms of the integral of
    the squared first or second derivative. For evenly spaced elements,
    order 1 gives (x[k+1] - x[k]) / std, and order 2 gives
    (x[k+1] - 2 x[k] + x[k-1]) / std.

    Parameters
    ----------
    expr : str
        The vector (an expression, e.g. 'posa').
    order : {1, 2}
        The order of the differences.
    std : float
        The standard deviation of a difference, for the mean spacing.
    positions : str, optional
        The positions of the elements (an expression, e.g. 'rnodes'),
        increasing; by default evenly spaced.

    Raises
    ------
    ValueError
        If the order is not 1 or 2, or std is not finite and positive.
    """

    @staticmethod
    def type() -> str:
        return 'smooth'

    @staticmethod
    def type_title() -> str:
        return 'Smoothness'

    def dump(self) -> dict[str, Any]:
        info = dict(
            type=self.type(), expr=self._expr, order=self._order,
            std=self._std)
        if self._positions is not None:
            info['positions'] = self._positions
        return info

    def __init__(
            self,
            expr: str,
            order: _Order,
            std: Annotated[FiniteFloat, Gt(0)],
            positions: str | None = None
    ):
        check_options(JointPriorSmooth.__init__, locals())
        self._expr = expr
        self._order = order
        self._std = std
        self._positions = positions

    def residuals(
            self,
            params: ParamSpace,
            functions: ParamFunctions | None = None
    ) -> Residuals:
        expression = params.expression(self._expr, functions)
        positions = None if self._positions is None \
            else params.expression(self._positions, functions)

        def residuals(values: ParamValues) -> np.ndarray:
            x = expression(values)
            spacing = np.ones(len(x) - 1) if positions is None \
                else self._spacing(positions(values), len(x))
            return self._differences(x, spacing) / self._std
        return residuals

    def _spacing(self, positions: np.ndarray, n: int) -> np.ndarray:
        """
        Return the spacing of the positions of n elements, in units of its
        mean.

        Raises
        ------
        ValueError
            If the positions are not one per element.
        ParamEvaluationError
            If they are not increasing (a point that cannot be evaluated,
            of positions that are parameters).
        """
        if positions.shape != (n,):
            raise ValueError(
                f"{self._positions}: there must be one position per element "
                f"of {self._expr} ({n}); there are {positions.size}")
        spacing = np.diff(positions)
        if not np.all(spacing > 0):
            raise ParamEvaluationError(
                f"{self._positions}: must be increasing")
        return spacing / spacing.mean()

    def _differences(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Return the differences of the derivative of the given order, of
        values at the spacing h, weighted as the terms of the integral of
        its square.

        Raises
        ------
        ValueError
            If there are not more values than the order.
        """
        if len(x) <= self._order:
            raise ValueError(
                f"{self._expr}: a smoothness prior of order {self._order} "
                f"needs more than {self._order} values; there are {len(x)}")
        slope = np.diff(x) / h
        if self._order == 1:
            # (x' at the middle of each interval, of length h)
            return slope * np.sqrt(h)
        # (x'' at each inner element, over half the intervals around it)
        width = (h[1:] + h[:-1]) / 2
        return np.diff(slope) / width * np.sqrt(width)


class JointPriorSmooth2D(JointPrior):
    """
    A prior on the smoothness of a map, e.g. the cells of a map trait.
    The map is a vector of nx ny values, row by row (an image c of shape
    (ny, nx), x along its columns), and the prior is on its differences
    from cell to cell, with standard deviation std. Order 1 (first
    differences: penalises jumps) gives (c[j, i+1] - c[j, i]) / std along
    x and (c[j+1, i] - c[j, i]) / std along y. Order 2 (second
    differences: penalises bends, not steady gradients) gives
    (c[j, i+1] - 2 c[j, i] + c[j, i-1]) / std along x, the same along y,
    and sqrt(2) (c[j+1, i+1] - c[j+1, i] - c[j, i+1] + c[j, i]) / std
    across: the terms of the thin-plate energy f_xx^2 + 2 f_xy^2 + f_yy^2,
    which is the same in every direction (without the term across, a
    saddle c = x y would cost nothing).

    Parameters
    ----------
    expr : str
        The map (an expression, e.g. 'pt1_a').
    size : Sequence of int
        Its numbers of cells along x and y, [nx, ny].
    order : {1, 2}
        The order of the differences.
    std : float
        The standard deviation of a difference.

    Raises
    ------
    ValueError
        If the size is not two numbers of at least 1, the order is not 1
        or 2, or std is not finite and positive.
    """

    @staticmethod
    def type() -> str:
        return 'smooth2d'

    @staticmethod
    def type_title() -> str:
        return 'Smoothness of a map'

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), expr=self._expr, size=list(self._size),
            order=self._order, std=self._std)

    def __init__(
            self,
            expr: str,
            size: Sequence[int],
            order: _Order,
            std: Annotated[FiniteFloat, Gt(0)]
    ):
        check_options(JointPriorSmooth2D.__init__, locals())
        if not (isinstance(size, Sequence) and len(size) == 2 and all(
                isinstance(n, numbers.Integral) and not isinstance(n, bool)
                and n >= 1 for n in size)):
            raise ValueError(
                f"size must be the numbers of cells along x and y, [nx, "
                f"ny], each at least 1; it is {size!r}")
        self._expr = expr
        self._size = (int(size[0]), int(size[1]))
        self._order = order
        self._std = std

    def residuals(
            self,
            params: ParamSpace,
            functions: ParamFunctions | None = None
    ) -> Residuals:
        expression = params.expression(self._expr, functions)
        nx, ny = self._size

        def residuals(values: ParamValues) -> np.ndarray:
            x = np.asarray(expression(values), dtype=float)
            if x.size != nx * ny:
                raise ValueError(
                    f"{self._expr}: a map of {nx} by {ny} cells needs "
                    f"{nx * ny} values; there are {x.size}")
            c = x.reshape(ny, nx)
            if self._order == 1:
                terms = (np.diff(c, axis=1), np.diff(c, axis=0))
            else:
                terms = (
                    np.diff(c, 2, axis=1), np.diff(c, 2, axis=0),
                    np.sqrt(2) * np.diff(np.diff(c, axis=0), axis=1))
            return np.concatenate([t.ravel() for t in terms]) / self._std
        return residuals


class JointPriorBound(JointPrior):
    """
    A soft bound on an expression: no penalty inside the bounds, and one
    that grows beyond them. Outside, the residual is (lower - g) / std or
    (g - upper) / std, for each element of its value (e.g. a mass that
    grows with radius: the differences of r v^2 above 0). The residuals
    are 0 inside, so their derivatives jump at the bounds; a hard bound
    (std near 0) would stall least squares.

    Parameters
    ----------
    expr : str
        The expression.
    lower, upper : float, optional
        The bounds (at least one).
    std : float
        How far beyond a bound one standard deviation of penalty is.

    Raises
    ------
    ValueError
        If neither bound is given, the lower is not below the upper, or
        std is not finite and positive.
    """

    @staticmethod
    def type() -> str:
        return 'bound'

    @staticmethod
    def type_title() -> str:
        return 'Soft bound'

    def dump(self) -> dict[str, Any]:
        info = dict(type=self.type(), expr=self._expr, std=self._std)
        if self._lower is not None:
            info['lower'] = self._lower
        if self._upper is not None:
            info['upper'] = self._upper
        return info

    def __init__(
            self,
            expr: str,
            std: Annotated[FiniteFloat, Gt(0)],
            lower: float | None = None,
            upper: float | None = None
    ):
        check_options(JointPriorBound.__init__, locals())
        if lower is None and upper is None:
            raise ValueError("a bound needs lower, upper or both")
        if lower is not None and upper is not None and not lower < upper:
            raise ValueError(
                f"lower must be below upper; they are {lower} and {upper}")
        self._expr = expr
        self._std = std
        self._lower = lower
        self._upper = upper

    def residuals(
            self,
            params: ParamSpace,
            functions: ParamFunctions | None = None
    ) -> Residuals:
        expression = params.expression(self._expr, functions)

        def residuals(values: ParamValues) -> np.ndarray:
            g = expression(values).ravel()
            parts = []
            if self._lower is not None:
                parts.append(np.maximum(self._lower - g, 0))
            if self._upper is not None:
                parts.append(np.maximum(g - self._upper, 0))
            return np.concatenate(parts) / self._std
        return residuals


class JointPriorMVN(JointPrior):
    """
    A multivariate Gaussian prior on several expressions together, with
    their covariance. It suits e.g. the inclination and position angle of
    a previous fit, and their correlation. Its residuals are
    L^-1 (g - mean), with L the Cholesky factor of the covariance
    (L L^T = cov), so that the sum of their squares is
    (g - mean)^T cov^-1 (g - mean).

    Parameters
    ----------
    exprs : Sequence of str
        The expressions, each of one value.
    mean : Sequence of float
        The mean of each.
    cov : Sequence of Sequence of float
        Their covariance, symmetric positive definite.

    Raises
    ------
    ValueError
        If the sizes do not match, or the covariance is not symmetric
        positive definite.
    """

    @staticmethod
    def type() -> str:
        return 'mvn'

    @staticmethod
    def type_title() -> str:
        return 'Multivariate Gaussian'

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), exprs=list(self._exprs), mean=list(self._mean),
            cov=[list(row) for row in self._cov])

    def __init__(
            self,
            exprs: Sequence[str],
            mean: Sequence[float],
            cov: Sequence[Sequence[float]]
    ):
        check_options(JointPriorMVN.__init__, locals())
        n = len(exprs)
        cov_ = np.asarray(cov, dtype=float)
        if len(mean) != n or cov_.shape != (n, n):
            raise ValueError(
                f"mean must have {n} values and cov {n} x {n}, one per "
                f"expression; they have {len(mean)} and {cov_.shape}")
        if not np.allclose(cov_, cov_.T):
            raise ValueError("cov must be symmetric")
        try:
            self._cholesky = np.linalg.cholesky(cov_)
        except np.linalg.LinAlgError:
            raise ValueError("cov must be positive definite") from None
        self._exprs = tuple(exprs)
        self._mean = tuple(float(m) for m in mean)
        self._cov = tuple(tuple(float(c) for c in row) for row in cov)

    def residuals(
            self,
            params: ParamSpace,
            functions: ParamFunctions | None = None
    ) -> Residuals:
        expressions = [
            params.expression(expr, functions) for expr in self._exprs]
        mean = np.array(self._mean)

        def residuals(values: ParamValues) -> np.ndarray:
            g = np.array([self._scalar(e(values), expr) for e, expr in zip(
                expressions, self._exprs)])
            return scipy.linalg.solve_triangular(
                self._cholesky, g - mean, lower=True)
        return residuals

    @staticmethod
    def _scalar(value: np.ndarray, expr: str) -> float:
        """Return the value of an expression, which must be one number."""
        if value.size != 1:
            raise ValueError(
                f"{expr}: each expression of an mvn prior must give one "
                f"value; it gives {value.size}")
        return float(value[0])


joint_prior_parser = TypedParser(JointPrior)
joint_prior_parser.register(JointPriorGauss)
joint_prior_parser.register(JointPriorSmooth)
joint_prior_parser.register(JointPriorSmooth2D)
joint_prior_parser.register(JointPriorBound)
joint_prior_parser.register(JointPriorMVN)


def joint_prior_residuals(
        params: ParamSpace,
        functions: ParamFunctions | None = None
) -> list[Residuals]:
    """
    Return the residual functions of the priors on several parameters of
    a parameter space (its priors: see ParamSpace), in order, with the
    functions of the models that the derived values they read call (see
    ParamSpace.derived_values).

    Raises
    ------
    ValueError
        If an expression of a prior is invalid (which the parameter space
        checks when it is made).
    """
    return [prior.residuals(params, functions) for prior in params.priors()]
