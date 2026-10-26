"""
The errors of evaluations at the values of parameters where the objective
is not defined, which fitters take as points they cannot evaluate.
"""

__all__ = [
    'EvaluationError',
    'ModelEvaluationError',
    'ParamEvaluationError'
]


class EvaluationError(RuntimeError):
    """
    The objective is not defined at the given values of the parameters
    (see ParamEvaluationError and ModelEvaluationError): fitters take such
    a point as one they cannot evaluate (an infinite cost), not as the end
    of the fit.
    """


class ParamEvaluationError(EvaluationError):
    """
    An evaluation of the parameters failed: an expression or the
    transforms function failed, or a value is not finite or not valid.
    """


class ModelEvaluationError(EvaluationError):
    """
    A model or a solve failed at the given values of the parameters: e.g.
    bins without light of the orbits of a spheroid, a solve that does not
    converge, more clouds than the kernels draw.
    """
