
import numpy as np
from pymoo.core.problem import Problem

from gbkfit.fitting import fitutils


class PymooProblem(Problem):

    def __init__(self, objective, parameters):
        self._objective = objective
        self._parameters = parameters
        n_var = len(parameters.infos())
        xl = np.array([p.minimum() for p in parameters.infos().values()])
        xu = np.array([p.maximum() for p in parameters.infos().values()])
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        enames = self._parameters.enames(fixed=False, tied=False, free=True)
        residuals = []
        for i, values in enumerate(x):
            eparams = dict(zip(enames, values))
            residuals.append(sum(
                fitutils.residual_scalar(
                    eparams, self._parameters, self._objective, None)))
        out['F'] = np.array(residuals)
