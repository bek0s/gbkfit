
import numpy as np
import pygmo as pg

from gbkfit.fitting import fitutils


class Problem:

    def __init__(
            self, objective, parameters, minimums, maximums,
            multi_objective, callback=None):
        self._objective = objective
        self._parameters = parameters
        self._minimums = minimums
        self._maximums = maximums
        self._multi_objective = multi_objective
        self._callback = callback
        self._bounds = (minimums, maximums)

    def get_bounds(self):
        return self._bounds

    def get_nobj(self):
        return 1 + int(self._multi_objective)

    def fitness(self, x):
        enames = self._parameters.enames(fixed=False, tied=False, free=True)
        eparams = dict(zip(enames, x))
        residual = fitutils.residual_scalar(
            eparams, self._parameters, self._objective, None)
        print(enames)
        print(eparams)
        print(residual)
        return [sum(residual)] + [sum(residual)] * self._multi_objective

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x_: self.fitness(x_), x)  # noqa
