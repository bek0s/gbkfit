
import pygmo as pg


class Problem:

    def __init__(self, objective, parameters, minimums, maximums):
        self._objective = objective
        self._parameters = parameters
        self._minimums = minimums
        self._maximums = maximums
        self._bounds = (minimums, maximums)

    def get_bounds(self):
        return self._bounds

    def fitness(self, x):
        eparams_dict = dict(zip(self._parameters.names(), x))
        params_dict = self._parameters.expressions().evaluate(eparams_dict)
        return self._objective.residual_scalar(params_dict)

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x_: self.fitness(x_), x)
