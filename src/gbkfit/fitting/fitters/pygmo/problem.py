
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
        def _residual_scalar(x, objective, params):
            eparams_dict = dict(zip(params.names(), x))
            params_dict = params.expressions().evaluate(eparams_dict)
            return objective.residual_scalar(params_dict)
        return _residual_scalar(x, self._objective, self._parameters)

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x_: self.fitness(x_), x)
