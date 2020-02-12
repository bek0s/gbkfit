

class Likelihood:

    def __init__(self, data, model):
        self._data = data
        self._model = model

    def log_likelihood(self):
        pass

    def residual_vector(self):
        pass

    def residual_scalar(self):
        pass


class LikelihoodExponential(Likelihood):
    pass


class LikelihoodGaussian(Likelihood):
    pass


class LikelihoodPoisson(Likelihood):
    pass


class LikelihoodStudentT(Likelihood):
    pass


class LikelihoodUniform(Likelihood):
    def __init__(self, data, model):
        super().__init__(data, model)