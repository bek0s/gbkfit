
import abc


class Likelihood:

    def __init__(self, data, model):
        self._data = data
        self._model = model
        self._residual = []

        """
        for model_item in model:
            residual_item = {}
            for oname in model_item.onames():
                residual_item[oname] = np.
        """

    def residual_scalar(self):
        pass

    def residual_vector(self):
        pass

    @abc.abstractmethod
    def log_likelihood(self):
        pass


class LikelihoodGaussian(Likelihood):

    def __init__(self, data, model):
        super().__init__(data, model)

    def log_likelihood(self):
        pass


class LikelihoodStudentT(Likelihood):

    def __init__(self, data, model):
        super().__init__(data, model)

    def log_likelihood(self):
        pass
