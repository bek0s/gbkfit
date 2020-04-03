
import abc

import copy

import numpy as np

from gbkfit.utils import iterutils


class Objective(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def type():
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, info):
        pass

    @abc.abstractmethod
    def dump(self):
        pass

    def __init__(self):
        pass


class Likelihood(Objective):

    def __init__(self, data, model):
        super().__init__()

        self.data = data
        self.model = model

    def residual(self):
        return self.model - self.data

    def likelihood(self):
        return 1

    def log_likelihood(self):
        return np.log(self.likelihood())


class JointLikelihood(Likelihood):
    def __init__(self, likelihoods, weights=None):

        super(Objective, self).__init__()

        likelihoods = iterutils.tuplify(copy.deepcopy(likelihoods))
        weights = iterutils.tuplify(copy.deepcopy(weights))

    def log_likelihood(self):
        pass

