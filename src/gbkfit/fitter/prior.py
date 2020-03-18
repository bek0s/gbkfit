
import abc

import numpy as np

import gbkfit.math


class Prior(abc.ABC):

    def __init__(self, minimum=-np.inf, maximum=np.inf):
        self.min = minimum
        self.max = maximum

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, minimum):
        self._min = minimum

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, maximum):
        self._max = maximum

    def sample(self, size):
        return self.rescale(np.random.uniform(0, 1, size))

    @abc.abstractmethod
    def rescale(self, x):
        pass

    @abc.abstractmethod
    def prob(self, x):
        pass

    def ln_prob(self, x):
        return np.log(self.prob(x))

    @abc.abstractmethod
    def cdf(self):
        pass


class PriorUniform(Prior):

    def __init__(self, minimum, maximum):
        super().__init__(minimum, maximum)

    def rescale(self, x):
        return self.min + x * (self.max - self.min)

    def prob(self, x):
        return ((x >= self.min) & (x <= self.max)) / (self.max - self.min)

    def ln_prob(self, x):
        pass


class PriorGauss(Prior):

    def __init__(self, mean, sigma):
        super().__init__()
        self._mean = mean
        self._sigma = sigma

    def rescale(self, x):
        pass

    def prob(self, x):
        return gbkfit.math.gauss_1d_pdf(x, self._mean, self._sigma)

    def ln_prob(self, x):
        pass
