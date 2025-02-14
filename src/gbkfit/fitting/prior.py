
import abc
import collections.abc

import numpy as np
import scipy.integrate
import scipy.interpolate

import gbkfit.math
from gbkfit.utils import parseutils


__all__ = [
    'Prior',
    'PriorUniform',
    'PriorGauss',
    'PriorGaussTrunc'
]


def _recover_min_max(prior_info, param_info):
    for key in ['min', 'max']:
        # todo: emit info/warning or recovery or duplicate
        if key not in prior_info and key in param_info:
            prior_info[key] = param_info.pop(key)
    return prior_info


def _dump_min_max(prior):
    info = dict()
    for key, value in (('min', prior.minimum), ('max', prior.maximum)):
        if np.isfinite(value):
            info[key] = value
    return info


class Prior(parseutils.TypedSerializable, abc.ABC):

    def dump(self):
        return _dump_min_max(self)

    def __init__(self, minimum=-np.inf, maximum=np.inf, seed=0):
        self._minimum = minimum
        self._maximum = maximum
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def minimum(self):
        return self._minimum

    @minimum.setter
    def minimum(self, minimum):
        self._minimum = minimum

    @property
    def maximum(self):
        return self._maximum

    @maximum.setter
    def maximum(self, maximum):
        self._maximum = maximum

    def sample(self, size):
        return self.rescale(self._rng.uniform(0, 1, size))

    @abc.abstractmethod
    def rescale(self, x):
        pass

    @abc.abstractmethod
    def prob(self, x):
        pass

    def log_prob(self, x):
        prob = self.prob(x)
        return np.log(prob) if prob > 0 else 0

    def cdf(self, x):
        if np.any(np.isinf([self.minimum, self.maximum])):
            raise RuntimeError(
                f"cannot compute the cdf of non-truncated "
                f"prior '{self.type()}' numerically")
        xdata = np.linspace(self.minimum, self.maximum, 1000)
        pdf = self.prob(xdata)
        cdf = scipy.integrate.cumtrapz(pdf, xdata, initial=0)
        interp = scipy.interpolate.interp1d(
            xdata, cdf,
            assume_sorted=True, bounds_error=False, fill_value=(0, 1))
        return interp(x)


class PriorUniform(Prior):

    @staticmethod
    def type():
        return 'uniform'

    @classmethod
    def load(cls, info, **kwargs):
        info = _recover_min_max(info, **kwargs)
        desc = parseutils.make_typed_desc(cls, 'prior')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_rename_args=dict(
                minimum='min',
                maximum='max'))
        return cls(**opts)

    def dump(self):
        return super().dump()

    def __init__(
            self,
            minimum: int | float,
            maximum: int | float
    ):
        super().__init__(minimum, maximum)

    def rescale(self, x):
        return gbkfit.math.uniform_1d_ppf(x, self.minimum, self.maximum)

    def prob(self, x):
        return gbkfit.math.uniform_1d_pdf(x, self.minimum, self.maximum)

    def cdf(self, x):
        return gbkfit.math.uniform_1d_cdf(x, self.minimum, self.maximum)


class PriorGauss(Prior):

    @staticmethod
    def type():
        return 'gauss'

    @classmethod
    def load(cls, info, **kwargs):
        desc = parseutils.make_typed_desc(cls, 'prior')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        return super().dump() | dict(mean=self.mean, std=self.std)

    def __init__(
            self,
            mean: int | float,
            std: int | float
    ):
        super().__init__()
        self._mean = mean
        self._std = std

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = mean

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, std):
        self._std = std

    def rescale(self, x):
        return gbkfit.math.gauss_1d_ppf(x, self.mean, self.std)

    def prob(self, x):
        return gbkfit.math.gauss_1d_pdf(x, self.mean, self.std)

    def cdf(self, x):
        return gbkfit.math.gauss_1d_cdf(x, self.mean, self.std)


class PriorGaussTrunc(Prior):

    @staticmethod
    def type():
        return 'gauss_trunc'

    @classmethod
    def load(cls, info, **kwargs):
        info = _recover_min_max(info, **kwargs)
        desc = parseutils.make_typed_desc(cls, 'prior')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_rename_args=dict(
                minimum='min',
                maximum='max'))
        return cls(**opts)

    def dump(self):
        return super().dump() | dict(mean=self.mean, std=self.std)

    def __init__(
            self,
            mean: int | float,
            std: int | float,
            minimum: int | float,
            maximum: int | float
    ):
        super().__init__(minimum, maximum)
        self._mean = mean
        self._std = std

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = mean

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, std):
        self._std = std

    def rescale(self, x):
        return gbkfit.math.gauss_trunc_1d_ppf(
            x, self.mean, self.std, self.minimum, self.maximum)

    def prob(self, x):
        return gbkfit.math.gauss_trunc_1d_pdf(
            x, self.mean, self.std, self.minimum, self.maximum)

    def cdf(self, x):
        return gbkfit.math.gauss_trunc_1d_cdf(
            x, self.mean, self.std, self.minimum, self.maximum)


prior_parser = parseutils.TypedParser(Prior, [
    PriorUniform,
    PriorGauss])
