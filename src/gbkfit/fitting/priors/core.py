
import abc

import numpy as np
import scipy.integrate
import scipy.interpolate

import gbkfit.math
from gbkfit.utils import parseutils


def _load(cls, prior_info, param_info):
    if 'min' not in prior_info and 'min' in param_info:
        prior_info['min'] = param_info.pop('min')
    if 'max' not in prior_info and 'max' in param_info:
        prior_info['max'] = param_info.pop('max')
    desc = ''
    opts = parseutils.parse_options_for_callable(
        prior_info, desc, cls.__init__, fun_rename_args=dict(
            minimum='min',
            maximum='max'))
    return opts


class Prior(parseutils.TypedParserSupport, abc.ABC):

    @classmethod
    def load(cls, info, **kwargs):
        return cls(**_load(cls, info, kwargs['param_info']))

    def dump(self):
        info = dict()
        if np.isfinite(self.minimum):
            info['min'] = self.minimum
        if np.isfinite(self.maximum):
            info['max'] = self.maximum
        return info

    def __init__(self, minimum=-np.inf, maximum=np.inf):
        self._minimum = minimum
        self._maximum = maximum

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

    def width(self):
        return self.maximum - self.minimum

    def sample(self, size):
        return self.rescale(np.random.uniform(0, 1, size))

    @abc.abstractmethod
    def rescale(self, x):
        pass

    @abc.abstractmethod
    def prob(self, x):
        pass

    def log_prob(self, x):
        # with np.errstate(divide='ignore'):
        return np.log(self.prob(x))

    def cdf(self, x):
        if np.any(np.isinf([self.minimum, self.maximum])):
            raise RuntimeError()
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

    def __init__(self, minimum, maximum):
        super().__init__(minimum, maximum)

    def rescale(self, x):
        return self.min + x * (self.max - self.min)

    def prob(self, x):
        return ((x >= self.min) & (x <= self.max)) / (self.max - self.min)

    def ln_prob(self, x):
        pass


class PriorGauss(Prior):

    @staticmethod
    def type():
        return 'gauss'

    def dump(self):
        info = super().dump()
        info.update(mean=self._mean, std=self._std)
        return info

    def __init__(self, mean, std):
        super().__init__()
        self._mean = mean
        self._std = std

    @property
    def mean(self):
        return self._minimum

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
        return self.mean + self.std * np.sqrt(2) * erfinv(2 * x - 1)

    def prob(self, x):
        return gbkfit.math.gauss_1d_pdf(x, self._mean, self._std)

    def ln_prob(self, x):
        pass


class PriorGaussTrunc(Prior):

    @staticmethod
    def type():
        return 'gauss_trunc'


class ConditionFunc:

    def dependencies(self):
        return 'cmp1_xpos'

    def __call__(self, *args, **kwargs):
        pass


class PriorDeltaFunction(Prior):

    def __init__(self, peak):
        super().__init__(peak, peak)
        self._peak = peak

    @property
    def peak(self):
        return self._peak

    @peak.setter
    def peak(self, peak):
        self._peak = peak

    def rescale(self, val):
        return self.peak * val ** 0

    def prob(self, val):
        at_peak = (val == self.peak)
        return np.nan_to_num(np.multiply(at_peak, np.inf))

    def cdf(self, val):
        pass


class PriorDict(dict):

    def __init__(self, dictionary, condition_func=None):
        super().__init__()
        for key, val in dictionary.items():
            pass
        self.update(dictionary)

    def sample(self, size=None):
        pass

    def prob(self, keys, vals):
        return np.product(
            [self[key].prob(val) for key, val in zip(keys, vals)])

    def log_prob(self, keys, vals):
        return np.product(
            [self[key].log_prob(val) for key, val in zip(keys, vals)])

    def rescale(self, keys, vals):
        return [self[key].rescale(val) for key, val in zip(keys, vals)]




prior_parser = parseutils.TypedParser(Prior)

prior_parser.register(PriorUniform)
prior_parser.register(PriorGauss)
