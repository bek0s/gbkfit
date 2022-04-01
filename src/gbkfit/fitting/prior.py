
import abc

import numpy as np
import scipy.integrate
import scipy.interpolate
from scipy.special import erf, erfinv, xlogy

import gbkfit.math
from gbkfit.utils import parseutils


def _parse_min_and_max(prior_info, param_info):
    for key in ['min', 'max']:
        if key not in prior_info and key in param_info:
            prior_info[key] = param_info.pop(key)
    return prior_info


class Prior(parseutils.TypedParserSupport, abc.ABC):

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

    def sample(self, size):
        return self.rescale(np.random.uniform(0, 1, size))

    @abc.abstractmethod
    def rescale(self, x):
        pass

    @abc.abstractmethod
    def prob(self, x):
        pass

    def log_prob(self, x):
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

    @classmethod
    def load(cls, info, **kwargs):
        info = _parse_min_and_max(info, **kwargs)
        desc = parseutils.make_typed_desc(cls, 'prior')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_rename_args=dict(
                minimum='min',
                maximum='max'))
        return cls(**opts)

    def dump(self):
        return super().dump()

    def __init__(self, minimum, maximum):
        super().__init__(minimum, maximum)

    def rescale(self, x):
        return self.minimum + x * (self.maximum - self.minimum)

    def prob(self, x):
        in_range = (x >= self.minimum) & (x <= self.maximum)
        return in_range / (self.maximum - self.minimum)

    def ln_prob(self, x):
        in_range = (x >= self.minimum) & (x <= self.maximum)
        return xlogy(1, in_range) - xlogy(1, self.maximum - self.minimum)

    def cdf(self, x):
        cdf = (x - self.minimum) / (self.maximum - self.minimum)
        cdf = np.minimum(cdf, 1)
        cdf = np.maximum(cdf, 0)
        return cdf


class PriorGauss(Prior):

    @staticmethod
    def type():
        return 'gauss'

    @classmethod
    def load(cls, info, **kwargs):
        desc = parseutils.make_typed_desc(cls, 'prior')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        return super().dump() | dict(mean=self.mean, std=self.std)

    def __init__(self, mean, std):
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

    def ln_prob(self, x):
        return -0.5 * ((self.mean - x) ** 2 / self.std ** 2
                       + np.log(2 * np.pi * self.std ** 2))

    def cdf(self, x):
        return gbkfit.math.gauss_1d_cdf(x, self.mean, self.std)


class PriorGaussTrunc(Prior):

    @staticmethod
    def type():
        return 'gauss_trunc'

    @classmethod
    def load(cls, info, **kwargs):
        info = _parse_min_and_max(info, **kwargs)
        desc = parseutils.make_typed_desc(cls, 'prior')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_rename_args=dict(
                minimum='min',
                maximum='max'))
        return cls(**opts)

    def __init__(self, mean, std, minimum, maximum):
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

    def __init__(self, dictionary):
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
