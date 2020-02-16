
import abc

import numpy as np
import scipy.interpolate


class Interpolator(abc.ABC):
    pass


class InterpolatorLinear(Interpolator):

    def __init__(self, x, y):
        self._interp = scipy.interpolate.interp1d(x, y, 'linear')

    def __call__(self, x):
        return self._interp(x)


class InterpolatorAkima(Interpolator):

    def __init__(self, x, y):
        self._interp = scipy.interpolate.Akima1DInterpolator(x, y)

    def __call__(self, x):
        return self._interp(x)


class InterpolatorPCHIP(Interpolator):

    def __init__(self, x, y):
        self._interp = scipy.interpolate.PchipInterpolator(x, y)

    def __call__(self, x):
        return self._interp(x)
