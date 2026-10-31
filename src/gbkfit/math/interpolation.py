
import abc

import scipy.interpolate


class Interpolator(abc.ABC):

    @staticmethod
    def type():
        pass


class InterpolatorLinear(Interpolator):

    @staticmethod
    def type():
        return 'linear'

    def __init__(self, x, y):
        self._interp = scipy.interpolate.interp1d(x, y, 'linear')

    def __call__(self, x):
        return self._interp(x)


class InterpolatorAkima(Interpolator):

    @staticmethod
    def type():
        return 'akima'

    def __init__(self, x, y):
        self._interp = scipy.interpolate.Akima1DInterpolator(x, y)

    def __call__(self, x):
        return self._interp(x)


class InterpolatorPCHIP(Interpolator):

    @staticmethod
    def type():
        return 'pchip'

    def __init__(self, x, y):
        self._interp = scipy.interpolate.PchipInterpolator(x, y)

    def __call__(self, x):
        return self._interp(x)
