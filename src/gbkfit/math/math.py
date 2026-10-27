
import functools
import operator

import numpy as np


def is_odd(x):
    return int(x) % 2 != 0

def is_even(x):
    return int(x) % 2 == 0


is_even = np.vectorize(is_even)
is_odd = np.vectorize(is_odd)


def roundu_even(x):
    return 2 * np.ceil(0.5 * x)


def roundu_odd(x):
    x_even = roundu_even(x)
    return x_even + 1 if x_even - 1 < x else x_even - 1


def roundu_multiple(x, multiple):
    assert multiple > 0
    return x - np.mod(x, multiple) + multiple if np.mod(x, multiple) != 0 else x


def roundu_po2(x):
    power = 1
    while power < x:
        power *= 2
    return power


roundu_po2 = np.vectorize(roundu_po2)


def prod(x):
    return functools.reduce(operator.mul, x, 1)


def transform_lh_rotate_z(x, y, theta):
    """
    The same transform as the native transform_lh_rotate_z (gmodels), so
    that an angle theta is a position angle, measured from north (+y)
    through east (-x).
    """
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    out_x = - x * sintheta + y * costheta
    out_y = - x * costheta - y * sintheta
    return out_x, out_y


def gauss_fwhm_to_sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))


def gauss_1d_fun(x, a, b, c):
    return a * np.exp(- (x - b) * (x - b) / (2 * c * c))


def ggauss_1d_fun(x, a, b, c, d):
    return a * np.exp(-np.power(np.abs(x - b) / c, d))


def lorentz_1d_fun(x, a, b, c):
    return a * c * c / ((x - b) * (x - b) + c * c)


def moffat_1d_fun(x, a, b, c, d):
    return a * np.power(1 + np.power((x - b) / c, 2), -d)
