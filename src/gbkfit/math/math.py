
import functools
import operator

import numpy as np
import scipy.special
import scipy.stats


def is_even(x):
    return not is_odd(x)


def is_odd(x):
    return int(x) & 1


def roundd_even(x):
    return 2 * np.floor(0.5 * x)


def roundu_even(x):
    return 2 * np.ceil(0.5 * x)


def roundd_odd(x):
    x_even = roundd_even(x)
    return x_even - 1 if x_even + 1 > x else x_even + 1


def roundu_odd(x):
    x_even = roundu_even(x)
    return x_even + 1 if x_even - 1 < x else x_even - 1


def roundd_multiple(x, multiple):
    assert multiple > 0
    return (x // multiple) * multiple


def roundu_multiple(x, multiple):
    assert multiple > 0
    return x - np.mod(x, multiple) + multiple if np.mod(x, multiple) != 0 else x


def roundd_po2(x):
    assert x > 1
    return roundu_po2(x) // 2


def roundu_po2(x):
    power = 1
    while power < x:
        power *= 2
    return power


def prod(x):
    return functools.reduce(operator.mul, x, 1)


def transform_lh_rotate_x(y, z, theta):
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    out_y = + y * costheta + z * sintheta
    out_z = - y * sintheta + z * costheta
    return out_y, out_z


def transform_lh_rotate_y(x, z, theta):
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    out_x = x * costheta - z * sintheta
    out_z = x * sintheta + z * costheta
    return out_x, out_z


def transform_lh_rotate_z(x, y, theta):
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    out_x = + x * costheta + y * sintheta
    out_y = - x * sintheta + y * costheta
    return out_x, out_y


def transform_rh_rotate_x(y, z, theta):
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    out_y = y * costheta - z * sintheta
    out_z = y * sintheta + z * costheta
    return out_y, out_z


def transform_rh_rotate_y(x, z, theta):
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    out_x = + x * costheta + z * sintheta
    out_z = - x * sintheta + z * costheta
    return out_x, out_z


def transform_rh_rotate_z(x, y, theta):
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    out_x = x * costheta - y * sintheta
    out_y = x * sintheta + y * costheta
    return out_x, out_y


def gauss_fwhm_to_sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))


def gauss_sigma_to_fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))


def lorentz_fwhm_to_gamma(fwhm):
    return fwhm / 2


def lorentz_gamma_to_fwhm(gamma):
    return gamma * 2


def moffat_fwhm_beta_to_alpha(fwhm, beta):
    return fwhm / (2 * np.sqrt(np.power(2, 1 / beta) - 1))


def moffat_alpha_beta_to_fwhm(alpha, beta):
    return 2 * alpha * np.sqrt(np.power(2, 1 / beta) - 1)


def _trunc_fun_1d(x, y, xmin, ymin, xmax, ymax):
    if isinstance(x, np.ndarray):
        if xmin is not None:
            y[np.where(x < xmin)] = ymin
        if xmax is not None:
            y[np.where(x > xmax)] = ymax
    else:
        if xmin is not None and x < xmin:
            y = ymin
        if xmax is not None and x > xmax:
            y = ymax
    return y


def _trunc_fun_1d_pdf(x, fun_pdf, fun_cdf, args, xmin, xmax):
    pdf_x = fun_pdf(x, *args)
    cdf_min = fun_cdf(xmin, *args)
    cdf_max = fun_cdf(xmax, *args)
    return _trunc_fun_1d(x, pdf_x / (cdf_max - cdf_min), xmin, 0, xmax, 0)


def _trunc_fun_1d_cdf(x, fun_cdf, args, xmin, xmax):
    cdf_x = fun_cdf(x, *args)
    cdf_min = fun_cdf(xmin, *args)
    cdf_max = fun_cdf(xmax, *args)
    cdf_scaled = (cdf_x - cdf_min) / (cdf_max - cdf_min)
    return _trunc_fun_1d(x, cdf_scaled, xmin, 0, xmax, 1)


def _trunc_fun_1d_ppf(x, fun_cdf, fun_ppf, args, xmin, xmax):
    # PPF_trunc = PPF(CDF(min) + u * (CDF(max) - CDF(min))
    fa = fun_cdf(xmin, *args)
    fb = fun_cdf(xmax, *args)
    return fun_ppf(fa + x * (fb - fa), *args)


def uniform_1d_fun(x, a, b, c):
    y = np.full_like(x, a) if isinstance(x, np.ndarray) else a
    return _trunc_fun_1d(x, y, b, 0, c, 0)


def uniform_1d_cdf(x, b, c):
    y = (x - b) / (c - b)
    return _trunc_fun_1d(x, y, b, 0, c, 1)


def uniform_1d_pdf(x, b, c):
    a = 1 / (c - b)
    return uniform_1d_fun(x, a, b, c)


def uniform_1d_ppf(u, b, c):
    return b + u * (c - b)


def expon_1d_fun(x, a, b, c):
    return (x >= 0) * (a * np.exp(-np.abs(x - b) / c))


def expon_1d_pdf(x, b, c):
    a = 1 / (2 * c)
    return expon_1d_fun(x, a, b, c)


def expon_1d_cdf(x, b, c):
    return (x >= 0) * (1 - np.exp(-np.abs(x - b) / c))


def expon_1d_ppf(x, b, c):
    return b - c * np.log(1 - x)


def expon_trunc_1d_fun(x, a, b, c, xmin, xmax):
    y = expon_1d_fun(x, a, b, c)
    return _trunc_fun_1d(x, y, xmin, 0, xmax, 0)


def expon_trunc_1d_cdf(x, b, c, xmin, xmax):
    cdf = expon_1d_cdf
    args = (b, c)
    return _trunc_fun_1d_cdf(x, cdf, args, xmin, xmax)


def expon_trunc_1d_pdf(x, b, c, xmin, xmax):
    pdf = expon_1d_pdf
    cdf = expon_1d_cdf
    args = (b, c)
    return _trunc_fun_1d_pdf(x, pdf, cdf, args, xmin, xmax)


def expon_trunc_1d_ppf(x, b, c, xmin, xmax):
    cdf = expon_1d_cdf
    ppf = expon_1d_ppf
    args = (b, c)
    return _trunc_fun_1d_ppf(x, cdf, ppf, args, xmin, xmax)


def laplace_1d_fun(x, a, b, c):
    return a * np.exp(-np.abs(x - b) / c)


def laplace_1d_pdf(x, b, c):
    a = 1 / (2 * c)
    return a * laplace_1d_fun(x, a, b, c)


def laplace_1d_cdf(x, b, c):
    return 0.5 + 0.5 * np.sign(x - b) * (1 - np.exp(-np.abs(x - b) / c))


def laplace_1d_ppf(x, b, c):
    return b - c * np.sign(x - 0.5) * np.log(1 - 2 * np.abs(x - 0.5))


def laplace_trunc_1d_fun(x, a, b, c, xmin, xmax):
    y = laplace_1d_fun(x, a, b, c)
    return _trunc_fun_1d(x, y, xmin, 0, xmax, 0)


def laplace_trunc_1d_pdf(x, b, c, xmin, xmax):
    pdf = laplace_1d_pdf
    cdf = laplace_1d_cdf
    args = (b, c)
    return _trunc_fun_1d_pdf(x, pdf, cdf, args, xmin, xmax)


def laplace_trunc_1d_cdf(x, b, c, xmin, xmax):
    cdf = laplace_1d_cdf
    args = (b, c)
    return _trunc_fun_1d_cdf(x, cdf, args, xmin, xmax)


def laplace_trunc_1d_ppf(x, b, c, xmin, xmax):
    cdf = laplace_1d_cdf
    ppf = laplace_1d_ppf
    args = (b, c)
    return _trunc_fun_1d_ppf(x, cdf, ppf, args, xmin, xmax)


def gauss_1d_fun(x, a, b, c):
    return a * np.exp(- (x - b) * (x - b) / (2 * c * c))


def gauss_1d_pdf(x, b, c):
    a = 1 / (c * np.sqrt(2 * np.pi))
    return gauss_1d_fun(x, a, b, c)


def gauss_1d_cdf(x, b, c):
    return 0.5 * (1 + scipy.special.erf((x - b)/(c * np.sqrt(2))))


def gauss_1d_ppf(x, b, c):
    return b + c * np.sqrt(2) * scipy.special.erfinv(2 * x - 1)


def gauss_trunc_1d_fun(x, a, b, c, xmin, xmax):
    y = gauss_1d_fun(x, a, b, c)
    return _trunc_fun_1d(x, y, xmin, 0, xmax, 0)


def gauss_trunc_1d_pdf(x, b, c, xmin, xmax):
    pdf = gauss_1d_pdf
    cdf = gauss_1d_cdf
    args = (b, c)
    return _trunc_fun_1d_pdf(x, pdf, cdf, args, xmin, xmax)


def gauss_trunc_1d_cdf(x, b, c, xmin, xmax):
    cdf = gauss_1d_cdf
    args = (b, c)
    return _trunc_fun_1d_cdf(x, cdf, args, xmin, xmax)


def gauss_trunc_1d_ppf(x, b, c, xmin, xmax):
    cdf = gauss_1d_cdf
    ppf = gauss_1d_ppf
    args = (b, c)
    return _trunc_fun_1d_ppf(x, cdf, ppf, args, xmin, xmax)


def ggauss_1d_fun(x, a, b, c, d):
    return a * np.exp(-np.power(np.abs(x - b) / c, d))


def ggauss_1d_pdf(x, b, c, d):
    a = d / (2 * c * scipy.special.gamma(1 / d))
    return ggauss_1d_fun(x, a, b, c, d)


def ggauss_1d_cdf(x, b, c, d):
    return 0.5 + 0.5 * np.sign(x - b) * scipy.special.gammainc(
            1 / d, np.power(np.abs(x - b) / c, d))


def ggauss_1d_ppf(x, b, c, d):
    arg0 = 2 * np.abs(x - 0.5)
    arg1 = 1 / d
    arg2 = 1 / np.power(c, d)
    gamma_ppf = scipy.stats.gamma.ppf(q=arg0, a=arg1, loc=0, scale=1/arg2)
    return np.sign(x - 0.5) * np.power(gamma_ppf, 1.0 / d) + b


def ggauss_trunc_1d_fun(x, a, b, c, d, xmin, xmax):
    y = ggauss_1d_fun(x, a, b, c, d)
    return _trunc_fun_1d(x, y, xmin, 0, xmax, 0)


def ggauss_trunc_1d_pdf(x, b, c, d, xmin, xmax):
    pdf = ggauss_1d_pdf
    cdf = ggauss_1d_cdf
    args = (b, c, d)
    return _trunc_fun_1d_pdf(x, pdf, cdf, args, xmin, xmax)


def ggauss_trunc_1d_cdf(x, b, c, d, xmin, xmax):
    cdf = ggauss_1d_cdf
    args = (b, c, d)
    return _trunc_fun_1d_cdf(x, cdf, args, xmin, xmax)


def ggauss_trunc_1d_ppf(x, b, c, d, xmin, xmax):
    cdf = ggauss_1d_cdf
    ppf = ggauss_1d_ppf
    args = (b, c, d)
    return _trunc_fun_1d_ppf(x, cdf, ppf, args, xmin, xmax)


def lorentz_1d_fun(x, a, b, c):
    return a * c * c / ((x - b) * (x - b) + c * c)


def lorentz_1d_pdf(x, b, c):
    a = 1 / (np.pi * c)
    return lorentz_1d_fun(x, a, b, c)


def lorentz_1d_cdf(x, b, c):
    return 0.5 + np.arctan((x - b) / c) / np.pi


def lorentz_1d_ppf(x, b, c):
    return b + c * np.tan(np.pi * (x - 0.5))


def lorentz_trunc_1d_fun(x, a, b, c, xmin, xmax):
    y = lorentz_1d_fun(x, a, b, c)
    return _trunc_fun_1d(x, y, xmin, 0, xmax, 0)


def lorentz_trunc_1d_pdf(x, b, c, xmin, xmax):
    pdf = lorentz_1d_pdf
    cdf = lorentz_1d_cdf
    args = (b, c)
    return _trunc_fun_1d_pdf(x, pdf, cdf, args, xmin, xmax)


def lorentz_trunc_1d_cdf(x, b, c, xmin, xmax):
    cdf = lorentz_1d_cdf
    args = (b, c)
    return _trunc_fun_1d_cdf(x, cdf, args, xmin, xmax)


def lorentz_trunc_1d_ppf(x, b, c, xmin, xmax):
    cdf = lorentz_1d_cdf
    ppf = lorentz_1d_ppf
    args = (b, c)
    return _trunc_fun_1d_ppf(x, cdf, ppf, args, xmin, xmax)


def moffat_1d_fun(x, a, b, c, d):
    return a * np.power(1 + np.power((x - b) / c, 2), -d)


def moffat_1d_pdf(x, b, c, d):
    a = (d - 1) / (c * c) * np.pi
    return moffat_1d_fun(x, a, b, c, d)


def moffat_1d_cdf(x, b, c, d):
    raise NotImplementedError()


def moffat_1d_ppf(x, b, c, d):
    raise NotImplementedError()


def moffat_trunc_1d_fun(x, a, b, c, d, xmin, xmax):
    y = moffat_1d_fun(x, a, b, c, d)
    return _trunc_fun_1d(x, y, xmin, 0, xmax, 0)


def moffat_trunc_1d_pdf(x, b, c, d, xmin, xmax):
    pdf = moffat_1d_pdf
    cdf = moffat_1d_cdf
    args = (b, c, d)
    return _trunc_fun_1d_pdf(x, pdf, cdf, args, xmin, xmax)


def moffat_trunc_1d_cdf(x, b, c, d, xmin, xmax):
    cdf = moffat_1d_cdf
    args = (b, c, d)
    return _trunc_fun_1d_cdf(x, cdf, args, xmin, xmax)


def moffat_trunc_1d_ppf(x, b, c, d, xmin, xmax):
    cdf = moffat_1d_cdf
    ppf = moffat_1d_ppf
    args = (b, c, d)
    return _trunc_fun_1d_ppf(x, cdf, ppf, args, xmin, xmax)


def sech2_1d_fun(x, a, b, c):
    return a * np.power(1 / np.cosh((x - b) / c), 2)


def sech2_1d_pdf(x, b, c):
    a = 1 / (2 * c)
    return sech2_1d_fun(x, a, b, c)


def sech2_1d_cdf(x, b, c):
    return 0.5 * (1 + np.tanh((x - b) / c))


def sech2_1d_ppf(x, b, c):
    raise NotImplementedError()


def sech2_trunc_1d_fun(x, a, b, c, xmin, xmax):
    y = sech2_1d_fun(x, a, b, c)
    return _trunc_fun_1d(x, y, xmin, 0, xmax, 0)


def sech2_trunc_1d_pdf(x, b, c, xmin, xmax):
    pdf = sech2_1d_pdf
    cdf = sech2_1d_cdf
    args = (b, c)
    return _trunc_fun_1d_pdf(x, pdf, cdf, args, xmin, xmax)


def sech2_trunc_1d_cdf(x, b, c, xmin, xmax):
    cdf = sech2_1d_cdf
    args = (b, c)
    return _trunc_fun_1d_cdf(x, cdf, args, xmin, xmax)


def sech2_trunc_1d_ppf(x, b, c, xmin, xmax):
    cdf = sech2_1d_cdf
    ppf = sech2_1d_ppf
    args = (b, c)
    return _trunc_fun_1d_ppf(x, cdf, ppf, args, xmin, xmax)
