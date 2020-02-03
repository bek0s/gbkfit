
import numpy as np
import scipy.special


def is_even(num):
    return not is_odd(num)


def is_odd(num):
    return int(num) & 1


def roundd_even(num):
    return 2 * np.floor(0.5*num)


def roundu_even(num):
    return 2 * np.ceil(0.5*num)


def roundd_odd(num):
    num_even = roundd_even(num)
    return num_even - 1 if num_even + 1 > num else num_even + 1


def roundu_odd(num):
    num_even = roundu_even(num)
    return num_even + 1 if num_even - 1 < num else num_even - 1


def roundd_multiple(num, multiple):
    return (num // multiple) * multiple


def roundu_multiple(num, multiple):
    return num - np.mod(num, multiple) + multiple \
        if np.mod(num, multiple) is not 0 else num


def roundd_po2(num):
    assert num > 1
    return roundu_po2(num) // 2


def roundu_po2(num):
    power = 1
    while power < num:
        power *= 2
    return power


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


def _trunc_pdf(x, fun_pdf, fun_cdf, args, tmin, tmax):
    pdf = fun_pdf(x, *args)
    cdf_tmin = fun_cdf(tmin, *args) if tmin is not None else None
    cdf_tmax = fun_cdf(tmax, *args) if tmax is not None else None
    area = 1
    indices = []
    if (tmin and tmax) is not None:
        area = cdf_tmax - cdf_tmin
        indices = np.logical_or(x < tmin, x > tmax)
    elif tmin is not None and tmax is None:
        area = 1 - cdf_tmin
        indices = np.where(x < tmin)
    elif tmin is None and tmax is not None:
        area = cdf_tmax
        indices = np.where(x > tmax)
    pdf /= area
    pdf[indices] = 0
    return pdf


def uniform_1d_fun(x, a, x0, s):
    res = np.empty_like(x)
    lt = x < x0 - s
    gt = x > x0 + s
    res[:] = a
    res[lt] = 0
    res[gt] = 0
    return res


def uniform_1d_pdf(x, x0, s):
    a = 1 / (2 * s)
    return uniform_1d_fun(x, a, x0, s)


def uniform_1d_rnd(u, b, c):
    return b + c * u


def expon_1d_fun(x, a, b, c):
    return a * np.exp(-np.abs(x - b) / c)


def expon_1d_cdf(x, b, c):
    return 0.5 + 0.5 * np.sign(x - b) * (1 - np.exp(-np.abs(x - b) / c))


def expon_1d_pdf(x, b, c):
    a = 1 / (2 * c)
    return expon_1d_fun(x, a, b, c)


def expon_1d_pdf_trunc(x, b, c, xmin, xmax):
    return _trunc_pdf(x, expon_1d_pdf, expon_1d_cdf, (b, c), xmin, xmax)


def expon_1d_rnd(u, b, c):
    return b - c * np.sign(u) * np.log(1 - 2 * np.abs(u))


def gauss_1d_fun(x, a, b, c):
    return a * np.exp(- (x - b) * (x - b) / (2 * c * c))


def gauss_1d_cdf(x, b, c):
    return 0.5 * (1 + scipy.special.erf((x - b)/(c * np.sqrt(2))))


def gauss_1d_pdf(x, b, c):
    a = 1 / (c * np.sqrt(2 * np.pi))
    return gauss_1d_fun(x, a, b, c)


def gauss_1d_pdf_trunc(x, b, c, xmin, xmax):
    return _trunc_pdf(x, gauss_1d_pdf, gauss_1d_cdf, (b, c), xmin, xmax)


def gauss_1d_rnd(u1, u2, b, c):
    return b + c * np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)


def ggauss_1d_fun(x, a, b, c, d):
    return a * np.exp(-np.power(np.abs(x - b) / c, d))


def ggauss_1d_cdf(x, b, c, d):
    return 0.5 + 0.5 * np.sign(x - b) * scipy.special.gammainc(
            1 / d, np.power(np.abs(x - b) / c, d))


def ggauss_1d_pdf(x, b, c, d):
    a = d / (2 * c * scipy.special.gamma(1 / d))
    return ggauss_1d_fun(x, a, b, c, d)


def ggauss_1d_pdf_trunc(x, b, c, d, xmin, xmax):
    return _trunc_pdf(x, ggauss_1d_pdf, ggauss_1d_cdf, (b, c, d), xmin, xmax)


def lorentz_1d_fun(x, a, b, c):
    return a * c * c / ((x - b) * (x - b) + c * c)


def lorentz_1d_cdf(x, b, c):
    return 0.5 + np.arctan((x - b) / c) / np.pi


def lorentz_1d_pdf(x, b, c):
    a = 1 / (np.pi * c)
    return lorentz_1d_fun(x, a, b, c)


def lorentz_1d_pdf_trunc(x, b, c, xmin, xmax):
    return _trunc_pdf(x, lorentz_1d_pdf, lorentz_1d_cdf, (b, c), xmin, xmax)


def lorentz_1d_rnd(u, b, c):
    return b + c * np.tan(np.pi * 0.5 * u)


def moffat_1d_fun(x, a, b, c, d):
    return a * np.power(1 + np.power((x - b) / c, 2), -d)


"""
def moffat_1d_cdf(x, b, c, d):
    raise NotImplementedError()
"""


def moffat_1d_pdf(x, b, c, d):
    a = (d - 1) / (c * c) * np.pi
    return moffat_1d_fun(x, a, b, c, d)


"""
def moffat_1d_pdf_trunc(x, b, c, d, xmin, xmax):
    return _trunc_pdf(x, moffat_1d_pdf, moffat_1d_cdf, (b, c, d), xmin, xmax)
"""


"""
def moffat_1d_rnd(u, b, c, d):
    raise NotImplementedError()
"""


def sech2_1d_fun(x, a, b, c):
    return a * np.power(1 / np.cosh((x - b) / c), 2)


def sech2_1d_cdf(x, b, c):
    return 0.5 * (1 + np.tanh((x - b) / c))


def sech2_1d_pdf(x, b, c):
    a = 1 / (2 * c)
    return sech2_1d_fun(x, a, b, c)


def sech2_1d_pdf_trunc(x, b, c, xmin, xmax):
    return _trunc_pdf(x, sech2_1d_pdf, sech2_1d_cdf, (b, c), xmin, xmax)


def sech2_1d_rnd(u, b, c):
    return b + c * np.arctanh(u)


def _calculate_ellipse_abc(size_x, size_y, phi):
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    size_x2 = np.power(size_x, 2)
    size_y2 = np.power(size_y, 2)
    a = np.power(cosphi / size_x, 2) + np.power(sinphi / size_y, 2)
    b = np.power(sinphi / size_x, 2) + np.power(cosphi / size_y, 2)
    c = 2 * sinphi * cosphi * ((1 / size_x2) - (1 / size_y2))
    return a, b, c


def gauss_2d_(x, y, ampl, x0, y0, sigma_x, sigma_y, phi):
    a, b, c = _calculate_ellipse_abc(sigma_x, sigma_y, phi)
    return ampl * np.exp(-0.5 * (a * np.power(x - x0, 2) +
                                 b * np.power(y - y0, 2) +
                                 c * (x - x0) * (y - y0)))


def lorentz_2d_(x, y, ampl, x0, y0, gamma_x, gamma_y, phi):
    a, b, c = _calculate_ellipse_abc(gamma_x, gamma_y, phi)
    return ampl / (a * np.power(x - x0, 2) +
                   b * np.power(y - y0, 2) +
                   c * (x - x0) * (y - y0) + 1)


def lorentz_2d(x, y, ampl, x0, y0, gamma_x, gamma_y, phi):
    x, y = transform_lh_rotate_z(x - x0, y - y0, phi)
    return ampl * 1 / (1 + np.power(x / gamma_x, 2) + np.power(y / gamma_y, 2))


def moffat_2d_(x, y, ampl, x0, y0, alpha_x, alpha_y, beta, phi):
    a, b, c = _calculate_ellipse_abc(alpha_x, alpha_y, phi)
    return ampl / np.power((a * np.power(x - x0, 2) +
                            b * np.power(y - y0, 2) +
                            c * (x - x0) * (y - y0) + 1), beta)
