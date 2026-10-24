import numpy as np


def cauchy_residuals(item, data, model, error, mask, params):
    """The residuals of Cauchy noise of the scale s times the errors."""
    x = (model - data) / (params['s'] * error)
    return np.sign(x) * np.sqrt(2 * np.log1p(x ** 2))


def student_residuals(item, data, model, error, mask, params):
    """
    The residuals of Student's t noise of nu degrees of freedom and the
    scale of the errors; a value the model lacks is 3 errors off.
    """
    nu = params['nu']
    x = np.where(np.isnan(model), 3.0, (model - data) / error)
    return np.sign(x) * np.sqrt((nu + 1) * np.log1p(x ** 2 / nu))


def jitter_log_likelihood(item, data, model, error, mask, params):
    """
    The log-likelihood of Gaussian noise of the errors and a jitter in
    quadrature; a value the model lacks is 3 sigma off.
    """
    sigma = np.hypot(error, params['jitter'])
    r = np.where(np.isnan(model), 3.0, (model - data) / sigma)
    kept = np.isfinite(data) if mask is None else mask != 0
    return -0.5 * np.sum(r[kept] ** 2 + np.log(2 * np.pi * sigma[kept] ** 2))
