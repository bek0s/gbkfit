

from gbkfit.fitting import fitutils
from gbkfit.fitting.core import FitParam, FitParams, Fitter


def _make_eparams(theta, parameters):
    return dict(zip(
        parameters.enames(fixed=False, tied=False, free=True), theta))


def prior_transform(theta, parameters):
    eparams = _make_eparams(theta, parameters)
    foo = fitutils.nested_sampling_prior_transform(eparams, parameters)
    print("prior:", foo)
    return foo


def log_likelihood(theta, parameters, objective):
    eparams = _make_eparams(theta, parameters)
    foo = fitutils.log_likelihood_without_prior(eparams, parameters, objective)
    print("log_likelihood:", foo)
    return foo


class FitParamDynesty(FitParam):
    pass


class FitParamsDynesty(FitParams):
    pass


class FitterDynesty(Fitter):

    def __init__(self, options_init, options_run_nested):
        super().__init__()
        self._options_init = options_init
        self._options_run_nested = options_run_nested
