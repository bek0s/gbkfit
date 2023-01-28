
import abc

from gbkfit.params.parsers import parse_param_values_strict
from gbkfit.utils import parseutils

from gbkfit.params.params import Param, Params


class FitParam(Param, abc.ABC):
    pass


class FitParams(Params, abc.ABC):

    def __init__(self, descs, parameters, conversions, param_type):
        infos, exprs = parse_param_values_strict(parameters, descs, param_type)
        super().__init__(descs, parameters, exprs, conversions)
        self._infos = infos

    def infos(self):
        return self._infos

    def enames(self, fixed=True, tied=True, free=True):
        return self._interpreter.enames(fixed=fixed, tied=tied, free=free)

    def evaluate(self, in_eparams, out_eparams=None, check=True):
        return self._interpreter.evaluate(in_eparams, out_eparams, check)


class Fitter(parseutils.TypedParserSupport, abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def load_params(info, desc):
        pass

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, objectives, parameters):
        if missing := set(objectives.pdescs()).difference(parameters.pdescs()):
            raise RuntimeError(
                f"fitting cannot start because information for the following "
                f"parameters is missing: {missing}")
        result = self._fit_impl(objectives, parameters)
        return result

    @abc.abstractmethod
    def _fit_impl(self, objective, params):
        pass


fitter_parser = parseutils.TypedParser(Fitter)
