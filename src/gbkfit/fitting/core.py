
import abc
import copy

from gbkfit.params import Interpreter
from gbkfit.params.utils import parse_param_values_strict
from gbkfit.utils import parseutils


class FitParam(parseutils.BasicParserSupport, abc.ABC):
    pass


class FitParams(parseutils.BasicParserSupport, abc.ABC):

    def __init__(self, descs, parameters, expressions, param_type):
        super().__init__()
        values, exprs = parse_param_values_strict(descs, parameters, param_type)
        self._descs = copy.deepcopy(descs)
        self._infos = values
        self._exprs = exprs
        self._parameters = copy.deepcopy(parameters)
        self._interpreter = Interpreter(descs, exprs, expressions)

    def descs(self):
        return self._descs

    def infos(self):
        return self._infos

    def exprs(self):
        return self._exprs

    def parameters(self):
        return self._parameters

    def interpreter(self):
        return self._interpreter


class Fitter(parseutils.TypedParserSupport, abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def load_params(info, desc):
        pass

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'fitter')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def __init__(self):
        pass

    def fit(self, objectives, parameters):
        if missing := set(objectives.pdescs()).difference(parameters.descs()):
            raise RuntimeError(
                f"fitting cannot start because information for the following "
                f"parameters is missing: {missing}")
        result = self._fit_impl(objectives, parameters)
        return result

    @abc.abstractmethod
    def _fit_impl(self, objective, params):
        pass


fitter_parser = parseutils.TypedParser(Fitter)
