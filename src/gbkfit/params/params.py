
import abc
import copy
from numbers import Real

from gbkfit.params.interpreter import Interpreter
from gbkfit.params import paramutils
from gbkfit.utils import parseutils


__all__ = ['Param', 'Params', 'EvaluationParams', 'evaluation_params_parser']


class Param(abc.ABC):
    pass


class Params(abc.ABC):

    def __init__(self, pdescs, parameters, expressions, conversions):
        self._pdescs = copy.deepcopy(pdescs)
        self._parameters = copy.deepcopy(parameters)
        self._expressions = copy.deepcopy(expressions)
        self._conversions = copy.deepcopy(conversions)
        self._interpreter = Interpreter(pdescs, expressions, conversions)

    def pdescs(self):
        return self._pdescs

    def parameters(self):
        return self._parameters

    def expressions(self):
        return self._expressions

    def conversions(self):
        return self._conversions


class EvaluationParams(parseutils.BasicParserSupport, Params):

    @classmethod
    def load(cls, info, pdescs):
        desc = parseutils.make_basic_desc(cls, 'params')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['pdescs'])
        opts = paramutils.load_params_parameters_conversions(
            opts, pdescs, Real, lambda x: x)
        return cls(pdescs, **opts)

    def dump(self, conversions_file):
        return paramutils.dump_params_parameters_conversions(
            self, Param, lambda x: x, conversions_file)

    def __init__(self, pdescs, parameters, conversions=None):
        expressions = paramutils.parse_param_values_strict(pdescs, parameters, ())[1]
        super().__init__(pdescs, parameters, expressions, conversions)

    def enames(self, fixed=True, tied=True):
        return self._interpreter.enames(fixed=fixed, tied=tied, free=False)

    def evaluate(self, out_eparams):
        return self._interpreter.evaluate(dict(), out_eparams, True)


evaluation_params_parser = parseutils.BasicParser(EvaluationParams)
