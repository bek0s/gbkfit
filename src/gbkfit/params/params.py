
import abc
import copy
import logging
import numbers
from collections.abc import Callable
from numbers import Real

from typing import Any
import numpy as np

from gbkfit.params import parsers as param_parsers
from gbkfit.params.interpreter import Interpreter
from gbkfit.params.pdescs import ParamDesc
from gbkfit.utils import parseutils


__all__ = [
    'Param',
    'Params',
    'EvaluationParams',
    'evaluation_params_parser'
]


_log = logging.getLogger(__name__)


class Param(abc.ABC):
    pass


class Params(abc.ABC):

    def __init__(
            self,
            pdescs: dict[str, ParamDesc],
            parameters,
            expressions,
            conversions
    ):
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


# class EvaluationParams2(parseutils.BasicSerializable, Params):
#
#     @classmethod
#     def l


# type ParamDescType = ParamScalarDesc | ParamVectorDesc

class EvaluationParams(parseutils.BasicSerializable, Params):

    @classmethod
    def load(cls, info, *args, **kwargs):
        pdescs = kwargs.get('pdescs')
        if pdescs is None:
            raise RuntimeError("pdescs were not provided")
        desc = parseutils.make_basic_desc(cls, 'params')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args={'pdescs'})
        opts = param_parsers.load_params_parameters_conversions(
            opts, pdescs, Real, lambda x: x)
        return cls(pdescs, **opts)

    def dump(self, conversions_file):
        return param_parsers.dump_params_parameters_conversions(
            self, Param, lambda x: x, conversions_file)

    def __init__(
            self,
            pdescs: dict[str, ParamDesc],
            parameters: dict[str, Any],
            conversions: Callable | None = None
    ):
        values, expressions = param_parsers.parse_param_values_strict(
            parameters, pdescs,
            # Make everything to be considered an expression
            value_types=(numbers.Real,))
        print(values)
        print(expressions)
        self._v = values
        super().__init__(pdescs, parameters, expressions, conversions)

    def names(self, fixed: bool = True, tied: bool = True) -> list[str]:
        return self._interpreter.enames(fixed=fixed, tied=tied, free=False)



    def evaluate(self, out_eparams, check=True):
        return self._interpreter.evaluate(self._v, check, out_eparams)


evaluation_params_parser = parseutils.BasicParser(EvaluationParams)
