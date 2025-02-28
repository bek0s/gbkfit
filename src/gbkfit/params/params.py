
import abc
import copy
import logging
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


class ParamProperty:
    pass


class FittingParamProperty(ParamProperty):
    pass


class ModelParams:
    pass


class FittingParams:
    pass


class Params(abc.ABC):

    def __init__(
            self,
            pdescs: dict[str, ParamDesc],
            expressions_dict: dict[str, Any],
            expressions_func: Callable
    ):
        self._pdescs = copy.deepcopy(pdescs)
        self._expressions_dict = copy.deepcopy(expressions_dict)
        self._expressions_func = copy.deepcopy(expressions_func)
        self._interpreter = Interpreter(
            pdescs, expressions_dict, expressions_func)

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._pdescs

    def expressions_dict(self) -> dict[str, Any]:
        return self._expressions_dict

    def conversions_func(self) -> Callable:
        return self._expressions_func


class EvaluationParams(parseutils.BasicSerializable, Params):

    @classmethod
    def load(cls, info, *args, **kwargs):
        pdescs = kwargs.get('pdescs')
        if pdescs is None:
            raise RuntimeError("pdescs were not provided")
        desc = parseutils.make_basic_desc(cls, 'params')
        info = param_parsers.load_params_parameters_conversions(
            info, pdescs, Real, lambda x: x)
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args={'pdescs'})
        return cls(pdescs, **opts)

    def dump(self, conversions_file):
        return param_parsers.dump_params_parameters_conversions(
            self, Param, lambda x: x, conversions_file)

    def __init__(
            self,
            pdescs: dict[str, ParamDesc],
            properties: dict[str, Any],
            expressions_func: Callable | None = None
    ):
        # Treat all acceptable values as expressions because
        # we will be passing them to super()._interpreter which can
        # handle everything correctly. No need to complicate things.
        values_dict, expressions_dict = (
            param_parsers.parse_param_values_strict(
                properties, pdescs, value_types=()))
        # Invalid values should have been caught by now.
        if values_dict:
            raise RuntimeError("impossible")
        # print(f"values_dict: {values_dict}")
        # print(f"expressions_dict: {expressions_dict}")
        super().__init__(pdescs, expressions_dict, expressions_func)

    def exploded_names(
            self, fixed: bool = True, tied: bool = True
    ) -> list[str]:
        return self._interpreter.exploded_names(
            fixed=fixed, tied=tied, free=False)

    def evaluate(
            self,
            check: bool = True,
            out_exploded_params: dict[str, Real] | None = None
    ) -> dict[str, Real | np.ndarray]:
        return self._interpreter.evaluate({}, check, out_exploded_params)


evaluation_params_parser = parseutils.BasicParser(EvaluationParams)
