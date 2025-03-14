
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
    'Params',
    'ParamProperty',
    'EvaluationParams',
    'evaluation_params_parser'
]


_log = logging.getLogger(__name__)


class ParamProperty:
    pass


class Params(abc.ABC):

    def __init__(
            self,
            pdescs: dict[str, ParamDesc],
            properties: dict[str, Any],
            property_types: type | tuple[()] | tuple[type],
            transforms: Callable | None
    ):
        values, expressions = (
            param_parsers.parse_param_values_strict(
                properties, pdescs, value_types=property_types))
        self._pdescs = copy.deepcopy(pdescs)
        self._values = values
        self._properties = copy.deepcopy(properties)
        self._expressions = copy.deepcopy(expressions)
        self._transforms = copy.deepcopy(transforms)
        self._interpreter = Interpreter(pdescs, expressions, transforms)

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._pdescs

    def properties(self) -> dict[str, Any]:
        return self._properties

    def expressions(self) -> dict[str, Any]:
        return self._expressions

    def transforms(self) -> Callable:
        return self._transforms

    def exploded_properties_with_values(self) -> dict[str, Any]:
        return self._values


class EvaluationParams(parseutils.BasicSerializable, Params):

    @classmethod
    def load(cls, info, *args, **kwargs):
        pdescs = kwargs.get('pdescs')
        if pdescs is None:
            raise RuntimeError("pdescs were not provided")
        desc = parseutils.make_basic_desc(cls, 'params')
        info = param_parsers.load_params_properties_transforms(
            info, pdescs, Real, lambda x: x)
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['pdescs'])
        return cls(pdescs, **opts)

    def dump(self, transforms_filename='transforms.py'):
        return param_parsers.dump_params_properties_transforms(
            self, ParamProperty, lambda x: x, transforms_filename)

    def __init__(
            self,
            pdescs: dict[str, ParamDesc],
            properties: dict[str, Any],
            transforms: Callable | None = None
    ):
        # By passing an empty tuple we treat all acceptable values as
        # expressions.
        super().__init__(pdescs, properties, (), transforms)

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
