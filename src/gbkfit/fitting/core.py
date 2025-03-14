
import abc
from collections.abc import Callable
from typing import Any

from gbkfit.params import ParamDesc
from gbkfit.utils import parseutils

from gbkfit.params.params import ParamProperty, Params


class FittingParamProperty(ParamProperty, abc.ABC):
    pass


class FittingParams(Params, abc.ABC):

    def __init__(
            self,
            pdescs: dict[str, ParamDesc],
            properties: dict[str, Any],
            property_types: type | tuple[()] | tuple[type],
            transforms: Callable | None
    ):
        super().__init__(pdescs, properties, property_types, transforms)

    def exploded_names(
            self, fixed: bool = True, tied: bool = True, free: bool = True
    ) -> list[str]:
        return self._interpreter.exploded_names(
            fixed=fixed, tied=tied, free=free)

    def evaluate(self, in_eparams, out_eparams=None, check=True):
        return self._interpreter.evaluate(in_eparams, check, out_eparams)


class Fitter(parseutils.TypedSerializable, abc.ABC):

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
