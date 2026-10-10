
from collections.abc import Callable
from typing import Any

import numpy as np

from gbkfit.params.pdescs import ParamDesc
from gbkfit.params.space import ParamSpace
from gbkfit.utils import funcutils, parseutils


__all__ = [
    'EvaluationParams',
    'evaluation_params_parser',
    'load_function',
    'dump_function'
]


def load_function(info):
    """Load a function from a file: info is {file: ..., func: ...}."""
    opts = parseutils.parse_options(info, {'file', 'func'})
    return funcutils.load_function_from_file(opts['file'], opts['func'])


def dump_function(func, file):
    """Append the source of a function to a file, and return its info."""
    with open(file, 'a') as f:
        f.write('\n')
        f.write(funcutils.function_source(func))
        f.write('\n')
    return dict(file=file, func=func.__name__)


class EvaluationParams(parseutils.Serializable):
    """
    The parameters of a model evaluation. The properties of free
    parameters (e.g. those of a fit configuration) are evaluated at their
    'value', so a fit configuration can be evaluated as it is.
    """

    @classmethod
    def load(cls, info, *args, **kwargs):
        pdescs = kwargs.get('pdescs')
        if pdescs is None:
            raise RuntimeError("pdescs were not provided")
        info = dict(info)
        if 'transforms' in info:
            info['transforms'] = parseutils.load_option(
                load_function, info, 'transforms')
        opts = parseutils.parse_options_for_callable(
            info, cls.__init__, ignore_params=['pdescs', 'constants'])
        return cls(pdescs, **opts, constants=kwargs.get('constants'))

    def dump(self):
        return dict(properties=self._space.properties())

    def __init__(
            self,
            pdescs: dict[str, ParamDesc],
            properties: dict[str, Any],
            transforms: Callable | None = None,
            constants: dict[str, Any] | None = None
    ):
        """
        constants are values that expressions can use (e.g. the radial
        nodes of the disks, from constants() of the models).
        """
        self._space = ParamSpace(pdescs, properties, transforms, constants)
        free = self._space.free_properties()
        if missing := [n for n, p in free.items() if 'value' not in p]:
            raise RuntimeError(
                f"these parameters have properties without a 'value', so "
                f"they cannot be evaluated: {missing}")
        self._free_values = {n: p['value'] for n, p in free.items()}

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._space.pdescs()

    def properties(self) -> dict[str, Any]:
        return self._space.properties()

    def evaluate(
            self, out_exploded_params: dict[str, float] | None = None
    ) -> dict[str, float | np.ndarray]:
        values = self._space.evaluate(self._free_values)
        if out_exploded_params is not None:
            out_exploded_params.update(self._space.exploded_values(values))
        return values


evaluation_params_parser = parseutils.BasicParser(EvaluationParams)
