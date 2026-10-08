
import inspect
import textwrap
from collections.abc import Callable
from typing import Any

import numpy as np

from gbkfit.params.pdescs import ParamDesc
from gbkfit.params.space import ParamSpace
from gbkfit.utils import miscutils, parseutils


__all__ = [
    'EvaluationParams',
    'evaluation_params_parser',
    'load_function',
    'dump_function'
]


def load_function(info, desc):
    """Load a function from a file: info is {file: ..., func: ...}."""
    opts = parseutils.parse_options(info, desc, {'file', 'func'})
    return miscutils.get_attr_from_file(opts['file'], opts['func'])


def dump_function(func, file):
    """Append the source of a function to a file, and return its info."""
    with open(file, 'a') as f:
        f.write('\n')
        f.write(textwrap.dedent(inspect.getsource(func)))
        f.write('\n')
    return dict(file=file, func=func.__name__)


class EvaluationParams(parseutils.BasicSerializable):
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
        desc = parseutils.make_basic_desc(cls, 'params')
        info = dict(info)
        if 'transforms' in info:
            info['transforms'] = load_function(
                info['transforms'], 'params transforms')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['pdescs'])
        return cls(pdescs, **opts)

    def dump(self):
        return dict(properties=self._space.properties())

    def __init__(
            self,
            pdescs: dict[str, ParamDesc],
            properties: dict[str, Any],
            transforms: Callable | None = None
    ):
        self._space = ParamSpace(pdescs, properties, transforms)
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
