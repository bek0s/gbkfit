"""
Helpers for functions.
"""

import inspect
import typing
from collections.abc import Callable


__all__ = [
    'ParameterNames',
    'parameter_names'
]


class ParameterNames(typing.NamedTuple):
    """
    The names of the parameters of a function, required (without a
    default value) and optional, in the order of its signature.
    """
    required: tuple[str, ...]
    optional: tuple[str, ...]

    @property
    def all(self) -> tuple[str, ...]:
        """Return the names of all the parameters, the required first."""
        return self.required + self.optional


def parameter_names(func: Callable) -> ParameterNames:
    """
    Return the names of the parameters of a function that can be passed
    by name.

    The first parameter of a method, self (e.g. of ``cls.__init__``), is
    left out, and so are ``*args``, ``**kwargs`` and the positional-only
    parameters, which cannot be passed by name.

    Parameters
    ----------
    func : Callable
        A function or a method.

    Returns
    -------
    ParameterNames
        The names of its required and optional parameters.
    """
    by_name = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY)
    required = []
    optional = []
    parameters = inspect.signature(func).parameters.values()
    for index, parameter in enumerate(parameters):
        if parameter.kind not in by_name:
            continue
        if index == 0 and parameter.name == 'self':
            continue
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter.name)
        else:
            optional.append(parameter.name)
    return ParameterNames(tuple(required), tuple(optional))
