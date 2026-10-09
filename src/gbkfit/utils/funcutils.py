"""
Helpers for functions.
"""

import importlib.util
import inspect
import pathlib
import textwrap
import typing
from collections.abc import Callable


__all__ = [
    'ParameterNames',
    'function_source',
    'load_function_from_file',
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


def function_source(func: Callable) -> str:
    """
    Return the source code of a function.

    Parameters
    ----------
    func : Callable
        A function defined in a file.

    Returns
    -------
    str
        Its source code, without the indentation of its definition.

    Raises
    ------
    OSError
        If its source code is not available (e.g. it was defined in an
        interactive session).
    """
    return textwrap.dedent(inspect.getsource(func))


def load_function_from_file(file_path: str, name: str) -> Callable:
    """
    Return a function defined in a Python file.

    The file is run as a module of its own, which is not added to
    sys.modules.

    Parameters
    ----------
    file_path : str
        The path of the file.
    name : str
        The name of the function.

    Returns
    -------
    Callable
        The function.

    Raises
    ------
    RuntimeError
        If the file cannot be run, or does not define a function of the
        given name.
    """
    module_name = pathlib.Path(file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load a module from '{file_path}'")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"error running '{file_path}': {e}") from e
    func = getattr(module, name, None)
    if not callable(func):
        raise RuntimeError(
            f"'{file_path}' does not define a function '{name}'")
    return func
