"""
Helpers for functions: the names of their parameters, and functions
loaded from Python files.
"""

import functools
import importlib.util
import inspect
import pathlib
import typing
from collections.abc import Callable
from typing import Any


__all__ = [
    'ParameterNames',
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
        """The names of all the parameters, the required first."""
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


def load_function_from_file(file_path: str, name: str) -> Callable:
    """
    Return a function defined in a Python file.

    The file is run as a module of its own, which is not added to
    sys.modules. Functions of such modules cannot be pickled, so the
    function is returned in a wrapper that can be: a copy of it (e.g.
    sent to another process) runs the file again there to get the
    function, so the code at the top level of the file runs again in each
    process. inspect.unwrap gives the function itself.

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
    return _FileFunction(str(pathlib.Path(file_path).resolve()), name)


class _FileFunction:
    """
    A function of a Python file (see load_function_from_file). It pickles
    as the absolute path of its file and the function's name, so that a
    copy can run the file again, from any directory.
    """

    def __init__(self, file_path: str, name: str):
        self._file_path = file_path
        self._name = name
        self._func = _run_file_function(file_path, name)
        functools.update_wrapper(self, self._func)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)

    def __reduce__(self) -> tuple[type, tuple[str, str]]:
        return _FileFunction, (self._file_path, self._name)

    def __repr__(self) -> str:
        return f"<function {self._name} of '{self._file_path}'>"


def _run_file_function(file_path: str, name: str) -> Callable:
    """
    Run a Python file as a module of its own, and return its function of
    the given name (see load_function_from_file).
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
