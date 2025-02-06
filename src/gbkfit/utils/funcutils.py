
import inspect

from collections.abc import Callable


def extract_args(func: Callable) -> tuple[list[str], list[str], list[str]]:
    """
    Extract required and optional arguments from a function.
    """
    required = []
    optional = []
    parameters = inspect.signature(func).parameters
    for name, info in parameters.items():
        if info.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        if info.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if info.default is inspect.Parameter.empty:
            required.append(name)
        else:
            optional.append(name)
    if 'self' in required:
        required.remove('self')
    return required + optional, required, optional
