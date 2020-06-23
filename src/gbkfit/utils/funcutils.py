
import inspect


def extract_args(func):
    required = set()
    optional = set()
    parameters = inspect.signature(func).parameters
    for pname, pinfo in parameters.items():
        if pinfo.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        if pinfo.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if pinfo.default is inspect.Parameter.empty:
            required.add(pname)
        else:
            optional.add(pname)
    required.discard('self')
    return required, optional
