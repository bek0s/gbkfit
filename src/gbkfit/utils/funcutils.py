
import inspect


def extract_args(func):
    required = []
    optional = []
    parameters = inspect.signature(func).parameters
    for pname, pinfo in parameters.items():
        if pinfo.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        if pinfo.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if pinfo.default is inspect.Parameter.empty:
            required.append(pname)
        else:
            optional.append(pname)
    if 'self' in required:
        required.remove('self')
    return required + optional, required, optional
