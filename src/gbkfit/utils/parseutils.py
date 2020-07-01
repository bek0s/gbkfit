
import abc
import copy
import inspect
import logging

from . import funcutils, iterutils


_log = logging.getLogger(__name__)


def _replace_items(data, mappings):
    data = list(data)
    for k, v in mappings.items():
        if k in data:
            data[data.index(k)] = v
    return set(data)


def parse_options(
        info, desc, add_required=None, add_optional=None,
        fun=None, fun_ignore_args=None, fun_rename_args=None):
    info = copy.deepcopy(info)
    required = set()
    optional = set()
    if add_required:
        required.update(add_required)
    if add_optional:
        optional.update(add_optional)
    # Infer required and optional options from callable
    if fun:
        fun_required, fun_optional = funcutils.extract_args(fun)
        if fun_ignore_args:
            fun_required.difference_update(fun_ignore_args)
            fun_optional.difference_update(fun_ignore_args)
        if fun_rename_args:
            for arg_name, opt_name in fun_rename_args.items():
                if opt_name in info:
                    info[arg_name] = info.pop(opt_name)
        required.update(fun_required)
        optional.update(fun_optional)
    # Check for missing or unknown options
    unknown = set(info) - (required | optional)
    missing = required - set(info)
    if unknown:
        if fun_rename_args:
            unknown = _replace_items(unknown, fun_rename_args)
        _log.warning(
            f"the following {desc} options are "
            f"not recognised and will be ignored: {str(unknown)}")
    if missing:
        if fun_rename_args:
            missing = _replace_items(missing, fun_rename_args)
        raise RuntimeError(
            f"the following {desc} options are "
            f"required but missing: {str(missing)}")
    # Return all recognised options
    return {k: v for k, v in info.items() if k in required | optional}


def parse_class_args(cls, info, add_args=(), skip_args=()):
    type_name = f' (type: {cls.type()})' if getattr(cls, 'type', None) else ''
    type_desc = f'{cls.__qualname__}{type_name}'
    required = set()
    optional = set()
    signature = inspect.signature(cls.__init__)
    for pname, pinfo in list(signature.parameters.items())[1:]:
        if pname in skip_args:
            continue
        if pinfo.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        if pinfo.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if pinfo.default is inspect.Parameter.empty:
            required.add(pname)
        else:
            optional.add(pname)
    required.update(add_args)
    required.difference_update(skip_args)
    optional.difference_update(skip_args)
    unknown = set(info) - (required | optional)
    missing = required - set(info)
    options = {k: v for k, v in info.items() if k in required | optional}
    if unknown:
        _log.warning(
            f"the following {type_desc} options are "
            f"not recognised and will be ignored: {', '.join(unknown)}")
    if missing:
        raise RuntimeError(
            f"the following {type_desc} options are "
            f"required but missing: {', '.join(missing)}")
    return options


class Parser(abc.ABC):

    def __init__(self, cls):
        self._clstype = cls
        self._clsname = cls.__name__

    def clstype(self):
        return self._clstype

    def clsname(self):
        return self._clsname

    def load(self, x):
        return self.load_many(x) if iterutils.is_sequence(x) \
            else self.load_one(x)

    @abc.abstractmethod
    def load_one(self, x, *args, **kwargs):
        pass

    def load_many(self, x, *args, **kwargs):
        nitems = len(x)
        args = list(args)
        for i, value in enumerate(args):
            if value is None:
                args[i] = nitems * [None]
        for key, value in kwargs.items():
            if value is None:
                kwargs[key] = nitems * [None]
        if any([nitems != len(arg) for arg in args + list(kwargs.values())]):
            raise RuntimeError(
                "all arguments must have the same length or be None")
        nargs = len(args)
        args_list_shape = (nitems, nargs)
        args_list = iterutils.make_list(args_list_shape, [], True)
        for i in range(len(x)):
            for j, arg in enumerate(args):
                args_list[i][j] = arg[i]
        kwargs_list_shape = (nitems,)
        kwargs_list = iterutils.make_list(kwargs_list_shape, {}, True)
        for i in range(len(x)):
            for key, value in kwargs.items():
                kwargs_list[i][key] = value[i]
        results = []
        for item, item_args, item_kwargs in zip(x, args_list, kwargs_list):
            results.append(self.load_one(item, *item_args, **item_kwargs))
        return results

    def dump(self, x):
        return self.dump_many(x) if iterutils.is_sequence(x) \
            else self.dump_one(x)

    def dump_many(self, x):
        return [self.dump_one(item) for item in iterutils.listify(x)]

    def dump_one(self, x):
        return x.dump() if x is not None else None


class SimpleParser(Parser):

    def __init__(self, cls):
        super().__init__(cls)

    def load(self, x):
        return self.load_many(x) if iterutils.is_sequence(x) \
            else self.load_one(x)

    def load_one(self, x, *args, **kwargs):
        return self._clstype.load(x, *args, **kwargs)


class TypedParser(Parser):

    def __init__(self, cls, noneval=None):
        super().__init__(cls)
        self._parsers = {}
        self._noneval = noneval

    def dict(self):
        return dict(self._parsers)

    def register(self, factory):
        type_ = factory.type()
        if type_ in self._parsers:
            raise RuntimeError(
                f"{self._clsname} parser is already registered: '{type_}'.")
        self._parsers[type_] = factory

    def load(self, x):
        return self.load_many(x) if iterutils.is_sequence(x) \
            else self.load_one(x)

    def load_one(self, x, *args, **kwargs):
        if x is None:
            return self._noneval
        if 'type' not in x:
            raise RuntimeError(
                f"All {self._clsname} descriptions must contain a 'type'.")
        type_ = x.pop('type')
        if type_ not in self._parsers:
            raise RuntimeError(
                f"Could not find a {self._clsname} parser for type '{type_}'. "
                f"Available parsers are: {list(self._parsers.keys())}.")

        obj = self._parsers[type_].load(x, *args, **kwargs)

        return obj
