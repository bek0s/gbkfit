
import abc
import copy
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
    add_required = set(add_required if add_required else [])
    add_optional = set(add_optional if add_optional else [])
    add_all = add_required | add_optional
    fun_all = set()
    # Required added options must not clash with optional added options
    if add_required.intersection(add_optional):
        raise RuntimeError()
    # Update total options with added options
    required.update(add_required)
    optional.update(add_optional)
    # Infer options from callable
    if fun:
        fun_required, fun_optional = funcutils.extract_args(fun)
        fun_all = fun_required | fun_optional
        # Callable options must not clash with added options
        if fun_all.intersection(add_all):
            raise RuntimeError()
        # Ignore callable options if requested
        if fun_ignore_args:
            fun_required.difference_update(fun_ignore_args)
            fun_optional.difference_update(fun_ignore_args)
        # Rename callable options if requested
        if fun_rename_args:
            for arg_name, opt_name in fun_rename_args.items():
                if opt_name in info:
                    info[arg_name] = info.pop(opt_name)
        # Update total options with callable options
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
    add_options = {k: v for k, v in info.items() if k in add_all}
    fun_options = {k: v for k, v in info.items() if k in fun_all}
    all_options = {**add_options, **fun_options}
    return all_options, add_options, fun_options


class Serializable(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def load(cls, *args, **kwargs):
        pass

    @abc.abstractmethod
    def dump(self, *args, **kwargs):
        pass


class ParserSupport(Serializable, abc.ABC):
    pass


class SimpleParserSupport(ParserSupport, abc.ABC):
    pass


class TypedParserSupport(ParserSupport, abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def type():
        pass


def _prepare_args_and_kwargs(length, args, kwargs):
    args = list(args)
    for i, value in enumerate(args):
        if value is None:
            args[i] = length * [None]
    for key, value in kwargs.items():
        if value is None:
            kwargs[key] = length * [None]
    if any([length != len(arg) for arg in args + list(kwargs.values())]):
        raise RuntimeError(
            "all arguments must have the same length or be None")
    nargs = len(args)
    args_list_shape = (length, nargs)
    args_list = iterutils.make_list(args_list_shape, [], True)
    for i in range(length):
        for j, arg in enumerate(args):
            args_list[i][j] = arg[i]
    kwargs_list_shape = (length,)
    kwargs_list = iterutils.make_list(kwargs_list_shape, {}, True)
    for i in range(length):
        for key, value in kwargs.items():
            kwargs_list[i][key] = value[i]
    return args_list, kwargs_list


class Parser(abc.ABC):

    def __init__(self, cls):
        self._cls = cls

    def cls(self):
        return self._cls

    def load(self, x):
        return self.load_many(x) if iterutils.is_sequence(x) \
            else self.load_one(x)

    def load_one(self, x, *args, **kwargs):
        x = copy.deepcopy(x)
        return self._load_one_impl(x, *args, **kwargs) if x else None

    @abc.abstractmethod
    def _load_one_impl(self, x, *args, **kwargs):
        pass

    def load_many(self, x, *args, **kwargs):
        args_list, kwargs_list = _prepare_args_and_kwargs(len(x), args, kwargs)
        results = []
        for item, item_args, item_kwargs in zip(x, args_list, kwargs_list):
            results.append(self.load_one(item, *item_args, **item_kwargs))
        return results

    def dump(self, x):
        return self.dump_many(x) if iterutils.is_sequence(x) \
            else self.dump_one(x)

    def dump_one(self, x, *args, **kwargs):
        return self._dump_one_impl(x, *args, **kwargs) if x else None

    @abc.abstractmethod
    def _dump_one_impl(self, x, *args, **kwargs):
        pass

    def dump_many(self, x, *args, **kwargs):
        args_list, kwargs_list = _prepare_args_and_kwargs(len(x), args, kwargs)
        results = []
        for item, item_args, item_kwargs in zip(x, args_list, kwargs_list):
            results.append(self.dump_one(item, *item_args, **item_kwargs))
        return results


class SimpleParser(Parser):

    def __init__(self, cls):
        super().__init__(cls)

    def _load_one_impl(self, x, *args, **kwargs):
        return self.cls().load(x, *args, **kwargs)

    def _dump_one_impl(self, x, *args, **kwargs):
        return x.dump(*args, **kwargs)


class TypedParser(Parser):

    def __init__(self, cls):
        super().__init__(cls)
        self._parsers = {}

    def register(self, parser):
        desc = self.cls().__name__
        type_ = parser.type()
        if type_ in self._parsers:
            raise RuntimeError(
                f"{desc} parser already registered: {type_}")
        self._parsers[type_] = parser

    def _load_one_impl(self, x, *args, **kwargs):
        #print(x)
        desc = self.cls().__name__
        if 'type' not in x:
            raise RuntimeError(
                f"{desc} description must define a type")
        type_ = x.pop('type')
        if type_ not in self._parsers:
            raise RuntimeError(
                f"could not find a {desc} parser for type '{type_}'; "
                f"the available parsers are: {list(self._parsers.keys())}")
        return self._parsers[type_].load(x, *args, **kwargs)

    def _dump_one_impl(self, x, *args, **kwargs):
        info = x.dump(*args, **kwargs)
        info['type'] = x.type()
        return info
