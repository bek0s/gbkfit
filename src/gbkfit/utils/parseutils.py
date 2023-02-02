
import abc
import copy
import importlib
import logging

from . import funcutils, iterutils


_log = logging.getLogger(__name__)


def make_basic_desc(cls, label):
    return f'{label} (class={cls.__qualname__})'


def make_typed_desc(cls, label):
    return f'{cls.type()} (class={cls.__qualname__}) {label}'


def parse_options(info, desc, required=None, optional=None):
    info = copy.deepcopy(info)
    required = set(required if required else [])
    optional = set(optional if optional else [])
    # Required options must not clash with optional options
    if conflicting := required.intersection(optional):
        raise RuntimeError(
            f"the following required and optional options are conflicting: "
            f"{conflicting}")
    # Check for missing or unknown options
    unknown = set(info) - (required | optional)
    missing = required - set(info)
    if unknown:
        _log.warning(
            f"the following {desc} options are "
            f"not recognised and will be ignored: {str(unknown)}")
    if missing:
        raise RuntimeError(
            f"the following {desc} options are "
            f"required but missing: {str(missing)}")
    # Return all recognised options
    return {k: v for k, v in info.items() if k in required | optional}


def parse_options_for_callable(
        info, desc,
        fun, fun_ignore_args=None, fun_rename_args=None,
        add_required=None, add_optional=None):
    required = set()
    optional = set()
    add_required = set(add_required if add_required else [])
    add_optional = set(add_optional if add_optional else [])
    add_all = add_required | add_optional
    # Required added options must not clash with optional added options
    if add_required.intersection(add_optional):
        raise RuntimeError()
    # Update total options with added options
    required.update(add_required)
    optional.update(add_optional)
    # Infer options from callable
    if fun:
        fun_required, fun_optional = funcutils.extract_args(fun)[1:]
        fun_required = set(fun_required)  # TODO clean up
        fun_optional = set(fun_optional)
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
                if arg_name in fun_required:
                    fun_required.discard(arg_name)
                    fun_required.add(opt_name)
                if arg_name in fun_optional:
                    fun_optional.discard(arg_name)
                    fun_optional.add(opt_name)
        # Update total options with callable options
        required.update(fun_required)
        optional.update(fun_optional)
    # Parse required and optional options
    options = parse_options(info, desc, required, optional)
    # Rename options back to their argument name if needed
    if fun_rename_args:
        for arg_name, opt_name in fun_rename_args.items():
            if opt_name in options:
                options[arg_name] = options.pop(opt_name)
    return options


def prepare_for_dump(options, remove_nones=True, remove_keys=()):
    options = copy.deepcopy(options)
    for key in list(options.keys()):
        if key in remove_keys or (remove_nones and options[key] is None):
            del options[key]
    return options


def prepare_for_dump2(options, predicate=lambda x: x is None, remove_keys=()):
    options = copy.deepcopy(options)
    for key in list(options.keys()):
        if key in remove_keys or predicate(options[key]):
            del options[key]
    return options


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


class BasicParserSupport(ParserSupport, abc.ABC):
    pass


class TypedParserSupport(ParserSupport, abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def type():
        pass


def _prepare_args_and_kwargs(length, args, kwargs):
    args = list(args)
    # Make sure all args and kwargs have the correct length or are None
    bad_args_found = False
    for i, value in enumerate(args):
        if value is None:
            args[i] = length * [None]
        elif not iterutils.is_sequence(value) or len(value) != length:
            bad_args_found = True
    bad_kwargs_found = False
    for key, value in kwargs.items():
        if value is None:
            kwargs[key] = length * [None]
        elif not iterutils.is_sequence(value) or len(value) != length:
            bad_kwargs_found = True
    if bad_args_found or bad_kwargs_found:
        raise RuntimeError(
            f"when loading or dumping a list of items, "
            f"args and kwargs must contain sequences with "
            f"a length equal to the list length; "
            f"list length: {length}; args: {tuple(args)}; kwargs: {kwargs}")
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

    def load(self, x, *args, **kwargs):
        return self.load_many(x, *args, **kwargs) if iterutils.is_sequence(x) \
            else self.load_one(x, *args, **kwargs)

    def load_one(self, x, *args, **kwargs):
        x = copy.deepcopy(x)
        return self._load_one_impl(x, *args, **kwargs) \
            if x is not None else None

    @abc.abstractmethod
    def _load_one_impl(self, x, *args, **kwargs):
        pass

    def load_many(self, x, *args, **kwargs):
        args_list, kwargs_list = _prepare_args_and_kwargs(len(x), args, kwargs)
        results = []
        for item, item_args, item_kwargs in zip(x, args_list, kwargs_list):
            results.append(self.load_one(item, *item_args, **item_kwargs))
        return results

    def dump(self, x, *args, **kwargs):
        return self.dump_many(x, *args, **kwargs) if iterutils.is_sequence(x) \
            else self.dump_one(x, *args, **kwargs)

    def dump_one(self, x, *args, **kwargs):
        x = copy.deepcopy(x)
        return self._dump_one_impl(x, *args, **kwargs) \
            if x is not None else None

    @abc.abstractmethod
    def _dump_one_impl(self, x, *args, **kwargs):
        pass

    def dump_many(self, x, *args, **kwargs):
        args_list, kwargs_list = _prepare_args_and_kwargs(len(x), args, kwargs)
        results = []
        for item, item_args, item_kwargs in zip(x, args_list, kwargs_list):
            results.append(self.dump_one(item, *item_args, **item_kwargs))
        return results


class BasicParser(Parser):

    def __init__(self, cls):
        super().__init__(cls)

    def _load_one_impl(self, x, *args, **kwargs):
        return self.cls().load(x, *args, **kwargs)

    def _dump_one_impl(self, x, *args, **kwargs):
        return x.dump(*args, **kwargs)


class TypedParser(Parser):

    def __init__(self, cls, parsers=None):
        super().__init__(cls)
        self._parsers = {}
        self.register(parsers)

    def register(self, parsers):
        parsers = iterutils.listify(parsers, False)
        for parser in parsers:
            parser_cls = self.cls().__name__
            factory_type = parser.type()
            factory_clss = parser.__name__

            if not issubclass(parser, self.cls()):
                raise RuntimeError(
                    f"{parser_cls} parser cannot register factory of type {factory_clss}")

            if factory_type in self._parsers:
                raise RuntimeError(
                    f"{parser_cls} parser already registered: {factory_type}")
            self._parsers[factory_type] = parser

    def _load_one_impl(self, x, *args, **kwargs):
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


def register_optional_parser_factories(parser, factories, desc_label):
    for factory in factories:
        try:
            mod_name = factory.rsplit('.', 1)[0]
            cls_name = factory.rsplit('.', 1)[1]
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            parser.register(cls)
        except Exception as e:
            _log.warning(
                f"could not register {desc_label} factory {factory}; "
                f"{e.__class__.__name__}: {e}")
