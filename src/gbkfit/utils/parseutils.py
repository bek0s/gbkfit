
import abc
import copy
import importlib
import inspect
import logging
import typing

from . import funcutils, iterutils, typeutils


_log = logging.getLogger(__name__)


def make_basic_desc(cls, label):
    return f'{label} (class={cls.__qualname__})'


def make_typed_desc(cls, label=None):
    desc = f'{cls.type()} (class={cls.__qualname__})'
    return f'{label} {desc}' if label else desc


def validate_option_value_type(options, types):
    for option_name, option_value in options.items():
        if option_name in types:
            option_type_info = types[option_name]
            option_type = option_type_info['type']
            option_size = option_type_info.get('size')
            option_trim = option_type_info.get('trim')
            option_allow_scalar = option_type_info.get('allow_scalar')
            option_is_scalar = option_size is None
            option_is_vector = option_size is not None
            option_type = iterutils.tuplify(option_type)
            option_size = iterutils.tuplify(option_size, False)
            option_type_name = [f"'{t.__name__}'" for t in option_type]
            option_type_name_joined = " or ".join(option_type_name)
            option_size_max = max(option_size) if option_size else None
            option_type_size_joined = " or ".join(map(str, option_size))
            # Option is scalar, value is scalar, type is correct
            # All good. Proceed to the next option
            if option_is_scalar and \
                    isinstance(option_value, option_type):
                continue
            # Option is vector, value is scalar, type is correct
            # and scalar values are allowed.
            # Multiply the scalar and proceed to the next option
            if option_is_vector and option_allow_scalar and \
                    isinstance(option_value, option_type):
                options[option_name] = iterutils.make_list(
                    option_size_max, option_value)
                continue
            # Option is vector, value is vector, type is correct
            # We need to check the size
            if option_is_vector and \
                    iterutils.is_sequence_of_type(option_value, option_type):
                # Any size is allowed.
                # Nothing to do. Proceed to the next option
                if option_size_max == 0:
                    continue
                # Value size is larger and trimming is allowed
                # Trim the value to maximum allowed size and
                # proceed to the next option
                if len(option_value) > option_size_max and option_trim:
                    option_value_new = option_value[:option_size_max]
                    _log.warning(
                        f"option '{option_name}' has a sequence value "
                        f"with a length longer than expected; "
                        f"expected length: {option_type_size_joined}, "
                        f"current length: {len(option_value)}; "
                        f"the value will be trimmed to: {option_value_new}")
                    options[option_name] = option_value_new
                    continue
                # Value size is allowed
                # Nothing to do. Proceed to the next option
                if len(option_value) in option_size:
                    continue
            # All the above checked failed.
            # This means that the option value is invalid.
            msg = f"option '{option_name}' should be "
            if option_is_scalar or option_is_vector and option_allow_scalar:
                msg += f"of type {option_type_name_joined}"
            if option_is_vector:
                if option_allow_scalar:
                    msg += ", or "
                msg += f"a sequence of type {option_type_name_joined}"
                if option_size_max > 0:
                    msg += f" and length {option_type_size_joined}"
            msg += f"; instead, the following invalid value was provided: " \
                   f"{option_value}"
            raise RuntimeError(msg)


def parse_options(info, desc, required=None, optional=None):
    info = copy.deepcopy(info)
    required = iterutils.setify(required, False)
    optional = iterutils.setify(optional, False)
    # Required options must not clash with optional options
    if conflicting := required & optional:
        raise RuntimeError(
            f"the following required and optional options are conflicting: "
            f"{conflicting}")
    # Check for unknown options
    if unknown := set(info) - (required | optional):
        _log.warning(
            f"the following {desc} options are "
            f"not recognised and will be ignored: {str(unknown)}")
    # Check for missing options
    if missing := required - set(info):
        raise RuntimeError(
            f"the following options for {desc} are "
            f"required but missing: {str(missing)}")
    # Return all recognised options
    return {k: v for k, v in info.items() if k in required | optional}


def parse_options_for_callable(
        info, desc,
        fun, fun_ignore_args=None, fun_rename_args=None,
        add_required=None, add_optional=None):
    fun_ignore_args = iterutils.setify(fun_ignore_args, False)
    fun_rename_args = fun_rename_args if fun_rename_args else {}
    add_required = add_required if add_required else {}
    add_optional = add_optional if add_optional else {}
    add_required_keys = set(add_required.keys())
    add_optional_keys = set(add_optional.keys())
    # Extract required and optional options/arguments from callable
    fun_required, fun_optional = funcutils.extract_args(fun)[1:]
    fun_required = set(fun_required)  # todo: clean up
    fun_optional = set(fun_optional)
    fun_all = fun_required | fun_optional
    # Ignore callable options if requested
    if fun_ignore_args:
        # Ensure the ignored options are known
        if unknown := fun_ignore_args - fun_all:
            raise RuntimeError(
                f"the following ignored options do not exist "
                f"in the callable's argument list: {unknown}")
        fun_required.difference_update(fun_ignore_args)
        fun_optional.difference_update(fun_ignore_args)
    # Rename callable options if requested
    if fun_rename_args:
        # Ensure the old names of the rename options are known
        if unknown := set(fun_rename_args.keys()) - fun_all:
            raise RuntimeError(
                f"the following rename options do not exist "
                f"in the callable's argument list: {unknown}")
        # Ensure the new names of the rename options are not
        # in conflict with the arguments of the callable
        if conflicting := set(fun_rename_args.values()) & fun_all:
            raise RuntimeError(
                f"the following rename options are in conflict "
                f"with callable's argument list: {conflicting}")
        for old_name, new_name in fun_rename_args.items():
            if old_name in fun_required:
                fun_required.discard(old_name)
                fun_required.add(new_name)
            if old_name in fun_optional:
                fun_optional.discard(old_name)
                fun_optional.add(new_name)
    # These will hold the total required and optional options
    required = set(fun_required)
    optional = set(fun_optional)
    # Added required options must not clash with added optional options
    if intersection := add_required_keys & add_optional_keys:
        raise RuntimeError(
            f"the following options should not exist "
            f"in both add_required and add_optional: "
            f"{intersection}")
    add_all_keys = add_required_keys | add_optional_keys
    # Added options should not conflict with callable's options
    if conflicting := add_all_keys & fun_all:
        raise RuntimeError(
            f"the following added options are in conflict "
            f"with callable's argument list: {conflicting}")
    # Renamed options must not clash with added options
    rename_old_names = set(fun_rename_args.keys())
    rename_new_names = set(fun_rename_args.values())
    if intersection := add_all_keys & (rename_old_names | rename_new_names):
        raise RuntimeError(
            f"the following options should not exist "
            f"in both fun_rename_args and add_[required|optional]: "
            f"{intersection}")
    # Update total options with added options
    required.update(add_required_keys)
    optional.update(add_optional_keys)
    # Parse required and optional options
    options = parse_options(info, desc, required, optional)
    # Validate option types
    option_types = inspect.get_annotations(fun)
    option_types = option_types | add_required | add_optional
    option_types = iterutils.remove_from_mapping_by_value(option_types, None)
    for option_name, option_value in options.items():
        if option_name in option_types:
            option_type = option_types[option_name]
            option_type_count = not bool(typing.get_args(option_type))
            option_type_label = \
                option_type.__name__ if option_type_count == 1 else option_type
            if not typeutils.validate_type(option_value, option_type):
                raise RuntimeError(
                    f"option '{option_name}' is set to '{option_value}', "
                    f"however, the expected type for this option is: "
                    f"{option_type_label}")
    # Rename options back to their argument name if needed
    if fun_rename_args:
        for arg_name, opt_name in fun_rename_args.items():
            if opt_name in options:
                options[arg_name] = options.pop(opt_name)
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

    def cls_name(self):
        return self.cls().__qualname__

    def load(self, x, *args, **kwargs):
        print(locals())
        return self.load_many(x, *args, **kwargs) if iterutils.is_sequence(x) \
            else self.load_one(x, *args, **kwargs)

    def load_one(self, x, *args, **kwargs):
        x = copy.deepcopy(x)
        if not isinstance(x, (dict, type(None))):
            raise RuntimeError(
                f"{self.cls_name()} parser "
                f"expected configuration in the form of a dictionary; "
                f"instead it found the following value: {x}")
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
            parser_desc = make_typed_desc(parser)
            if not issubclass(parser, self.cls()):
                raise RuntimeError(
                    f"{self.cls_name()} parser "
                    f"could not register parser of type {parser_desc}; "
                    f"the two parsers are incompatible")
            if parser.type() in self._parsers:
                raise RuntimeError(
                    f"{self.cls_name()} parser "
                    f"could not register parser of type {parser_desc}; "
                    f"a parser of the same type is already registered")
            self._parsers[parser.type()] = parser

    def _load_one_impl(self, x, *args, **kwargs):
        if 'type' not in x:
            raise RuntimeError(
                f"{self.cls_name()} parser "
                f"configurations must define a 'type'")
        type_ = x.pop('type')
        if type_ not in self._parsers:
            raise RuntimeError(
                f"{self.cls_name()} parser "
                f"could not find a parser for type '{type_}'; "
                f"the available parsers are: {list(self._parsers.keys())}")
        # The parser exists. Get a reference to it for convenience
        parser = self._parsers[type_]
        try:
            instance = parser.load(x, *args, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"{self.cls_name()} parser "
                f"could not parse configuration for item of type "
                f"{make_typed_desc(parser)}, (reason)=> {e}") from e
        return instance

    def _dump_one_impl(self, x, *args, **kwargs):
        info = x.dump(*args, **kwargs)
        info['type'] = x.type()
        return info


def register_optional_parsers(abstract_parser, parsers, desc_label):
    for parser in parsers:
        try:
            mod_name = parser.rsplit('.', 1)[0]
            cls_name = parser.rsplit('.', 1)[1]
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            abstract_parser.register(cls)
        except Exception as e:
            _log.warning(
                f"could not register {desc_label} parser {parser}; "
                f"{e.__class__.__name__}: {e}")
