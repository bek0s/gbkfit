"""
The loaders and dumpers of configurations: of one object, or of a list
of objects, of a class (BasicParser) or of one of several, chosen by
their type (TypedParser).
"""

import abc
import copy
from collections.abc import Sequence
from typing import Any

from gbkfit.utils import iterutils
from .errors import ConfigError, config_path, describe_unknown, quoted_names
from .families import register_family
from .serializable import Serializable, TypedSerializable, missing_modules


__all__ = [
    'BasicParser',
    'Parser',
    'TypedParser'
]


def _prepare_args_and_kwargs(
        length: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any]
) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    """
    Split the arguments for a list of items into those of each item.

    Each argument is a sequence with a value for each item, or None for
    None for each item.
    """
    def validate_sequence(
            value: Sequence[Any] | None,
            name: str
    ) -> Sequence[Any]:
        if value is None:
            return [None] * length
        if not iterutils.is_sequence(value) or len(value) != length:
            raise RuntimeError(
                f"expected '{name}' to be a sequence of length {length}, "
                f"but got: {value}")
        return value
    args_list = [[] for _ in range(length)]
    for i, arg in enumerate(args):
        for j, value_ in enumerate(validate_sequence(arg, f"args[{i}]")):
            args_list[j].append(value_)
    kwargs_list = [{} for _ in range(length)]
    for key, val in kwargs.items():
        for j, value_ in enumerate(validate_sequence(val, f"kwargs['{key}']")):
            kwargs_list[j][key] = value_

    return args_list, kwargs_list


class Parser(abc.ABC):
    """
    The loader and dumper of the configurations of the objects of a class.

    A configuration is a dict, which describes one object, or a list of
    dicts, which describes a list of objects. A null configuration
    describes no object (None).

    Parameters
    ----------
    cls : type
        The class.
    """

    def __init__(self, cls: type[Serializable]):
        self._cls = cls

    def cls(self) -> type[Serializable]:
        return self._cls

    def cls_name(self) -> str:
        return self.cls().__qualname__

    def load(
            self,
            x: dict[str, Any] | Sequence[dict[str, Any] | None] | None,
            *args: Any,
            **kwargs: Any
    ) -> Any:
        """
        Load an object, or a list of objects, from its configuration.

        The configuration is not changed.

        Parameters
        ----------
        x : dict or list of dict or None
            The configuration.
        *args, **kwargs
            Passed to the load of the class; for a list, sequences with a
            value for each item.

        Returns
        -------
        Any
            The object, or the list of objects.

        Raises
        ------
        ConfigError
            If the configuration has an error.
        """
        if iterutils.is_sequence(x):
            result = self.load_many(x, *args, **kwargs)
        else:
            result = self.load_one(x, *args, **kwargs)
        return result

    def load_one(
            self,
            x: dict[str, Any] | None,
            *args: Any,
            **kwargs: Any
    ) -> Any:
        return self._load_one_impl_wrapper(x, None, *args, **kwargs)

    def _load_one_impl_wrapper(
            self,
            x: object,
            index: int | None,
            *args: Any,
            **kwargs: Any
    ) -> Any:
        # Create a copy of the configuration for safety
        x = copy.deepcopy(x)
        with config_path(*([] if index is None else [index])):
            if not isinstance(x, (dict, type(None))):
                raise ConfigError(
                    f"expected configuration in the form of a dictionary; "
                    f"instead it found the following value: {x}")
            return self._load_one_impl(x, *args, **kwargs) \
                if x is not None else None

    @abc.abstractmethod
    def _load_one_impl(
            self,
            x: dict[str, Any],
            *args: Any,
            **kwargs: Any
    ) -> Serializable:
        pass

    def load_many(
            self,
            x: Sequence[dict[str, Any] | None],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        args_list, kwargs_list = _prepare_args_and_kwargs(
            len(x), args, kwargs)
        results = []
        for i, (item, item_args, item_kwargs) in enumerate(
                zip(x, args_list, kwargs_list)):
            results.append(self._load_one_impl_wrapper(
                item, i, *item_args, **item_kwargs))
        return results

    def dump(
            self,
            x: Serializable | Sequence[Serializable] | None,
            *args: Any,
            **kwargs: Any
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """
        Dump an object, or a list of objects, to its configuration.

        Parameters
        ----------
        x : Serializable or list of Serializable or None
            The object, or the list of objects.
        *args, **kwargs
            Passed to the dump of the class; for a list, sequences with a
            value for each item.

        Returns
        -------
        dict or list of dict or None
            The configuration.
        """
        if iterutils.is_sequence(x):
            result = self.dump_many(x, *args, **kwargs)
        else:
            result = self.dump_one(x, *args, **kwargs)
        return result

    def dump_one(
            self,
            x: Serializable | None,
            *args: Any,
            **kwargs: Any
    ) -> dict[str, Any] | None:
        return None if x is None else self._dump_one_impl(x, *args, **kwargs)

    @abc.abstractmethod
    def _dump_one_impl(
            self,
            x: Serializable,
            *args: Any,
            **kwargs: Any
    ) -> dict[str, Any]:
        pass

    def dump_many(
            self,
            x: Sequence[Serializable],
            *args: Any,
            **kwargs: Any
    ) -> list[dict[str, Any]]:
        args_list, kwargs_list = _prepare_args_and_kwargs(len(x), args, kwargs)
        results = []
        for item, item_args, item_kwargs in zip(x, args_list, kwargs_list):
            results.append(self.dump_one(item, *item_args, **item_kwargs))
        return results


class BasicParser(Parser):
    """
    The parser of a class with one form of configuration.
    """

    def _load_one_impl(
            self,
            x: dict[str, Any],
            *args: Any,
            **kwargs: Any
    ) -> Serializable:
        return self.cls().load(x, *args, **kwargs)

    def _dump_one_impl(
            self,
            x: Serializable,
            *args: Any,
            **kwargs: Any
    ) -> dict[str, Any]:
        return x.dump(*args, **kwargs)


class TypedParser(Parser):
    """
    The parser of a class whose configurations choose one of its
    subclasses by their type (the option 'type').

    Subclasses that need modules GBKFIT does not always have (see
    Serializable.requires) are registered as the others, and their
    options can be read without them; asking for one whose modules are
    missing says what to install.

    Parameters
    ----------
    cls : type
        The class.
    classes : type or list of type, optional
        The subclasses to register (see register).
    family : str, optional
        The name of the family of the class, which the parser registers
        (see gbkfit.config.families; e.g. 'psf'): every parser of the
        library names one.
    """

    def __init__(
            self,
            cls: type[TypedSerializable],
            classes: (type[TypedSerializable] | list[type[TypedSerializable]]
                      | None) = None,
            *,
            family: str | None = None
    ):
        super().__init__(cls)
        self._classes: dict[str, type[TypedSerializable]] = {}
        self.register(classes)
        if family is not None:
            register_family(family, self)

    def register(
            self,
            classes: (type[TypedSerializable] | list[type[TypedSerializable]]
                      | None)
    ) -> None:
        """
        Register subclasses, each for its type.

        Parameters
        ----------
        classes : type or list of type or None
            The subclasses.

        Raises
        ------
        RuntimeError
            If a class is not a subclass, or its type is registered.
        """
        for cls in iterutils.listify(classes):
            desc = f"{cls.__qualname__} (type '{cls.type()}')"
            if not issubclass(cls, self.cls()):
                raise RuntimeError(
                    f"{self.cls_name()} parser could not register {desc}; "
                    f"it is not a subclass of {self.cls_name()}")
            if cls.type() in self._classes:
                raise RuntimeError(
                    f"{self.cls_name()} parser could not register {desc}; "
                    f"a class of the same type is already registered")
            self._classes[cls.type()] = cls

    def types(self) -> list[str]:
        """
        Return all the registered types, those of classes whose modules
        are missing included (see Serializable.requires).
        """
        return sorted(self._classes)

    def registered_class(self, type_: str) -> type[TypedSerializable]:
        """
        Return the class registered for a type.

        Parameters
        ----------
        type_ : str
            The type.

        Returns
        -------
        type
            The class.

        Raises
        ------
        ConfigError
            If no class is registered for the type, or the modules it
            needs are missing (see Serializable.requires).
        """
        if not isinstance(type_, str) or type_ not in self._classes:
            raise ConfigError(
                f"unknown type {describe_unknown([type_], self.types())}; "
                f"the types are: {quoted_names(self.types())}")
        cls = self._classes[type_]
        if missing := missing_modules(cls):
            how = '; '.join(dict.fromkeys(missing.values()))
            raise ConfigError(f"type '{type_}' needs {quoted_names(missing)}: {how}")
        return cls

    def registered_classes(self) -> dict[str, type[TypedSerializable]]:
        """
        Return the registered classes, those whose modules are missing
        included (see Serializable.requires).

        Returns
        -------
        dict
            The class registered for each type.
        """
        return dict(self._classes)

    def _load_one_impl(
            self,
            x: dict[str, Any],
            *args: Any,
            **kwargs: Any
    ) -> TypedSerializable:
        if 'type' not in x:
            raise ConfigError(
                f"missing required options: 'type'; the types are: "
                f"{quoted_names(self.types())}")
        type_ = x.pop('type')
        cls = self.registered_class(type_)
        with config_path(context=type_):
            return cls.load(x, *args, **kwargs)

    def _dump_one_impl(
            self,
            x: Serializable,
            *args: Any,
            **kwargs: Any
    ) -> dict[str, Any]:
        """Dump the object, with its type."""
        if not isinstance(x, TypedSerializable):
            raise RuntimeError(
                f"unsupported object; value: {x}, type: {type(x)}")
        return dict(type=x.type()) | x.dump(*args, **kwargs)
