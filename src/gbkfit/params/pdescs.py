
import abc
import copy
from collections.abc import Mapping

from typing import Any

import numpy as np

from gbkfit.utils import iterutils, parseutils


__all__ = [
    'ParamDesc',
    'ParamDescDict',
    'ParamScalarDesc',
    'ParamVectorDesc',
    'load_pdescs_dict',
    'dump_pdescs_dict',
    'pdesc_parser'
]


class ParamDesc(parseutils.TypedSerializable, abc.ABC):

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'ParamDesc':
        desc = parseutils.make_typed_desc(cls, 'pdesc')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self) -> dict[str, Any]:
        info: dict[str, str | int | float] = dict(
            type=self.type(), name=self.name(), size=self.size())
        if self.desc() is not None:
            info.update(desc=self.desc())
        if self.default() is not None:
            info.update(default=self.default())
        if np.isfinite(self.minimum()):
            info.update(minimum=self.minimum())
        if np.isfinite(self.maximum()):
            info.update(maximum=self.maximum())
        return info

    def __init__(
            self,
            name: str,
            size: int,
            desc: None | str,
            default: None | int | float,
            minimum: None | int | float,
            maximum: None | int | float
    ):
        # Import here to avoid circular dependency
        from .symbols import is_param_symbol_name
        if not is_param_symbol_name(name):
            raise RuntimeError(
                f"'{name}' is not a valid parameter description name")
        minimum = -np.inf if minimum is None else minimum
        maximum = +np.inf if maximum is None else maximum
        if minimum > maximum:
            raise RuntimeError(
                f"minimum ({minimum}) must be <= maximum ({maximum})")
        if default is not None and (default < minimum or default > maximum):
            raise RuntimeError(
                f"default ({default}) must be within "
                f"[minimum ({minimum}), maximum ({maximum})]"
            )
        self._name = name
        self._size = size
        self._desc = desc
        self._default = float(default) if default is not None else None
        self._minimum = float(minimum)
        self._maximum = float(maximum)

    def name(self) -> str:
        return self._name

    def size(self) -> int:
        return self._size

    def desc(self) -> None | str:
        return self._desc

    def default(self) -> None | float:
        return self._default

    def minimum(self) -> float:
        return self._minimum

    def maximum(self) -> float:
        return self._maximum


class ParamScalarDesc(ParamDesc):

    @staticmethod
    def type() -> str:
        return 'scalar'

    def dump(self) -> dict[str, Any]:
        info = super().dump()
        del info['size']
        return info

    def __init__(
            self,
            name: str,
            desc: None | str = None,
            default: None | int | float = None,
            minimum: None | int | float = None,
            maximum: None | int | float = None
    ):
        super().__init__(name, 1, desc, default, minimum, maximum)


class ParamVectorDesc(ParamDesc):

    @staticmethod
    def type() -> str:
        return 'vector'

    def __init__(
            self,
            name: str,
            size: int,
            desc: None | str = None,
            default: None | int | float = None,
            minimum: None | int | float = None,
            maximum: None | int | float = None
    ):
        super().__init__(name, size, desc, default, minimum, maximum)


class ParamDescDict(Mapping):
    """Not used at the moment"""

    def __init__(self, pdescs):
        self._pdescs = copy.deepcopy(pdescs)

    def __getitem__(self, key):
        return self._pdescs.__getitem__(key)

    def __iter__(self):
        return self._pdescs.__iter__()

    def __len__(self):
        return self._pdescs.__len__()

    def __repr__(self):
        return self._pdescs.__repr__()

    def __str__(self):
        return self._pdescs.__str__()


def load_pdescs_dict(info: dict[str, Any]) -> dict[str, ParamDesc]:
    if bad := [k for k, v in info.items() if not iterutils.is_mapping(v)]:
        raise RuntimeError(
            f"the value of the following keys must be a dictionary: {bad}")
    if bad := [k for k, v in info.items() if k != v.get('name', k)]:
        raise RuntimeError(
            f"the following keys are not equal to "
            f"their corresponding pdesc name: {bad}")
    pdescs = {}
    for key, val in info.items():
        pdescs[key] = pdesc_parser.load(dict(name=key) | val)
    return pdescs


def dump_pdescs_dict(pdescs: dict[str, ParamDesc]) -> dict[str, Any]:
    if bad := [k for k, v in pdescs.items() if k != v.name()]:
        raise RuntimeError(
            f"the following keys are not equal to "
            f"their corresponding pdesc name: {bad}")
    info = {key: pdesc_parser.dump(val) for key, val in pdescs.items()}
    for val in info.values():
        del val['name']
    return info


class ParamDescTypedParser(parseutils.TypedParser):

    def __init__(self):
        super().__init__(ParamDesc, [ParamScalarDesc, ParamVectorDesc])

    def load_many(
            self,
            x: list[dict[str, Any]],
            allow_duplicates: bool = False,
            *args, **kwargs
    ) -> list[ParamDesc]:
        pdescs = super().load_many(x, *args, **kwargs)
        if not allow_duplicates:
            self._ensure_no_duplicates(pdescs)
        return pdescs

    def dump_many(
            self,
            x: list[ParamDesc],
            allow_duplicates: bool = False,
            *args, **kwargs
    ) -> list[dict[str, Any]]:
        if not allow_duplicates:
            self._ensure_no_duplicates(x)
        return super().dump_many(x, *args, **kwargs)

    @staticmethod
    def _ensure_no_duplicates(x: list[ParamDesc]):
        seen = []
        duplicates = []
        for pdesc in x:
            name = pdesc.name()
            # Already marked as a duplicate. Do nothing.
            if name in duplicates:
                continue
            # Already marked as seen. Mark it as duplicate
            if name in seen:
                duplicates.append(name)
                continue
            # Not seen before. Mark it as seen
            seen.append(name)
        if duplicates:
            raise RuntimeError(
                f"the following pdescs names appear multiple times: "
                f"{duplicates}")


pdesc_parser = ParamDescTypedParser()
