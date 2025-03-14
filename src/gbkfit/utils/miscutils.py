
import numpy as np

import importlib.abc
import importlib.util
import inspect
import pathlib
import sys
import textwrap

from typing import Any


def get_attr_from_file(file_path: str, attr: str, cache: bool = False) -> Any:
    """
    Loads a Python module from a file and retrieves an attribute.
    """
    module_name = pathlib.Path(file_path).stem
    if cache and module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        module_spec = importlib.util.spec_from_file_location(
            module_name, file_path)
        if not module_spec or not module_spec.loader:
            raise RuntimeError(
                f"could not load module from file '{file_path}'")
        module = importlib.util.module_from_spec(module_spec)
        try:
            module_spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"error executing module '{file_path}': {e}")
        if cache:
            sys.modules[module_name] = module
    if not hasattr(module, attr):
        raise RuntimeError(
            f"module '{module_name}' from '{file_path}' "
            f"does not have attribute '{attr}'")
    return getattr(module, attr)


def merge_lists_and_make_mappings(
        list_list: list[list[str]],
        prefix: str,
        zero_prefix: bool = False,
        zero_index: bool = False
) -> tuple[list, list[dict[str, str]]]:
    """
    Merges multiple lists while ensuring unique values by prefixing
    them with a specified identifier and index.
    I have no idea why I wrote this function. It has no good use! lol
    """
    list_merged = list()
    list_mappings = list()
    for i, item in enumerate(list_list):
        list_mappings.append(dict())
        for old_name in item:
            full_prefix = ''
            if i or zero_prefix:
                full_prefix += prefix
            if i or zero_index:
                full_prefix += str(i)
            if full_prefix:
                full_prefix += '_'
            new_name = f'{full_prefix}{old_name}'
            list_mappings[i][old_name] = new_name
            list_merged.append(new_name)
    return list_merged, list_mappings


def merge_dicts_and_make_mappings(
        dict_list: list[dict[str, Any]],
        prefix: str,
        zero_prefix: bool = False,
        zero_index: bool = False
) -> tuple[dict[str, Any], tuple[dict[str, str], ...]]:
    """
    Merges multiple dictionaries while ensuring unique keys by
    prefixing them with a specified identifier and index.
    """
    dict_merged = dict()
    dict_mappings = list()
    for i, item in enumerate(dict_list):
        dict_mappings.append(dict())
        for old_name, value in item.items():
            full_prefix = ''
            if i or zero_prefix:
                full_prefix += prefix
            if i or zero_index:
                full_prefix += str(i)
            if full_prefix:
                full_prefix += '_'
            new_name = f'{full_prefix}{old_name}'
            dict_mappings[i][old_name] = new_name
            dict_merged[new_name] = value
    return dict_merged, tuple(dict_mappings)


def to_native_byteorder(arr: np.ndarray) -> np.ndarray:
    """
    Ensure the given NumPy array has the native byte order.
    """
    return arr if arr.dtype.isnative else arr.byteswap().view(arr.dtype.newbyteorder('='))


def make_unique_path(path: pathlib.Path) -> pathlib.Path:
    """
    Generate a unique file path by appending an incrementing number.

    If the given path already exists, appends '_1', '_2', etc.,
    until a unique path is found.
    """
    path = pathlib.Path(path)
    if not path.exists():
        return path
    base, ext = path.stem, path.suffix
    i = 1
    while (new_path := path.with_name(f"{base}_{i}{ext}")).exists():
        i += 1
    return new_path


def get_source(
        obj: Any,
        dedent: bool = True,
        silent: bool = False
) -> str | None:
    """
    Retrieve the source code of a given object.
    """
    try:
        src = inspect.getsource(obj)
        src = textwrap.dedent(src) if dedent else src
    except Exception as e:
        src = None
        if not silent:
            raise RuntimeError(
                f"error retrieving source code from object: {e}")
    return src
