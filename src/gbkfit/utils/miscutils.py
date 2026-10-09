import pathlib

import numpy as np


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
