
import importlib.util
import os
import pathlib


def get_attr_from_file(file, attr):
    module_name = pathlib.Path(file).stem
    module_spec = importlib.util.spec_from_file_location(module_name, file)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, attr)


def merge_lists_and_make_mappings(
        list_list, prefix, zero_prefix=False, zero_index=False):
    list_merged = list()
    list_mappings = list()
    for i, item in enumerate(list_list):
        list_mappings.append(dict())
        for old_name in item:
            full_prefix = ''
            if bool(i) or zero_prefix:
                full_prefix += prefix
            if bool(i) or zero_index:
                full_prefix += str(i)
            if full_prefix:
                full_prefix += '_'
            new_name = f'{full_prefix}{old_name}'
            list_mappings[i][old_name] = new_name
            list_merged.append(new_name)
    return list_merged, list_mappings


def merge_dicts_and_make_mappings(
        dict_list, prefix, zero_prefix=False, zero_index=False):
    dict_merged = dict()
    dict_mappings = list()
    for i, item in enumerate(dict_list):
        dict_mappings.append(dict())
        for old_name, value in item.items():
            full_prefix = ''
            if bool(i) or zero_prefix:
                full_prefix += prefix
            if bool(i) or zero_index:
                full_prefix += str(i)
            if full_prefix:
                full_prefix += '_'
            new_name = f'{full_prefix}{old_name}'
            dict_mappings[i][old_name] = new_name
            dict_merged[new_name] = value
    return dict_merged, dict_mappings


def to_native_byteorder(arr):
    return arr if arr.dtype.isnative else arr.byteswap().newbyteorder()


def make_unique_path(path):
    i = 0
    base = path
    while os.path.exists(path):
        i += 1
        path = f'{base}_{i}'
    return path
