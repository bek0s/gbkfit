
def merge_lists_and_make_mappings(list_list, prefix, force_prefix=False):
    list_merged = list()
    list_mappings = list()
    for i, item in enumerate(list_list):
        list_mappings.append(dict())
        for old_name in item:
            use_prefix = force_prefix or bool(i)
            new_name = f'{prefix}{i}_{old_name}' if use_prefix else old_name
            list_mappings[i][old_name] = new_name
            list_merged.append(new_name)
    return list_merged, list_mappings


def merge_dicts_and_make_mappings(dict_list, prefix, zero_prefix=False, zero_index=False):
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
