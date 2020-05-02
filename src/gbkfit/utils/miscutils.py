
def merge_lists_and_make_mappings(list_list, prefix):
    list_merged = list()
    list_mappings = list()
    for i, item in enumerate(list_list):
        list_mappings.append(dict())
        for old_name in item:
            new_name = f'{prefix}{i}_{old_name}' if i else old_name
            list_mappings[i][old_name] = new_name
            list_merged.append(new_name)
    return list_merged, list_mappings


def merge_dicts_and_make_mappings(dict_list, prefix):
    dict_merged = dict()
    dict_mappings = list()
    for i, item in enumerate(dict_list):
        dict_mappings.append(dict())
        for old_name, value in item.items():
            new_name = f'{prefix}{i}_{old_name}' if i else old_name
            dict_mappings[i][old_name] = new_name
            dict_merged[new_name] = value
    return dict_merged, dict_mappings
