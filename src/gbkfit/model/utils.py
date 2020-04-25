
def make_param_group(gmodels, prefix='model'):
    descs = {}
    mappings = {}
    for i, gmodel in enumerate(gmodels):
        for old_name, desc in gmodel.params().items():
            new_name = f'{prefix}{i}_{old_name}' if i else old_name
            mappings[i][old_name] = new_name
            descs[new_name] = desc
    return descs, mappings
