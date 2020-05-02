
def merge_pdescs(pdescs, prefix='model'):
    descs = {}
    mappings = []
    for i, pdesc in enumerate(pdescs):
        mappings.append({})
        for old_name, desc in pdesc.items():
            new_name = f'{prefix}{i}_{old_name}' if i else old_name
            mappings[i][old_name] = new_name
            descs[new_name] = desc
    return descs, mappings

