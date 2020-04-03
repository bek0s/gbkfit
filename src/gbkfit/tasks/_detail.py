
import numpy as np

import gbkfit.model.model
from gbkfit.utils import iterutils


def prepare_config_require_sections(config, sections):
    missing = []
    for section in sections:
        if not config.get(section):
            missing.append(section)
    if len(missing) > 0:
        raise RuntimeError(
            f"the following configuration keys must be defined "
            f"and not empty/null: {missing}")


def prepare_config_listify_sections(config, sections):
    for section in sections:
        if config.get(section):
            config[section] = iterutils.listify(config[section])


def prepare_config_check_sections_length(config, sections):
    lengths = {}
    for section in sections:
        if config.get(section):
            lengths[section] = len(config[section])
    if len(set(lengths.values())) > 1:
        raise RuntimeError(
            f"the following configuration sections must have the same length: "
            f"{list(lengths.keys())}")


def prepare_config_pdescs(config):
    pdescs = {}
    invalid = []
    for key, value in config.items():
        if not isinstance(value, dict):
            invalid.append(key)
            continue
        value['name'] = key
        pdescs[key] = value
    if invalid:
        raise RuntimeError(
            f"the values of the following pdescs must be a dictionary: "
            f"{invalid}")
    return pdescs


def make_models(dmodels, gmodels, drivers, brokers, pdescs):
    models = []
    for i in range(len(dmodels)):
        models.append(gbkfit.model.model.Model(
            dmodels[i], gmodels[i],
            drivers[i] if drivers else None,
            brokers[i] if brokers else None))
    param_descs = {}
    param_mappings = []
    for i, model in enumerate(models):
        param_mappings.append({})
        for old_name, desc in model.params().items():
            new_name = f'model{i}_{old_name}' if i else old_name
            param_mappings[i][old_name] = new_name
            param_descs[new_name] = desc
    if pdescs:
        duplicates = set(param_descs).intersection(pdescs)
        if duplicates:
            raise RuntimeError(
                f"the following parameter names are present in both model and "
                f"pdescs: {duplicates}")
        param_descs.update(pdescs)
    return models, param_descs, param_mappings


def nativify(node):
    if isinstance(node, np.ndarray):
        node = node.tolist()
    elif isinstance(node, np.integer):
        node = int(node)
    elif isinstance(node, np.floating):
        node = float(node)
    elif isinstance(node, list):
        for i in range(len(node)):
            node[i] = nativify(node[i])
    elif isinstance(node, dict):
        for key in node:
            node[key] = nativify(node[key])
    return node
