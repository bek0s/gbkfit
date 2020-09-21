
import contextlib
import json
import logging
import os

import numpy as np

from gbkfit.utils import iterutils


log = logging.getLogger(__name__)


def prepare_config(config, req_sections=(), opt_sections=()):

    # Get rid of all unrecognised sections and
    # all empty optional sections
    empty_sections = []
    known_sections = []
    unknown_sections = []
    for section in config:
        if section in opt_sections and not config[section]:
            empty_sections.append(section)
        elif section in req_sections + opt_sections:
            known_sections.append(section)
        else:
            unknown_sections.append(section)
    if empty_sections:
        log.info(
            f"the following optional sections are empty and will be ignored: "
            f"{', '.join(empty_sections)}")
    if unknown_sections:
        log.info(
            f"the following sections are not recognised and will be ignored: "
            f"{', '.join(unknown_sections)}")
    config = {section: config[section] for section in known_sections}

    # Ensure that the required sections are present
    missing_sections = []
    for section in req_sections:
        if not config.get(section):
            missing_sections.append(section)
    if len(missing_sections) > 0:
        raise RuntimeError(
            f"the following sections must be defined and not empty/null: "
            f"{', '.join(missing_sections)}")

    # Make sure the sections have the right type
    wrong_type_dict = []
    wrong_type_dict_seq = []
    for section in ['fitter', 'pdescs', 'params']:
        if (section in config
                and not isinstance(config[section], (dict,))):
            wrong_type_dict.append(section)
    for section in ['objectives', 'datasets', 'drivers', 'dmodels', 'gmodels']:
        if (section in config
                and not isinstance(config[section], (dict, list, tuple, set))):
            wrong_type_dict_seq.append(section)
    if wrong_type_dict:
        raise RuntimeError(
            f"the following sections must be dictionaries: "
            f"{', '.join(wrong_type_dict)}")
    if wrong_type_dict_seq:
        raise RuntimeError(
            f"the following sections must be dictionaries or sequences: "
            f"{', '.join(wrong_type_dict_seq)}")

    # Listify some sections to make parsing more streamlined
    for section in ['objectives', 'datasets', 'drivers', 'dmodels', 'gmodels']:
        if section in config:
            config[section] = iterutils.listify(config[section])

    # Make sure some sections have the same length
    lengths = {}
    for section in ['objectives', 'datasets', 'drivers', 'dmodels', 'gmodels']:
        if section in config:
            lengths[section] = len(config[section])
    if len(set(lengths.values())) > 1:
        raise RuntimeError(
            f"the following sections must have the same length: "
            f"{', '.join(lengths)}")

    """
    datasets = config.get('datasets')
    dmodels = config.get('dmodels')
    if datasets and dmodels:
        for dataset, dmodel in zip(datasets, dmodels):
            if dataset.get('type') is None:
                dataset['type'] = dmodel.get('type')

        pass
    """

    # Place pdesc keys inside values
    # in order to make them readable by the pdesc parser.
    invalid_pdescs = []
    if 'pdescs' in config:
        for key, value in config['pdescs'].items():
            if not isinstance(value, dict):
                invalid_pdescs.append(key)
                continue
            value['name'] = key
            config['pdescs'][key] = value
        if invalid_pdescs:
            raise RuntimeError(
                f"the values of the following pdescs must be a dictionary: "
                f"{str(invalid_pdescs)}")

    for pname, pinfo in config['params'].items():
        #print(';')
        pass

    config = json.loads(json.dumps(config))

    return config


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





@contextlib.contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
