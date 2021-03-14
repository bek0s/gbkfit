
import json
import logging

import numpy as np

from gbkfit.utils import iterutils


log = logging.getLogger(__name__)


def prepare_config(config, req_sections=(), opt_sections=()):

    def _is_empty_section(info):
        return info is None or (iterutils.is_iterable(info) and len(info) == 0)

    # Get rid of unrecognised and empty optional sections
    empty_sections = []
    known_sections = []
    unknown_sections = []
    for scn in config:
        if scn in opt_sections and _is_empty_section(config[scn]):
            empty_sections.append(scn)
        elif scn in req_sections + opt_sections:
            known_sections.append(scn)
        else:
            unknown_sections.append(scn)
    if empty_sections:
        log.info(
            f"the following optional sections are empty and will be ignored: "
            f"{str(empty_sections)}")
    if unknown_sections:
        log.info(
            f"the following sections are not recognised and will be ignored: "
            f"{str(unknown_sections)}")
    config = {scn: config[scn] for scn in known_sections}

    # Ensure that the required sections are present and valid
    missing_sections = []
    for scn in req_sections:
        if scn not in config or _is_empty_section(config[scn]):
            missing_sections.append(scn)
    if missing_sections:
        raise RuntimeError(
            f"the following sections must be defined and not empty/null: "
            f"{str(missing_sections)}")

    # Ensure that the sections have the right type
    wrong_type_dict = []
    wrong_type_dict_seq = []
    for scn in ['objective', 'fitter', 'pdescs', 'params']:
        if scn in config and not iterutils.is_mapping(config[scn]):
            wrong_type_dict.append(scn)
    for scn in ['datasets', 'drivers', 'dmodels', 'gmodels']:
        if scn in config and not iterutils.is_iterable(config[scn]):
            wrong_type_dict_seq.append(scn)
    if wrong_type_dict:
        raise RuntimeError(
            f"the following sections must be dictionaries: "
            f"{str(wrong_type_dict)}")
    if wrong_type_dict_seq:
        raise RuntimeError(
            f"the following sections must be dictionaries or sequences: "
            f"{str(wrong_type_dict_seq)}")

    # Listify some sections to make parsing more streamlined
    for scn in ['datasets', 'drivers', 'dmodels', 'gmodels']:
        if scn in config:
            config[scn] = iterutils.listify(config[scn])

    # Ensure that some sections have the same length
    lengths = {}
    for scn in ['datasets', 'drivers', 'dmodels', 'gmodels']:
        if scn in config:
            lengths[scn] = len(config[scn])
    if len(set(lengths.values())) > 1:
        raise RuntimeError(
            f"the following sections must have the same length: "
            f"{str(lengths)}")

    # Place pdesc keys as names inside values.
    # This will make them readable by the pdesc parser.
    invalid_pdescs = []
    pdesc_info = config.get('pdesc')
    if pdesc_info:
        for key, val in pdesc_info.items():
            if not iterutils.is_mapping(val):
                invalid_pdescs.append(key)
                continue
            val['name'] = key
            pdesc_info[key] = val
        if invalid_pdescs:
            raise RuntimeError(
                f"the values of the following pdescs must be a dictionary: "
                f"{str(invalid_pdescs)}")

    # Make sure the return value is pure json
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
        for k in node:
            node[k] = nativify(node[k])
    return node
