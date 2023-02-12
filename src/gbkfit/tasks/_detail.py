
import json
import logging
import os

from gbkfit.utils import iterutils, miscutils


_log = logging.getLogger(__name__)


def prepare_config(config, req_sections=(), opt_sections=()):

    def _is_empty_section(info):
        return info is None or (iterutils.is_iterable(info) and len(info) == 0)

    # If the configuration file was empty, config will be None
    # Convert it to an empty dict to keep the validation rolling
    if config is None:
        config = {}

    # Get rid of unrecognised and empty optional sections
    empty_sections = []
    known_sections = []
    unknown_sections = []
    for s in config:
        if s in opt_sections and _is_empty_section(config[s]):
            empty_sections.append(s)
        elif s in req_sections + opt_sections:
            known_sections.append(s)
        else:
            unknown_sections.append(s)
    if empty_sections:
        _log.info(
            f"the following optional sections are empty and will be ignored: "
            f"{empty_sections}")
    if unknown_sections:
        _log.info(
            f"the following sections are not recognised and will be ignored: "
            f"{unknown_sections}")
    config = {s: config[s] for s in known_sections}

    # Ensure that the required sections are present/valid
    # and keep track of the missing ones
    missing_sections = []
    for s in req_sections:
        if s not in config or _is_empty_section(config[s]):
            missing_sections.append(s)
    if missing_sections:
        raise RuntimeError(
            f"the following sections must be defined and not empty/null: "
            f"{missing_sections}")

    # Ensure the sections have the right type (if they are present)
    wrong_type_dict = []
    wrong_type_dict_or_seq = []
    for s in ['pdescs', 'params', 'objective', 'fitter']:
        if s in config and not iterutils.is_mapping(config[s]):
            wrong_type_dict.append(s)
    for s in ['datasets', 'drivers', 'dmodels', 'gmodels']:
        if s in config and not iterutils.is_iterable(config[s]):
            wrong_type_dict_or_seq.append(s)
    if wrong_type_dict:
        raise RuntimeError(
            f"the following sections must be dictionaries: "
            f"{wrong_type_dict}")
    if wrong_type_dict_or_seq:
        raise RuntimeError(
            f"the following sections must be dictionaries or sequences: "
            f"{wrong_type_dict_or_seq}")

    # Listify some sections to make parsing more streamlined
    # and ensure they have the same length
    lengths = {}
    for s in ['datasets', 'drivers', 'dmodels', 'gmodels']:
        if s in config:
            config[s] = iterutils.listify(config[s])
            lengths[s] = len(config[s])
    if len(set(lengths.values())) > 1:
        raise RuntimeError(
            f"the following sections must have the same length: "
            f"{lengths}")

    # Make sure the return value is pure json
    return json.loads(json.dumps(config))


def merge_pdescs(dict1, dict2):
    if dict1 is None:
        dict1 = {}
    if dict2 is None:
        dict2 = {}
    if conflicting := set(dict1) & set(dict2):
        raise RuntimeError(
            f"the names of the following user-defined pdescs "
            f"conflict with the names of the model parameters: {conflicting}")
    return dict1 | dict2


def make_output_dir(output_dir, output_dir_unique):
    output_dir = os.path.abspath(output_dir)
    output_dir_exists = os.path.exists(output_dir)
    output_dir_isdir = os.path.isdir(output_dir)
    if output_dir_exists:
        if output_dir_unique:
            output_dir = miscutils.make_unique_path(output_dir)
            os.makedirs(output_dir)
        else:
            if not output_dir_isdir:
                raise RuntimeError(f"{output_dir} already exists as a file")
    else:
        os.makedirs(output_dir)
    return output_dir


def dump_dict(json_, yaml_, info, filename):
    json_.dump(info, open(f'{filename}.json', 'w+'), indent=2)
    yaml_.dump(info, open(f'{filename}.yaml', 'w+'))
