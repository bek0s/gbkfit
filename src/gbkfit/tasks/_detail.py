
import logging

import numpy as np

import gbkfit.model.model
import gbkfit.model.utils
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
                f"{', '.joint(invalid_pdescs)}")

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


def setup_datasets(cfg):
    datasets = None
    if cfg.get('datasets'):
        log.info("setting up datasets...")
        datasets = gbkfit.dataset.parser.load_many(cfg['datasets'])
    return datasets


def setup_drivers(cfg):
    drivers = None
    if cfg.get('drivers'):
        log.info("setting up drivers...")
        drivers = gbkfit.driver.driver.parser.load_many(cfg['drivers'])
    return drivers


def setup_dmodels(cfg, datasets):
    log.info("setting up dmodels...")
    return gbkfit.model.dmodel.parser.load_many(cfg['dmodels'], datasets)


def setup_gmodels(cfg):
    log.info("setting up gmodels...")
    return gbkfit.model.gmodel.parser.load_many(cfg['gmodels'])


def setup_fitter(cfg):
    log.info("setting up fitter...")
    return gbkfit.fitting.fitter.parser.load_one(cfg['fitter'])


def setup_objectives(cfg, datasets, drivers, dmodels, gmodels, fitter):
    if cfg.get('objectives'):
        log.info("setting up objectives...")
        objectives = gbkfit.fitting.objective.parser.load_many(
            cfg['objectives'],
            dataset=datasets, driver=drivers, dmodel=dmodels, gmodel=gmodels)
        objectives_weight = [o.get('weight', 1.) for o in cfg['objectives']]
    else:
        objectives = []
        objectives_weight = []
        for i in range(len(drivers)):
            objectives.append(fitter.default_objective(
                datasets[i], drivers[i], dmodels[i], gmodels[i]))
            objectives_weight.append(1.0)
    objective = gbkfit.fitting.objective.JointObjective(objectives, objectives_weight)

    return objective, objectives_weight


def setup_pdescs(cfg):
    pdescs = None
    if cfg.get('pdescs'):
        log.info("setting up pdescs...")
        pdesc_keys = cfg['pdescs'].keys()
        pdesc_vals = cfg['pdescs'].values()
        pdesc_list = gbkfit.params.descs.parser.load_many(pdesc_vals)
        pdescs = dict(zip(pdesc_keys, pdesc_list))
    return pdescs