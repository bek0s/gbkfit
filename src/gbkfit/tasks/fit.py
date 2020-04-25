
import json
import logging
import time

import ruamel.yaml as yaml

import gbkfit
import gbkfit.dataset.dataset
import gbkfit.driver
import gbkfit.fitting.fitter
import gbkfit.fitting.objective
import gbkfit.model.dmodel
import gbkfit.model.gmodel
import gbkfit.params
import gbkfit.params.descs
import gbkfit.params.utils
from . import _detail


log = logging.getLogger(__name__)


# This is needed for dumping dicts in the correct order
yaml.add_representer(dict, lambda self, data: self.represent_mapping(
    'tag:yaml.org,2002:map', data.items()))


def _prepare_pdescs(objectives, extra_pdescs=None):
    pdescs, pdescs_mappings = gbkfit.params.utils.merge_pdescs(
        [objective.params() for objective in objectives])
    if extra_pdescs:
        duplicates = set(pdescs).intersection(extra_pdescs)
        if duplicates:
            raise RuntimeError(
                f"the following parameter descriptions are present in both "
                f"model and pdescs: {', '.join(duplicates)}")
        pdescs.update(extra_pdescs)
    return pdescs, pdescs_mappings


def _prepare_params(info, descs):

    # Remove all param info pairs with invalid keys
    info_keys, info_values, _, _ = \
        gbkfit.params.parse_param_keys(info, descs)
    info = dict(zip(info_keys, info_values))

    param_exprs = {}
    param_infos = {}
    for key, value in info.items():

        if not isinstance(value, dict):
            param_exprs[key] = value
        elif 'expr' in value:
            param_exprs[key] = value['expr']
        else:
            param_infos[key] = value

    # ...
    gbkfit.params.parse_param_exprs(param_exprs, descs)
    foo = gbkfit.params.parse_param_fit_info(param_infos, descs)

    return param_exprs, foo


def fit(config):

    #
    # Read configuration file and
    # perform all necessary validation/patching/preparation
    #

    log.info(f"reading configuration file: '{config}'...")
    config = _detail.prepare_config(
        yaml.YAML().load(open(config)),
        ('datasets', 'drivers', 'dmodels', 'gmodels', 'params', 'fitter'),
        ('objectives', 'pdescs'))

    #
    # Setup all the components described in the configuration
    #

    log.info("setting up datasets...")
    datasets = gbkfit.dataset.dataset.parser.load_many(config['datasets'])

    drivers = None
    if config.get('drivers'):
        log.info("setting up drivers...")
        drivers = gbkfit.driver.driver.parser.load_many(config['drivers'])

    log.info("setting up dmodels...")
    dmodels = gbkfit.model.dmodel.parser.load_many(
        config['dmodels'], dataset=datasets)

    log.info("setting up gmodels...")
    gmodels = gbkfit.model.gmodel.parser.load_many(config['gmodels'])

    log.info("setting up fitter...")
    fitter = gbkfit.fitting.fitter.parser.load_one(config['fitter'])

    log.info("setting up objectives...")
    if config.get('objectives'):
        objectives = gbkfit.fitting.objective.parser.load_many(
            config['objectives'],
            dataset=datasets, driver=drivers, dmodel=dmodels, gmodel=gmodels)
        objectives_weight = [o.get('weight', 1.) for o in config['objectives']]
    else:
        objectives = []
        objectives_weight = []
        for dataset, driver, dmodel, gmodel in zip(
                datasets, drivers, dmodels, gmodels):
            objectives.append(fitter.default_objective(
                dataset, driver, dmodel, gmodel))
            objectives_weight.append(1.0)
    objective = gbkfit.fitting.objective.JointObjective(objectives, objectives_weight)

    log.info("setting up pdescs...")
    pdescs_extra = None
    if config.get('pdescs'):
        pdesc_keys = config['pdescs'].keys()
        pdesc_vals = config['pdescs'].values()
        pdesc_list = gbkfit.params.descs.parser.load_many(pdesc_vals)
        pdescs_extra = dict(zip(pdesc_keys, pdesc_list))
    pdescs_all, pdescs_mappings = _prepare_pdescs(gmodels, pdescs_extra)

    log.info("setting up params...")
    param_exprs, param_infos = _prepare_params(config['params'], objective.params())

    #
    # Perform fit
    #

    log.info("model-fitting started")
    t1 = time.time_ns()

    print(param_exprs)

    residual = objectives[0].residual(param_exprs)

    exit()

    #exit()
    result = fitter.fit(objectives, None, None, param_infos, None)
    #exit()

    t2 = time.time_ns()
    t_ms = (t2 - t1) // 1000000
    log.info(f"model-fitting completed (elapsed time: {t_ms} ms)")
