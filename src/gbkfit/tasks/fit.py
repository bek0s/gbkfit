
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
import gbkfit.utils.miscutils
from . import _detail

import gbkfit.params.interpreter
import numpy as np
log = logging.getLogger(__name__)


# This is needed for dumping dicts in the correct order
yaml.add_representer(dict, lambda self, data: self.represent_mapping(
    'tag:yaml.org,2002:map', data.items()))


def _prepare_pdescs(objectives, extra_pdescs):
    pdescs, mappings = gbkfit.utils.miscutils.merge_dicts_and_make_mappings(
        [objective.params() for objective in objectives], 'model')
    if extra_pdescs:
        duplicates = set(pdescs).intersection(extra_pdescs)
        if duplicates:
            raise RuntimeError(
                f"the following parameter descriptions are present in both "
                f"model and pdescs: {str(duplicates)}")
        pdescs.update(extra_pdescs)
    return pdescs, mappings


def fit(config):

    #
    # Read configuration file and
    # perform all necessary validation/patching/preparation
    #

    log.info(f"reading configuration file: '{config}'...")
    config = _detail.prepare_config(
        yaml.YAML().load(open(config)),
        ('datasets', 'drivers', 'dmodels', 'gmodels', 'params', 'fitter'),
        ('pdescs',))

    #
    # Setup all the components described in the configuration
    #

    log.info("setting up datasets...")
    datasets = gbkfit.dataset.dataset.parser.load_many(config['datasets'])

    drivers = None
    if config.get('drivers'):
        log.info("setting up drivers...")
        drivers = gbkfit.driver.driver.parser.load_many(config['drivers'])
    """
    foo = [d.dump() for d in datasets]
    print(foo)
    exit()
    """
    log.info("setting up dmodels...")
    dmodels = gbkfit.model.dmodel.parser.load_many(config['dmodels'], dataset=datasets)

    log.info("setting up gmodels...")
    gmodels = gbkfit.model.gmodel.parser.load_many(config['gmodels'])

    log.info("setting up fitter...")
    fitter = gbkfit.fitting.fitter.parser.load_one(config['fitter'])

    log.info("setting up objective...")
    objective = gbkfit.fitting.objective.Objective(
        datasets, dmodels, gmodels, drivers)

    log.info("setting up pdescs...")
    pdescs_extra = None
    if config.get('pdescs'):
        pdesc_keys = config['pdescs'].keys()
        pdesc_vals = config['pdescs'].values()
        pdesc_list = gbkfit.params.descs.parser.load_many(pdesc_vals)
        pdescs_extra = dict(zip(pdesc_keys, pdesc_list))
    pdescs_all, pdescs_mappings = _prepare_pdescs(gmodels, pdescs_extra)

    log.info("setting up params...")
    params = fitter.load_params(config['params'], pdescs_all)

    #
    # Perform fit
    #

    log.info("model-fitting started")
    t1 = time.time_ns()
    print(params)

    """
    interpreter = gbkfit.params.interpreter.ParamInterpreter(
        params.descs(), params.exprs())

    for i in range(1000):
        residual = objective.residual_scalar(interpreter.evaluate(dict(
            xpos=10, ypos=10
        )))
        print(residual)
        #print(np.sum(np.abs(residual)))
    """

    result = fitter.fit(objective, params)

    t2 = time.time_ns()
    t_ms = (t2 - t1) // 1000000
    log.info(f"model-fitting completed (elapsed time: {t_ms} ms)")
