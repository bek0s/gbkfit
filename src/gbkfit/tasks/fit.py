
import json
import logging
import time


import gbkfit
import gbkfit.broker
import gbkfit.dataset
import gbkfit.dmodel
import gbkfit.driver
import gbkfit.gmodel
import gbkfit.model
import gbkfit.params
from . import _detail

log = logging.getLogger(__name__)


def _prepare_config(config):

    _detail.require_config_sections(
        config, ['datasets', 'dmodels', 'gmodels', 'fitter', 'params'])

    _detail.listify_config_sections(
        config, ['brokers', 'drivers', 'datasets', 'dmodels', 'gmodels'])

    _detail.check_config_sections_length(
        config, ['datasets', 'dmodels'])


def fit(config):

    log.info("Initializing gbkfit...")
    gbkfit.init()

    log.info(f"Reading configuration file: '{config}'...")
    config = json.load(open(config))
    _prepare_config(config)

    brokers = None
    if config.get('brokers'):
        log.info("Setting up brokers...")
        brokers = gbkfit.broker.parser.load(config['brokers'])

    drivers = None
    if config.get('drivers'):
        log.info("Setting up drivers...")
        drivers = gbkfit.driver.parser.load(config['drivers'])

    log.info("Setting up datasets...")
    datasets = gbkfit.dataset.parser.load(config['datasets'])

    log.info("Setting up dmodels...")
    dmodels = gbkfit.dmodel.parser.load(config['dmodels'])

    log.info("Setting up gmodels...")
    gmodels = gbkfit.gmodel.parser.load(config['gmodels'])

    log.info("Setting up model...")
    model = gbkfit.model.Model(dmodels, gmodels, drivers, brokers)

    log.info("Setting up params...")
    params = gbkfit.params.convert_params_free_to_fixed(config['params'])
    model.set_param_exprs(params)

    log.info("Model-fitting started")
    t1 = time.time_ns()

    t2 = time.time_ns()
    t_ms = (t2 - t1) // 1000000
    log.info("Model-fitting completed.")
    log.info(f"Elapsed time: {t_ms} ms.")
