
import json
import logging
import time


import gbkfit
import gbkfit.broker
import gbkfit.dmodel
import gbkfit.driver
import gbkfit.gmodel
import gbkfit.model
import gbkfit.params

log = logging.getLogger(__name__)


def fit(config):

    log.info("Initializing gbkfit...")
    gbkfit.init()

    log.info(f"Reading configuration file: '{config}'...")
    config = json.load(open(config))

    log.info("Setting up brokers...")
    brokers = gbkfit.broker.parser.load(config.get('brokers'))

    log.info("Setting up backends...")
    drivers = gbkfit.driver.parser.load(config['drivers'])

    log.info("Setting up dmodels...")
    dmodels = gbkfit.dmodel.parser.load(config['dmodels'])

    log.info("Setting up gmodels...")
    gmodels = gbkfit.gmodel.parser.load(config['gmodels'])

    log.info("Setting up model...")
    model = gbkfit.model.Model(dmodels, gmodels, drivers, brokers)

    log.info("Setting up params...")
    params = gbkfit.params.convert_params_free_to_fixed(config['params'])
    model.set_param_exprs(params)
