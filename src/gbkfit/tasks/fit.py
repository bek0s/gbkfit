
import logging
import time

import ruamel.yaml as yaml

import gbkfit
import gbkfit.broker
import gbkfit.dataset
import gbkfit.dmodel
import gbkfit.driver
import gbkfit.fitter
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
        config, ['datasets', 'dmodels', 'gmodels'])


def _prepare_params(params):
    param_infos = {}
    param_exprs = {}
    for key, value in params.items():
        if isinstance(value, dict):
            param_infos[key] = value
        else:
            param_exprs[key] = value
    return param_infos, param_exprs


def fit(config):

    log.info("Initializing gbkfit...")
    gbkfit.init()

    log.info(f"Reading configuration file: '{config}'...")
    config = yaml.YAML().load(open(config))
    _prepare_config(config)

    brokers = None
    if config.get('brokers'):
        log.info("Setting up brokers...")
        brokers = gbkfit.broker.parser.load_many(config['brokers'])

    drivers = None
    if config.get('drivers'):
        log.info("Setting up drivers...")
        drivers = gbkfit.driver.parser.load_many(config['drivers'])

    log.info("Setting up datasets...")
    datasets = gbkfit.dataset.parser.load_many(config['datasets'])

    log.info("Setting up dmodels...")
    dmodels = gbkfit.dmodel.parser.load_many(config['dmodels'], datasets)

    log.info("Setting up gmodels...")
    gmodels = gbkfit.gmodel.parser.load_many(config['gmodels'])

    log.info("Setting up model...")
    model = gbkfit.model.Model(dmodels, gmodels, drivers, brokers)

    log.info("Setting up params...")
    param_infos, param_exprs = _prepare_params(config['params'])
    param_info = gbkfit.params.parse_param_fit_info(param_infos, model.get_param_descs())
    model.set_param_exprs(param_exprs)
    print(param_infos, param_exprs)

    exit()

    """
    for k, v in param_infos.items():
        print(k, ': ', v)
    print(param_exprs)
    exit()
    """

    """
    param_infos, param_exprs = _prepare_params(config['params'])
    model.set_param_exprs(param_exprs)
    """

    log.info("Setting up fitter...")
    fitter = gbkfit.fitter.parser.load_one(config['fitter'])

    log.info("Model-fitting started")
    t1 = time.time_ns()

    result = fitter.fit(datasets, model, param_infos)
    """
    from gbkfit.fitters.dynesty.dynesty import FitterDynestyStaticNestedSampling
    fitter = FitterDynestyStaticNestedSampling(foo='bar')
    fitter.fit(datasets, model, param_infos)
    """



    """
    print("-----------------------------")
    print(result)
    print("-----------------------------")
    for k, v in zip(model.get_param_names(), result.x):
        print(f'{k}: {round(v, 3)}')
    """



    t2 = time.time_ns()
    t_ms = (t2 - t1) // 1000000
    log.info("Model-fitting completed.")
    log.info(f"Elapsed time: {t_ms} ms.")
