
import json
import logging
import time

import astropy.io.fits as fits
import numpy as np
import ruamel.yaml as yaml
import scipy.stats as stats

import gbkfit
import gbkfit.broker
import gbkfit.dataset
import gbkfit.dmodel
import gbkfit.driver
import gbkfit.gmodel
import gbkfit.model
import gbkfit.params
import gbkfit.params.descs
from . import _detail


log = logging.getLogger(__name__)


def _prepare_config(config):
    _detail.prepare_config_require_sections(
        config, ['dmodels', 'gmodels', 'params'])
    _detail.prepare_config_listify_sections(
        config, ['brokers', 'drivers', 'datasets', 'dmodels', 'gmodels'])
    _detail.prepare_config_check_sections_length(
        config, ['brokers', 'drivers', 'datasets', 'dmodels', 'gmodels'])


def _prepare_params(info, descs):

    # Overwrite param info with only known and valid keys
    info_keys, info_values, _, _ = \
        gbkfit.params.parse_param_keys(info, descs)
    info = dict(zip(info_keys, info_values))

    # ...
    param_exprs = {}
    param_values = {}
    required_xor = ['expr', 'init', 'value']
    recovery_failed = []
    recovery_succeed = {}
    for key, value in info.items():
        # Values of dictionary type are assumed to be fit configs.
        # Try to transform the dictionary to a valid value.
        if isinstance(value, dict):
            items_found = list(set(value).intersection(required_xor))
            if len(items_found) != 1:
                recovery_failed.append(key)
                continue
            value = value[items_found[0]]
            recovery_succeed[key] = value
        # Values of string type are treated as expressions
        if isinstance(value, str):
            param_exprs[key] = value
        # Values of any other type are treated as normal values
        else:
            param_values[key] = value

    if recovery_failed:
        raise RuntimeError(
            f"failed to recover expressions and/or values "
            f"for the following keys: {recovery_failed}; "
            f"only one of the following keys must be provided: {required_xor}")
    if recovery_succeed:
        log.info(
            f"successfully recovered expressions and/or values "
            f"for the following: {recovery_succeed}")

    # Validate param expressions
    gbkfit.params.parse_param_exprs(param_exprs, descs)
    # Validate and explode param values
    _, _, _, _, \
    param_values = gbkfit.params.parse_param_values(param_values, descs)

    return param_exprs, param_values


def eval_(config, perf=None):

    log.info("Initializing gbkfit...")
    gbkfit.init()

    log.info(f"Reading configuration file: '{config}'...")
    config = yaml.YAML().load(open(config))
    _prepare_config(config)

    #
    # Setup optional stuff
    #

    brokers = None
    if config.get('brokers'):
        log.info("Setting up brokers...")
        brokers = gbkfit.broker.parser.load_many(config['brokers'])

    drivers = None
    if config.get('drivers'):
        log.info("Setting up drivers...")
        drivers = gbkfit.driver.parser.load_many(config['drivers'])

    datasets = None
    if config.get('datasets'):
        log.info("Setting up datasets...")
        datasets = gbkfit.dataset.parser.load_many(config['datasets'])

    pdescs = None
    if config.get('pdescs'):
        log.info("Setting up pdescs...")
        pdesc_info = _detail.prepare_config_pdescs(config['pdescs'])
        pdesc_keys = pdesc_info.keys()
        pdesc_vals = pdesc_info.values()
        pdesc_list = gbkfit.params.descs.parser.load_many(pdesc_vals)
        pdescs = dict(zip(pdesc_keys, pdesc_list))

    #
    # Setup required stuff
    #

    log.info("Setting up dmodels...")
    dmodels = gbkfit.dmodel.parser.load_many(config['dmodels'], datasets)

    log.info("Setting up gmodels...")
    gmodels = gbkfit.gmodel.parser.load_many(config['gmodels'])

    log.info("Setting up models...")
    models, param_descs, param_mappings = _detail.make_models(
        dmodels, gmodels, drivers, brokers, pdescs)

    log.info("Setting up params...")
    param_exprs, param_values = _prepare_params(config['params'], param_descs)

    #
    # Calculate model parameters
    #

    interpreter = gbkfit.params.ParamInterpreter(param_descs, param_exprs)
    result_params = interpreter.evaluate(param_values, True)
    params_list = [{param: result_params[mapping[param]] for param in mapping}
                   for mapping in param_mappings]

    filename = 'gbkfit_result_params'
    params_list = _detail.nativify(params_list)
    json.dump(params_list, open(f'{filename}.json', 'w+'))
    yaml.dump(params_list, open(f'{filename}.yaml', 'w+'))

    #
    # Evaluate models
    #

    log.info("Model evaluation started.")
    t0 = time.time_ns()
    out_models = []
    out_dextra = []
    out_gextra = []
    for model, params in zip(models, params_list):
        out_dextra_i = {}
        out_gextra_i = {}
        out_models.append(model.evaluate(params, out_dextra_i, out_gextra_i))
        out_dextra.append(out_dextra_i)
        out_gextra.append(out_gextra_i)
    t1 = time.time_ns()
    t_ms = (t1 - t0) // 1000000
    log.info("Model evaluation completed.")
    log.info(f"Elapsed time: {t_ms} ms.")

    def save_output(filename, data):
        hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=True)

    log.info("Writing output to the filesystem...")
    for i in range(len(out_models)):
        prefix = 'model' + f'_{i}' * i
        for key, value in out_models[i].items():
            save_output(f'{prefix}_{key}.fits', value)
        for key, value in out_dextra[i].items():
            save_output(f'{prefix}_extra_dmodel_{key}.fits', value)
        for key, value in out_gextra[i].items():
            save_output(f'{prefix}_extra_gmodel_{key}.fits', value)

    #
    # Run performance tests, if requested
    #

    if perf > 0:
        log.info("Model evaluation performance test started.")
        times = []
        for i in range(perf):
            t0 = time.time_ns()
            for model, params in zip(models, params_list):
                model.evaluate(params)
            t1 = time.time_ns()
            t_ms = (t1 - t0) // 1000000
            times.append(t_ms)
            log.info(f"Evaluation {i}: {t_ms} ms")
        log.info(f"Model evaluation performance test completed.")
        time_stats = _detail.nativify(dict(
            min=np.round(np.min(times), 1),
            max=np.round(np.max(times), 1),
            mean=np.round(np.mean(times), 1),
            stddev=np.round(np.std(times), 1),
            median=np.round(np.median(times), 1),
            mad=np.round(stats.median_absolute_deviation(times), 1)))
        log.info(', '.join(f'{k}: {v} ms' for k, v in time_stats.items()))
        filename = 'gbkfit_result_time'
        json.dump(time_stats, open(f'{filename}.json', 'w+'))
        yaml.dump(time_stats, open(f'{filename}.yaml', 'w+'))
