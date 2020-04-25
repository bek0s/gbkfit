
import json
import logging
import time

import astropy.io.fits as fits
import numpy as np
import ruamel.yaml as yaml
import scipy.stats as stats

import gbkfit
import gbkfit.dataset.dataset
import gbkfit.driver
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


def _prepare_pdescs(gmodels, extra_pdescs=None):
    pdescs, pdescs_mappings = gbkfit.params.utils.merge_pdescs(
        [gmodel.params() for gmodel in gmodels])
    if extra_pdescs:
        duplicates = set(pdescs).intersection(extra_pdescs)
        if duplicates:
            raise RuntimeError(
                f"the following parameter descriptions are present in both "
                f"model and pdescs: {', '.join(duplicates)}")
        pdescs.update(extra_pdescs)
    return pdescs, pdescs_mappings


def _prepare_params(info, descs):

    # Validate and cleanup param info dict keys
    info_keys, info_values = gbkfit.params.parse_param_keys(info, descs)[:2]
    info = dict(zip(info_keys, info_values))

    # Check if the values of the
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
    if recovery_succeed:
        log.info(
            f"successfully recovered expressions and/or values "
            f"from the following: {recovery_succeed}")
    if recovery_failed:
        raise RuntimeError(
            f"failed to recover expressions and/or values "
            f"from the following keys: {recovery_failed}; "
            f"only one of the following keys must be provided: {required_xor}")

    # Validate param expressions
    gbkfit.params.parse_param_exprs(param_exprs, descs)
    # Validate and explode param values
    param_values = gbkfit.params.parse_param_values(param_values, descs)[4]

    return param_exprs, param_values


def eval_(config, perf=None):

    #
    # Read configuration file and
    # perform all necessary validation/patching/preparation
    #

    log.info(f"reading configuration file: '{config}'...")
    config = _detail.prepare_config(
        yaml.YAML().load(open(config)),
        ('drivers', 'dmodels', 'gmodels', 'params'),
        ('datasets', 'pdescs'))

    #
    # Setup all the components described in the configuration
    #

    datasets = None
    if config.get('datasets'):
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

    log.info("setting up pdescs...")
    pdescs_extra = None
    if config.get('pdescs'):
        pdesc_keys = config['pdescs'].keys()
        pdesc_vals = config['pdescs'].values()
        pdesc_list = gbkfit.params.descs.parser.load_many(pdesc_vals)
        pdescs_extra = dict(zip(pdesc_keys, pdesc_list))
    pdescs_all, pdescs_mappings = _prepare_pdescs(gmodels, pdescs_extra)

    log.info("setting up params...")
    exprs, eparams = _prepare_params(config['params'], pdescs_all)

    #
    # Calculate model parameters
    #

    interpreter = gbkfit.params.ParamInterpreter(pdescs_all, exprs)
    params_all = interpreter.evaluate(eparams, True)
    params_list = [{param: params_all[mapping[param]] for param in mapping}
                   for mapping in pdescs_mappings]

    filename = 'gbkfit_result_params'
    params_all = _detail.nativify(params_all)
    json.dump(params_all, open(f'{filename}.json', 'w+'))
    yaml.dump(params_all, open(f'{filename}.yaml', 'w+'))

    #
    # Evaluate models
    #

    log.info("evaluating model...")
    models = []
    extras = []
    for driver, dmodel, gmodel, params in zip(
            drivers, dmodels, gmodels, params_list):
        extra = {}
        model_d = dmodel.evaluate(driver, gmodel, params, extra)
        model_h = {k: driver.mem_copy_d2h(v) for k, v in model_d.items()}
        models.append(model_h)
        extras.append(extra)

    def save_model(file, data):
        hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(file, overwrite=True)

    log.info("writing model to the filesystem...")
    for i in range(len(models)):
        prefix = 'model' + f'_{i}' * bool(i)
        for key, value in models[i].items():
            save_model(f'{prefix}_{key}.fits', value)
        for key, value in extras[i].items():
            save_model(f'{prefix}_extra_{key}.fits', value)

    #
    # Run performance tests
    #

    if perf > 0:
        log.info("running performance test...")
        times = []
        for i in range(perf):
            t1 = time.time_ns()
            for driver, dmodel, gmodel, params in zip(
                    drivers, dmodels, gmodels, params_list):
                model_d = dmodel.evaluate(driver, gmodel, params)
                _ = {k: driver.mem_copy_d2h(v) for k, v in model_d.items()}
            t2 = time.time_ns()
            t_ms = (t2 - t1) // 1000000
            times.append(t_ms)
            log.info(f"evaluation {i}: {t_ms} ms")
        log.info("calculating performance test statistics...")
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
