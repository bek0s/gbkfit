
import json
import logging
import time

import astropy.io.fits as fits
import numpy as np
import ruamel.yaml
import scipy.stats as stats

import gbkfit
import gbkfit.dataset
import gbkfit.driver
import gbkfit.model
import gbkfit.params
import gbkfit.params.params
import gbkfit.params.descs
import gbkfit.params.utils
from gbkfit.utils import iterutils
from . import _detail


log = logging.getLogger(__name__)


# Use this object to load and dump yaml
yaml = ruamel.yaml.YAML()

# This is needed for dumping ordered dicts
ruamel.yaml.add_representer(dict, lambda self, data: self.represent_mapping(
    'tag:yaml.org,2002:map', data.items()))


def _prepare_params(info, descs):
    parameters = info['parameters']
    keys, values = gbkfit.params.utils.parse_param_keys(parameters, descs)[:2]
    parameters = dict(zip(keys, values))
    recovery_failed = []
    recovery_succeed = []
    for key, val in parameters.items():
        if iterutils.is_mapping(val):
            val = {k.lstrip('*'): v for k, v in val.items()}
            if 'val' in val:
                parameters[key] = val['val']
                recovery_succeed.append(key)
            else:
                recovery_failed.append(key)
    if recovery_succeed:
        log.info(
            f"successfully recovered values "
            f"for the following parameter keys: {recovery_succeed}")
    if recovery_failed:
        raise RuntimeError(
            f"failed to recover values "
            f"for the following parameter keys: {recovery_failed}")
    info['parameters'] = parameters
    return info


def eval_(config, perf=None):

    #
    # Read configuration file and
    # perform all necessary validation/preparation
    #

    log.info(f"reading configuration file: '{config}'...")

    try:
        cfg = yaml.load(open(config))
    except Exception as e:
        raise RuntimeError(
            "error while reading configuration file; "
            "see preceding exception for additional information") from e

    cfg = _detail.prepare_config(
        cfg,
        ('drivers', 'dmodels', 'gmodels', 'params'),
        ('datasets', 'pdescs'))

    #
    # Setup all the components described in the configuration
    #

    log.info("setting up drivers...")
    drivers = gbkfit.driver.driver_parser.load(cfg['drivers'])

    datasets = None
    if 'datasets' in cfg:
        log.info("setting up datasets...")
        datasets = gbkfit.dataset.dataset_parser.load(cfg['datasets'])

    log.info("setting up dmodels...")
    dmodels = gbkfit.model.dmodel_parser.load(cfg['dmodels'], dataset=datasets)

    log.info("setting up gmodels...")
    gmodels = gbkfit.model.gmodel_parser.load(cfg['gmodels'])

    log.info("setting up models...")
    models = gbkfit.model.make_model_group_from_cmp(dmodels, gmodels, drivers)

    pdescs = None
    if 'pdescs' in cfg:
        log.info("setting up pdescs...")
        pdescs = gbkfit.params.descs.load_desc_dicts(cfg['pdescs'])
    pdescs = gbkfit.params.descs.merge_desc_dicts(models.pdescs(), pdescs)

    log.info("setting up params...")
    cfg['params'] = _prepare_params(cfg['params'], pdescs)
    #exit()
    params = gbkfit.params.params.EvalParams.load(cfg['params'], pdescs)

    #
    # Calculate model parameters
    #

    log.info("calculating model parameters...")

    eparams = {ename: None for ename in params.expressions().enames()}
    params = params.expressions().evaluate({}, eparams)

    params_info = _detail.nativify(dict(
        params=params,
        eparams=eparams))
    filename = 'gbkfit_result_params'
    json.dump(params_info, open(f'{filename}.json', 'w+'), indent=2)
    yaml.dump(params_info, open(f'{filename}.yaml', 'w+'))

    #
    # Evaluate models
    #

    log.info("evaluating model...")

    extras = []
    output = models.evaluate_h(params, extras)

    def save_model(file, data):
        if data is not None:
            hdu = fits.PrimaryHDU(data)
            hdulist = fits.HDUList([hdu])
            hdulist.writeto(file, overwrite=True)

    log.info("writing model to the filesystem...")
    for i in range(len(output)):
        prefix = 'model' + f'_{i}' * bool(i)
        for key, value in output[i].items():
            save_model(f'{prefix}_{key}_d.fits', value.get('d'))
            save_model(f'{prefix}_{key}_m.fits', value.get('m'))
            save_model(f'{prefix}_{key}_w.fits', value.get('w'))
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
            models.evaluate_h(params)
            t2 = time.time_ns()
            t_ms = (t2 - t1) / 1000000
            times.append(t_ms)
            log.info(f"evaluation {i}: {t_ms} ms")
        log.info("calculating performance test statistics...")
        time_stats = dict(_detail.nativify(dict(
            min=np.round(np.min(times), 2),
            max=np.round(np.max(times), 2),
            mean=np.round(np.mean(times), 2),
            median=np.round(np.median(times), 2),
            stddev=np.round(np.std(times), 2),
            mad=np.round(stats.median_absolute_deviation(times), 2))))
        log.info(', '.join(f'{k}: {v} ms' for k, v in time_stats.items()))
        time_stats = dict(unit='milliseconds', **time_stats)
        filename = 'gbkfit_result_time'
        json.dump(time_stats, open(f'{filename}.json', 'w+'), indent=2)
        yaml.dump(time_stats, open(f'{filename}.yaml', 'w+'))
