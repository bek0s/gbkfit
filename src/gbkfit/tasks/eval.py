
import json
import logging
import time

import astropy.io.fits as fits
import numpy as np
import ruamel.yaml as yaml
import scipy.stats as stats

import gbkfit
import gbkfit.dataset
import gbkfit.driver
import gbkfit.model
import gbkfit.params
import gbkfit.params.params
import gbkfit.params.descs
import gbkfit.params.utils
from . import _detail

log = logging.getLogger(__name__)


# This is needed for dumping ordered dicts
yaml.add_representer(dict, lambda self, data: self.represent_mapping(
    'tag:yaml.org,2002:map', data.items()))


def _patch_parameters(info, descs):
    keys, values = gbkfit.params.utils.parse_param_keys(info, descs)[:2]
    info = dict(zip(keys, values))
    recovery_failed = []
    recovery_succeed = []
    for key, val in info.items():
        if isinstance(val, dict):
            if 'val' in val:
                info[key] = val['val']
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
    return info


def eval_(config, perf=None):

    #
    # Read configuration file and
    # perform all necessary validation/patching/preparation
    #

    log.info(f"reading configuration file: '{config}'...")
    cfg = _detail.prepare_config(
        yaml.YAML().load(open(config)),
        ('drivers', 'dmodels', 'gmodels', 'params'),
        ('datasets', 'pdescs'))

    #
    # Setup all the components described in the configuration
    #

    datasets = None
    if 'datasets' in cfg:
        log.info("setting up datasets...")
        datasets = gbkfit.dataset.dataset_parser.load(cfg['datasets'])

    log.info("setting up drivers...")
    drivers = gbkfit.driver.driver.parser.load(cfg['drivers'])

    log.info("setting up dmodels...")
    dmodels = gbkfit.model.dmodel_parser.load(cfg['dmodels'], dataset=datasets)

    log.info("setting up gmodels...")
    gmodels = gbkfit.model.gmodel_parser.load(cfg['gmodels'])

    log.info("setting up model...")
    model = gbkfit.model.Model(dmodels, gmodels, drivers)

    pdescs = None
    if 'pdescs' in cfg:
        log.info("setting up pdescs...")
        pdescs = gbkfit.params.descs.load_descriptions(cfg['pdescs'])
    pdescs = gbkfit.params.descs.merge_descriptions(model.pdescs(), pdescs)

    log.info("setting up params...")
    cfg['params']['parameters'] = _patch_parameters(
        cfg['params']['parameters'], pdescs)
    params = gbkfit.params.params.EvalParams.load(cfg['params'], pdescs)

    #
    # Calculate model parameters
    #

    params = params.expressions().evaluate({})
    filename = 'gbkfit_result_params'
    params = _detail.nativify(params)
    json.dump(params, open(f'{filename}.json', 'w+'))
    yaml.dump(params, open(f'{filename}.yaml', 'w+'))

    #
    # Evaluate models
    #

    log.info("evaluating model...")

    extras = []
    models = model.evaluate_h(params, extras)

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
            model.evaluate_h(params)
            t2 = time.time_ns()
            t_ms = (t2 - t1) // 1000000
            times.append(t_ms)
            log.info(f"evaluation {i}: {t_ms} ms")
        log.info("calculating performance test statistics...")
        time_stats = _detail.nativify(dict(
            min=np.round(np.min(times), 1),
            max=np.round(np.max(times), 1),
            mean=np.round(np.mean(times), 1),
            median=np.round(np.median(times), 1),
            stddev=np.round(np.std(times), 1),
            mad=np.round(stats.median_absolute_deviation(times), 1)))
        log.info(', '.join(f'{k}: {v} ms' for k, v in time_stats.items()))
        filename = 'gbkfit_result_time'
        json.dump(time_stats, open(f'{filename}.json', 'w+'))
        yaml.dump(time_stats, open(f'{filename}.yaml', 'w+'))
