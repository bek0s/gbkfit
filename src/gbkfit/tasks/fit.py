
import json
import logging
import os
import time

import ruamel.yaml

import gbkfit
import gbkfit.dataset
import gbkfit.driver
import gbkfit.fitting
import gbkfit.model
import gbkfit.params
import gbkfit.params.descs
import gbkfit.params.utils
from gbkfit.utils import miscutils
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

    return 1


def fit(config):

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
        ('drivers', 'datasets', 'dmodels', 'gmodels', 'params', 'fitter'),
        ('objective', 'pdescs'))

    #
    # Setup all the components described in the configuration
    #

    log.info("setting up drivers...")
    drivers = gbkfit.driver.driver_parser.load(cfg['drivers'])

    log.info("setting up datasets...")
    datasets = gbkfit.dataset.dataset_parser.load(cfg['datasets'])

    log.info("setting up dmodels...")
    dmodels = gbkfit.model.dmodel_parser.load(cfg['dmodels'], dataset=datasets)

    log.info("setting up gmodels...")
    gmodels = gbkfit.model.gmodel_parser.load(cfg['gmodels'])

    log.info("setting up model...")
    models = gbkfit.model.make_model_group_from_cmp(dmodels, gmodels, drivers)

    log.info("setting up fitter...")
    fitter = gbkfit.fitting.fitter_parser.load(cfg['fitter'])

    log.info("setting up objective...")
    objective = gbkfit.fitting.objective_parser.load(
        cfg.get('objective', {}), datasets=datasets, models=models)

    pdescs = None
    if 'pdescs' in cfg:
        log.info("setting up pdescs...")
        pdescs = gbkfit.params.descs.load_descs(cfg['pdescs'])
    pdescs = gbkfit.params.descs.merge_descs(objective.pdescs(), pdescs)

    log.info("setting up params...")

    cfg['params'] = _prepare_params(cfg['params'], pdescs)

    exit()
    params = fitter.load_params(cfg['params'], pdescs)

    #
    # Perform fit
    #
    #exit()
    log.info("model-fitting started")
    t1 = time.time_ns()
    result = fitter.fit(objective, params)
    t2 = time.time_ns()
    t_ms = (t2 - t1) // 1000000
    log.info(f"model-fitting completed (elapsed time: {t_ms} ms)")

    output_dir = os.path.abspath(miscutils.make_unique_path('out'))
    log.info(f"saving result under '{output_dir}'...")
    gbkfit.fitting.result.dump_result(output_dir, result)
    print(result.summary())
