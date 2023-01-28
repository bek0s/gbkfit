
import logging
import os
import time

import ruamel.yaml

import gbkfit
import gbkfit.dataset
import gbkfit.driver
import gbkfit.fitting
import gbkfit.model
import gbkfit.objective
import gbkfit.params
import gbkfit.params.pdescs
import gbkfit.params.parsers
from gbkfit.params import parsers as param_parsers
from gbkfit.utils import miscutils
from . import _detail


_log = logging.getLogger(__name__)


# Use this object to load and dump yaml
yaml = ruamel.yaml.YAML()

# This is needed for dumping dicts with correct order
ruamel.yaml.add_representer(dict, lambda self, data: self.represent_mapping(
    'tag:yaml.org,2002:map', data.items()))


def _prepare_params(info, pdescs):
    # Prepare the supplied parameters. This will:
    # - Ensure the parameter keys are valid
    # - Explode parameter values that are dicts and can be exploded
    parameters = param_parsers.prepare_param_info(
        info.get('parameters'), pdescs)

    # Update parameter info and return it
    return info | dict(parameters=parameters)


def fit(config):

    #
    # Read configuration file and
    # perform all necessary validation/preparation
    #

    _log.info(f"reading configuration file: '{config}'...")

    try:
        cfg = yaml.load(open(config))
    except Exception as e:
        raise RuntimeError(
            "error while reading configuration file; "
            "see preceding exception for additional information") from e

    required_sections = (
        'drivers', 'datasets', 'dmodels', 'gmodels', 'params', 'fitter')
    optional_sections = ('objective', 'pdescs')
    cfg = _detail.prepare_config(cfg, required_sections, optional_sections)

    #
    # Setup all the components described in the configuration
    #

    _log.info("setting up drivers...")
    drivers = gbkfit.driver.driver_parser.load(cfg['drivers'])

    _log.info("setting up datasets...")
    datasets = gbkfit.dataset.dataset_parser.load(cfg['datasets'])

    _log.info("setting up dmodels...")
    dmodels = gbkfit.model.dmodel_parser.load(cfg['dmodels'], dataset=datasets)

    _log.info("setting up gmodels...")
    gmodels = gbkfit.model.gmodel_parser.load(cfg['gmodels'])

    _log.info("setting up objective...")
    objective = gbkfit.objective.goodness_objective_parser.load(
        cfg.get('objective', {}), datasets, drivers, dmodels, gmodels)

    _log.info("setting up fitter...")
    fitter = gbkfit.fitting.fitter_parser.load(cfg['fitter'])

    # exit(fitter.dump())
    # yaml.dump(fitter.dump(), open(f'fitter.yaml', 'w+'))

    pdescs = None
    if 'pdescs' in cfg:
        _log.info("setting up pdescs...")
        pdescs = gbkfit.params.pdescs.load_pdescs_dict(cfg['pdescs'])
    pdescs = _detail.merge_pdescs(objective.pdescs(), pdescs)

    _log.info("setting up params...")
    cfg['params'] = _prepare_params(cfg['params'], pdescs)

    print(cfg['params'])

    params = fitter.load_params(cfg['params'], pdescs)

    #
    # Perform fit
    #

    _log.info("model-fitting started")
    t1 = time.time_ns()
    result = fitter.fit(objective, params)
    t2 = time.time_ns()
    t_ms = (t2 - t1) // 1_000_000
    _log.info(f"model-fitting completed (elapsed time: {t_ms} ms)")

    output_dir = os.path.abspath(miscutils.make_unique_path('out'))
    _log.info(f"saving result under '{output_dir}'...")
    gbkfit.fitting.result.dump_result(output_dir, result)
    print(result.summary())
