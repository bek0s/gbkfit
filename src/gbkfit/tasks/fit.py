import copy
import logging
import os
import time
from typing import Any, Literal

import numpy as np
import ruamel.yaml

import gbkfit.dataset
import gbkfit.driver
import gbkfit.fitting
import gbkfit.model
import gbkfit.objective
import gbkfit.params
from gbkfit.params import ParamDesc
from gbkfit.utils import miscutils
from . import _detail


_log = logging.getLogger(__name__)


# Use this object to load and dump yaml
yaml = ruamel.yaml.YAML()

# This is needed for dumping dicts with correct order
ruamel.yaml.add_representer(dict, lambda self, data: self.represent_mapping(
    'tag:yaml.org,2002:map', data.items()))


def _prepare_params(
        info: dict[str, Any],
        pdescs: dict[str, ParamDesc]
) -> dict[str, Any]:

    # Copy info for safety
    info = copy.deepcopy(info)

    # Prepare the supplied parameter properties. This will:
    # - Ensure the parameter keys are valid
    # - Explode all parameter names that are can be exploded
    parameters = gbkfit.params.parse_param_info(
        info.get('properties'), pdescs).info

    # Update parameter info and return it
    return info | dict(properties=parameters)


def fit(config: str,
        output_dir: str,
        output_dir_mode: Literal['terminate', 'overwrite', 'unique']):

    #
    # Read configuration file and
    # perform all necessary validation/preparation
    #

    config = os.path.abspath(config)
    _log.info(f"reading configuration from file: {config}...")

    try:
        cfg = yaml.load(open(config))
    except Exception as e:
        raise RuntimeError(
            f"error while reading configuration file {config}; "
            f"see reason below:\n{e}") from e

    _log.info("preparing configuration...")
    # This is not a full-fledged validation. It just tries to catch
    # and inform the user about the really obvious mistakes.
    required_sections = (
        'datasets', 'models', 'params', 'fitter')
    optional_sections = ('pdescs', 'objective')
    cfg = _detail.prepare_config(cfg, required_sections, optional_sections)

    #
    # Ensure an output directory is available for the outputs
    #

    _log.info("preparing output directory...")
    output_dir = _detail.make_output_dir(output_dir, output_dir_mode)
    _log.info(f"output will be stored under directory: {output_dir}")

    #
    # Setup all the components described in the configuration.
    # After running the configuration through _detail.prepare_config():
    # - datasets and models configurations are lists
    # - objective, pdescs, params, and fitter configurations are dicts
    #

    _log.info("setting up datasets...")
    datasets = gbkfit.dataset.dataset_parser.load(cfg['datasets'])

    _log.info("setting up model...")
    models = gbkfit.model.model_parser.load(cfg['models'], dataset=datasets)
    model_group = gbkfit.model.ModelGroup(models)

    _log.info("setting up objective...")
    objective = gbkfit.objective.objective_parser.load(
        cfg.get('objective', {}), datasets=datasets, models=model_group)

    _log.info("setting up fitter...")
    fitter = gbkfit.fitting.fitter_parser.load(cfg['fitter'])

    pdescs = None
    if 'pdescs' in cfg:
        _log.info("setting up pdescs...")
        pdescs = gbkfit.params.pdescs.load_pdescs_dict(cfg['pdescs'])
    pdescs = _detail.merge_pdescs(objective.pdescs(), pdescs)

    _log.info("setting up params...")
    cfg['params'] = _prepare_params(cfg['params'], pdescs)

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
