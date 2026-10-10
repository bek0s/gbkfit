import json
import logging
import os
from typing import Any, Literal

import ruamel.yaml

import gbkfit.dataset
import gbkfit.driver
import gbkfit.fitting
import gbkfit.model
import gbkfit.objective
import gbkfit.params
from gbkfit.utils import numutils
from gbkfit.utils.parseutils import ConfigError, config_path
from . import _detail


_log = logging.getLogger(__name__)


# Use this object to load and dump yaml
yaml = ruamel.yaml.YAML()

# This is needed for dumping dicts with correct order
ruamel.yaml.add_representer(dict, lambda self, data: self.represent_mapping(
    'tag:yaml.org,2002:map', data.items()))


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
    required_sections = ('models', 'observations', 'params', 'fitter')
    optional_sections = ('pdescs',)
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
    # - models and observations configurations are lists
    # - pdescs, params and fitter configurations are dicts
    #


    _log.info("setting up models and observations...")
    group = _detail.load_observation_group(cfg)

    _log.info("setting up objective...")
    objective = gbkfit.objective.Objective(group)

    _log.info("setting up fitter...")
    with config_path('fitter'):
        fitter = gbkfit.fitting.fitter_parser.load(cfg['fitter'])

    pdescs = objective.pdescs()
    if 'pdescs' in cfg:
        _log.info("setting up pdescs...")
        with config_path('pdescs'):
            user_pdescs = gbkfit.params.load_pdescs_dict(cfg['pdescs'])
        pdescs = _detail.merge_pdescs(pdescs, user_pdescs)

    _log.info("setting up params...")
    with config_path('params'):
        params = gbkfit.params.param_space_parser.load(
            cfg['params'], pdescs=pdescs, constants=objective.constants())
    # Check the free properties for the fitter before the fit starts
    with config_path('params', 'properties'):
        fitter.load_fit_params(params)

    #
    # Perform fit
    #

    _log.info("model-fitting started")
    result = fitter.fit(objective, params)
    _log.info(f"model-fitting completed (elapsed time: {result.time:.1f} s)")

    _log.info(f"saving result under '{output_dir}'...")
    result.dump(output_dir)
    outputs = _detail.evaluation_outputs(
        group, objective, result.best().values, True, True)
    _detail.write_outputs(
        json, yaml, output_dir, outputs, 'gbkfit_fit_extra')
    _write_restart(cfg, params, result, output_dir)
    _log.info(f"result:\n{result.summary()}")


def _write_restart(
        cfg: dict[str, Any],
        params: gbkfit.params.ParamSpace,
        result: gbkfit.fitting.FitResult,
        output_dir: str
) -> None:
    """
    Write restart.yaml: the configuration with the free parameters
    starting at the best fit, to fit again from there (e.g. with another
    fitter). Its paths are those of the configuration (relative to the
    directory the fit ran in).
    """
    try:
        restart = params.with_values(result.best().elements).dump()
    except ConfigError as e:
        _log.warning(f"restart.yaml is not written: {e}")
        return
    with open(os.path.join(output_dir, 'restart.yaml'), 'w') as f:
        yaml.dump(dict(cfg) | dict(params=numutils.nativify(restart)), f)
