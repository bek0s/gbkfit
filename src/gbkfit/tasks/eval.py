import json
import logging
import os
from typing import Literal

import pandas as pd
import ruamel.yaml

import gbkfit.dataset
import gbkfit.driver
import gbkfit.model
import gbkfit.objective
import gbkfit.params
from gbkfit.utils import numutils
from gbkfit.utils.parseutils import config_path
from . import _detail


_log = logging.getLogger(__name__)


# Use this object to load and dump yaml
yaml = ruamel.yaml.YAML()

# This is needed for dumping dicts with correct order
ruamel.yaml.add_representer(dict, lambda self, data: self.represent_mapping(
    'tag:yaml.org,2002:map', data.items()))


def eval_(
        mode: Literal['model', 'objective'],
        config: str,
        profile_iters: int,
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
    required_sections = ('models', 'observations', 'params')
    optional_sections = ('pdescs',)
    if mode not in ('model', 'objective'):
        raise RuntimeError("impossible")
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
    # - objective, pdescs, and params configurations are dicts
    #

    _log.info("setting up models and observations...")
    group = _detail.load_observation_group(cfg)

    objective = None
    if mode == 'objective':
        _log.info("setting up objective...")
        objective = gbkfit.objective.Objective(group)

    _log.info("setting up pdescs...")
    pdescs = objective.pdescs() \
        if objective is not None else group.pdescs()
    if 'pdescs' in cfg:
        with config_path('pdescs'):
            user_pdescs = gbkfit.params.load_pdescs_dict(cfg['pdescs'])
        pdescs = _detail.merge_pdescs(pdescs, user_pdescs)

    _log.info("setting up params...")
    constants = objective.constants() \
        if objective is not None else group.constants()
    with config_path('params'):
        params = gbkfit.params.param_space_parser.load(
            cfg['params'], pdescs=pdescs, constants=constants)

    #
    # Calculate model parameters
    #

    _log.info("calculating model parameters...")

    param_values = params.evaluate()
    params_info = numutils.nativify(dict(
        params=param_values,
        eparams=params.element_values(param_values)))
    filename = os.path.join(output_dir, 'gbkfit_eval_params')
    _detail.dump_dict(json, yaml, params_info, filename)

    #
    # Evaluate objective
    #

    _log.info("evaluating...")
    outputs = _detail.evaluation_outputs(
        group, objective, param_values, mode == 'model', mode == 'objective')
    if objective is not None:
        _log.info(f"chi-square: {objective.chi2(param_values)}")

    # #
    # # Calculate outputs statistics
    # #
    #
    # _log.info("calculating statistics for outputs...")
    #
    # outputs_stats = {}
    # for filename, data in outputs.items():
    #     if data is not None:
    #         sum_ = np.nansum(data)
    #         min_ = np.nanmin(data)
    #         max_ = np.nanmax(data)
    #         mean = np.nanmean(data)
    #         stddev = np.nanstd(data)
    #         median = np.nanmedian(data)
    #         outputs_stats.update({filename: dict(
    #             sum=sum_, min=min_, max=max_, mean=mean, stddev=stddev,
    #             median=median)})
    #
    # filename = os.path.join(output_dir, 'gbkfit_eval_outputs')
    # outputs_stats = numutils.nativify(outputs_stats)
    # _detail.dump_dict(json, yaml, outputs_stats, filename)

    #
    # Store outputs
    #

    _log.info("storing outputs to the filesystem...")

    _detail.write_outputs(
        json, yaml, output_dir, outputs, 'gbkfit_eval_extra')

    #
    # Run performance tests
    #

    if profile_iters > 0:
        _log.info("running performance test...")
        # Time the profiled evaluations only (the first evaluation, above,
        # also warmed up the drivers)
        group.timers().clear()
        for i in range(profile_iters):
            if mode == 'model':
                group.model_d(param_values)
            if mode == 'objective':
                objective.chi2(param_values)
        _log.info("calculating timing statistics...")
        time_stats = numutils.nativify({
            name: stats.to_dict()
            for name, stats in group.timers().stats().items()})
        _log.info(pd.DataFrame.from_dict(time_stats, orient='index'))
        filename = os.path.join(output_dir, 'gbkfit_eval_timings')
        _detail.dump_dict(json, yaml, time_stats, filename)

