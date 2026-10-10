"""
The eval task: evaluate the models of a configuration, or its objective
(the residuals against the data), and write the outputs.
"""

import logging
import os
from typing import Literal

import pandas as pd

import gbkfit.objective
from gbkfit.utils import numutils, parseutils
from . import _detail


_log = logging.getLogger(__name__)


def eval_(
        mode: Literal['model', 'objective'],
        config: str,
        profile_iters: int,
        output_dir: str,
        output_dir_mode: Literal['terminate', 'overwrite', 'unique']
) -> None:
    """
    Evaluate the models of a configuration at its parameters (their
    fixed values, and the 'value' of the free ones), or its objective,
    and write the outputs: the parameters (gbkfit_eval_params), the
    models or the residuals (FITS), and their extra outputs.

    Parameters
    ----------
    mode : {'model', 'objective'}
        What to evaluate.
    config : str
        The configuration file (YAML or JSON); its relative paths are
        relative to its directory (or to its section 'root').
    profile_iters : int
        The number of evaluations to time (gbkfit_eval_timings).
    output_dir : str
        The output directory.
    output_dir_mode : {'terminate', 'overwrite', 'unique'}
        What to do if the output directory exists (see make_output_dir).

    Raises
    ------
    ConfigError
        If the configuration is invalid.
    """
    _log.info(f"reading configuration from file: {config}...")
    cfg, root = _detail.read_config(
        config, ('models', 'observations', 'params'), ('pdescs',))
    output_dir = _detail.make_output_dir(output_dir, output_dir_mode)
    _log.info(f"output will be stored under directory: {output_dir}")

    with parseutils.paths_relative_to(root):
        _log.info("setting up models and observations...")
        group = _detail.load_observation_group(cfg)
        objective = None
        if mode == 'objective':
            _log.info("setting up objective...")
            objective = gbkfit.objective.Objective(group)
        _log.info("setting up params...")
        params = _detail.load_params(cfg, group.pdescs(), group.constants())

    _log.info("calculating model parameters...")
    param_values = params.evaluate()
    _detail.dump_dict(
        numutils.nativify(dict(
            params=param_values,
            eparams=params.element_values(param_values))),
        os.path.join(output_dir, 'gbkfit_eval_params'))

    _log.info("evaluating...")
    outputs = _detail.evaluation_outputs(
        group, objective, param_values, mode == 'model', mode == 'objective')
    if objective is not None:
        _log.info(f"chi-square: {objective.chi2(param_values)}")
    _log.info("storing outputs to the filesystem...")
    _detail.write_outputs(output_dir, outputs, 'gbkfit_eval_extra')

    if profile_iters > 0:
        _log.info("running performance test...")
        # Time the profiled evaluations only (the first evaluation, above,
        # also warmed up the drivers)
        group.timers().clear()
        for _ in range(profile_iters):
            if objective is None:
                group.model_d(param_values)
            else:
                objective.chi2(param_values)
        time_stats = numutils.nativify({
            name: stats.to_dict()
            for name, stats in group.timers().stats().items()})
        _log.info(pd.DataFrame.from_dict(time_stats, orient='index'))
        _detail.dump_dict(
            time_stats, os.path.join(output_dir, 'gbkfit_eval_timings'))
