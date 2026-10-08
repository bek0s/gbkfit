import copy
import json
import logging
import os
from typing import Literal

import astropy.io.fits as fits
import numpy as np
import pandas as pd
import ruamel.yaml

import gbkfit.dataset
import gbkfit.driver
import gbkfit.model
import gbkfit.objective
import gbkfit.params
from gbkfit.utils import fitsutils, iterutils, timeutils
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
    required_sections = ('models', 'params')
    optional_sections = ('pdescs',)
    if mode == 'model':
        optional_sections += ('datasets',)
    elif mode == 'objective':
        required_sections += ('datasets',)
        optional_sections += ('objective',)
    else:
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
    # - datasets and models configurations are lists
    # - objective, pdescs, and params configurations are dicts
    #

    datasets = None
    if 'datasets' in cfg:
        _log.info("setting up datasets...")
        with config_path('datasets'):
            datasets = gbkfit.dataset.dataset_parser.load(cfg['datasets'])

    _log.info("setting up models...")
    with config_path('models'):
        models = gbkfit.model.model_parser.load(
            cfg['models'], dataset=datasets)
    model_group = gbkfit.model.ModelGroup(models)

    objective = None
    if mode == 'objective':
        _log.info("setting up objective...")
        with config_path('objective'):
            objective = gbkfit.objective.objective_parser.load(
                cfg.get('objective', {}),
                datasets=datasets, models=model_group)

    _log.info("setting up pdescs...")
    pdescs = objective.pdescs() \
        if objective is not None else model_group.pdescs()
    if 'pdescs' in cfg:
        with config_path('pdescs'):
            user_pdescs = gbkfit.params.load_pdescs_dict(cfg['pdescs'])
        pdescs = _detail.merge_pdescs(pdescs, user_pdescs)

    _log.info("setting up params...")
    constants = objective.constants() \
        if objective is not None else model_group.constants()
    with config_path('params'):
        params = gbkfit.params.evaluation_params_parser.load(
            cfg['params'], pdescs=pdescs, constants=constants)

    #
    # Calculate model parameters
    #

    _log.info("calculating model parameters...")

    exploded_param_values = {}
    param_values = params.evaluate(out_exploded_params=exploded_param_values)
    params_info = iterutils.nativify(dict(
        params=param_values,
        eparams=exploded_param_values))
    filename = os.path.join(output_dir, 'gbkfit_eval_params')
    _detail.dump_dict(json, yaml, params_info, filename)

    #
    # Evaluate objective
    #

    # Always evaluate model
    model_extra = {}
    model_data = []
    if mode == 'model':
        model_data = model_group.model_h(param_values, model_extra)

    resid_u_extra = {}
    resid_u_data = []
    resid_w_extra = {}
    resid_w_data = []
    if mode == 'objective':
        # The objective reuses its residual buffers on every call,
        # so keep copies of the unweighted and weighted residuals
        resid_u_data = copy.deepcopy(objective.residual_nddata_h(
            param_values, False, resid_u_extra))
        resid_w_data = copy.deepcopy(objective.residual_nddata_h(
            param_values, True, resid_w_extra))
        residual_sum = objective.residual_scalar(param_values, True)
        _log.info(f"sum of squared residuals: {residual_sum}")

    #
    # Gather objective outputs
    #

    _log.info("gathering outputs...")

    # The outputs by file name: the data, and the grid of the dmodel or
    # dataset it is on (None for the extra outputs, which are not on
    # the grid of the data)
    outputs = {}
    model_prefix = 'model'
    resid_u_prefix = 'residual'
    resid_w_prefix = 'wresidual'

    # Store model
    for i, data_i in enumerate(model_data):
        # prefix_i = model_prefix + f'_{i}' * bool(model.nitems() > 0)
        prefix_i = model_prefix + f'_{i}'
        dmodel = model_group.models()[i].dmodel()
        for key, value in data_i.items():
            outputs |= {
                f'{prefix_i}_{key}_d.fits': (value.get('d'), dmodel),
                f'{prefix_i}_{key}_m.fits': (value.get('m'), dmodel),
                f'{prefix_i}_{key}_w.fits': (value.get('w'), dmodel)}
    # Store residual (if available)
    for resid_data, prefix in [
            (resid_u_data, resid_u_prefix), (resid_w_data, resid_w_prefix)]:
        for i, data_i in enumerate(resid_data):
            prefix_i = prefix + f'_{i}' * bool(objective.nitems() > 1)
            dataset = objective.datasets()[i]
            for key, value in data_i.items():
                outputs |= {f'{prefix_i}_{key}_d.fits': (value, dataset)}
    # Store model and residual extra (if available)
    for extra, prefix in [
            (model_extra, model_prefix),
            (resid_u_extra, resid_u_prefix),
            (resid_w_extra, resid_w_prefix)]:
        for key, value in extra.items():
            outputs |= {f'{prefix}_extra_{key}.fits': (value, None)}

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
    # outputs_stats = iterutils.nativify(outputs_stats)
    # _detail.dump_dict(json, yaml, outputs_stats, filename)

    #
    # Store outputs
    #

    _log.info("storing outputs to the filesystem...")

    for filename, (data, grid) in outputs.items():
        if not isinstance(data, np.ndarray):
            continue
        filename = os.path.join(output_dir, filename)
        if grid is None:
            fits.writeto(filename, data, overwrite=True)
        else:
            coords = fitsutils.Coords(
                grid.step(), grid.rpix(), grid.rval(), grid.rota())
            fitsutils.write_data(
                filename, data, coords, grid.spectral_axis(), overwrite=True)

    #
    # Run performance tests
    #

    if profile_iters > 0:
        _log.info("running performance test...")
        for i in range(profile_iters):
            if mode == 'model':
                model_group.model_d(param_values)
            if mode == 'objective':
                objective.log_likelihood(param_values)
                objective.residual_scalar(param_values, squared=True)
        _log.info("calculating timing statistics...")
        time_stats = iterutils.nativify(timeutils.get_time_stats())
        _log.info(pd.DataFrame.from_dict(time_stats, orient='index'))
        filename = os.path.join(output_dir, 'gbkfit_eval_timings')
        _detail.dump_dict(json, yaml, time_stats, filename)
