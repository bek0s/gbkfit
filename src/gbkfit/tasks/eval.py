
import json
import logging

import astropy.io.fits as fits
import pandas as pd
import ruamel.yaml

import gbkfit
import gbkfit.dataset
import gbkfit.driver
import gbkfit.model
import gbkfit.objective
import gbkfit.params
import gbkfit.params.params
import gbkfit.params.pdescs
from gbkfit.params import paramutils
from gbkfit.utils import iterutils
from . import _detail


_log = logging.getLogger(__name__)


# Use this object to load and dump yaml
yaml = ruamel.yaml.YAML()

# This is needed for dumping ordered dicts
ruamel.yaml.add_representer(dict, lambda self, data: self.represent_mapping(
    'tag:yaml.org,2002:map', data.items()))


def _prepare_params(info, pdescs):

    parameters = paramutils.prepare_param_info(info.get('parameters'), pdescs)

    recovery_failed = []
    recovery_succeed = {}

    def recover_value(dict_, key_, index_):
        value = dict_.get('value')
        value_id = key if index_ is None else (key_, index_)
        if 'value' in dict_:
            recovery_succeed[value_id] = value
        else:
            recovery_failed.append(value_id)
        return value

    for key, val in parameters.items():
        if iterutils.is_mapping(val):
            parameters[key] = recover_value(val, key, None)
        elif iterutils.is_sequence(val):
            parameters[key] = [recover_value(ival, key, i)
                               if iterutils.is_mapping(ival) else ival
                               for i, ival in enumerate(val)]

    # Report successful recoveries
    if recovery_succeed:
        _log.info(
            f"successfully recovered values "
            f"for the following parameter keys: {recovery_succeed}")

    # Check for errors
    if recovery_failed:
        raise RuntimeError(
            f"failed to recover values "
            f"for the following parameter keys: {recovery_failed}")

    # Update parameter info and return it
    return info | dict(parameters=parameters)


def eval_(objective_type, config, profile=None):

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

    required_sections = ('drivers', 'dmodels', 'gmodels', 'params')
    optional_sections = ('pdescs',)
    if objective_type == 'model':
        optional_sections += ('datasets',)
    if objective_type == 'goodness':
        required_sections += ('datasets',)
        optional_sections += ('objective',)
    cfg = _detail.prepare_config(cfg, required_sections, optional_sections)

    #
    # Setup all the components described in the configuration
    #

    _log.info("setting up drivers...")
    drivers = gbkfit.driver.driver_parser.load(cfg['drivers'])

    datasets = None
    if 'datasets' in cfg:
        _log.info("setting up datasets...")
        datasets = gbkfit.dataset.dataset_parser.load(cfg['datasets'])

    _log.info("setting up dmodels...")
    dmodels = gbkfit.model.dmodel_parser.load(cfg['dmodels'], dataset=datasets)

    _log.info("setting up gmodels...")
    gmodels = gbkfit.model.gmodel_parser.load(cfg['gmodels'])

    _log.info("setting up objective...")
    objective = gbkfit.objective.ObjectiveModel(drivers, dmodels, gmodels) \
        if objective_type == 'model' \
        else gbkfit.objective.goodness_objective_parser.load(
            cfg.get('objective', {}), datasets, drivers, dmodels, gmodels)

    pdescs = None
    if 'pdescs' in cfg:
        _log.info("setting up pdescs...")
        pdescs = gbkfit.params.pdescs.load_pdescs_dict(cfg['pdescs'])
    pdescs = _detail.merge_pdescs(objective.pdescs(), pdescs)

    _log.info("setting up params...")
    cfg['params'] = _prepare_params(cfg['params'], pdescs)
    params = gbkfit.params.params.evaluation_params_parser.load(
        cfg['params'], pdescs)

    #
    # Calculate model parameters
    #

    _log.info("calculating model parameters...")

    eparams = {}
    params = params.evaluate(eparams)
    params_info = _detail.nativify(dict(
        params=params,
        eparams=eparams))
    filename = 'gbkfit_result_params'
    json.dump(params_info, open(f'{filename}.json', 'w+'), indent=2)
    yaml.dump(params_info, open(f'{filename}.yaml', 'w+'))

    #
    # Evaluate objective
    #

    _log.info("evaluating objective...")

    # Always evaluate model
    model_extra = {}
    model_data = objective.model_h(params, model_extra)

    resid_u_extra = {}
    resid_u_data = []
    resid_w_extra = {}
    resid_w_data = []
    if objective_type == 'goodness':
        resid_u_data = objective.residual_nddata_h(params, False, resid_u_extra)
        resid_w_data = objective.residual_nddata_h(params, True, resid_w_extra)

    def save_model(file, data):
        if data is not None:
            hdu = fits.PrimaryHDU(data)
            hdulist = fits.HDUList([hdu])
            hdulist.writeto(file, overwrite=True)

    _log.info("writing objective output to the filesystem...")
    model_prefix = 'model'
    resid_u_prefix = 'residual'
    resid_w_prefix = 'wresidual'
    # Store model
    for i, data_i in enumerate(model_data):
        prefix_i = model_prefix + f'_{i}' * bool(objective.nitems() > 1)
        for key, value in data_i.items():
            save_model(f'{prefix_i}_{key}_d.fits', value.get('d'))
            save_model(f'{prefix_i}_{key}_m.fits', value.get('m'))
            save_model(f'{prefix_i}_{key}_w.fits', value.get('w'))
    # Store residual (if available)
    for i, data_i in enumerate(resid_u_data):
        prefix_i = resid_u_prefix + f'_{i}' * bool(objective.nitems() > 1)
        for key, value in data_i.items():
            save_model(f'{prefix_i}_{key}_d.fits', value)
    for i, data_i in enumerate(resid_w_data):
        prefix_i = resid_w_prefix + f'_{i}' * bool(objective.nitems() > 1)
        for key, value in data_i.items():
            save_model(f'{prefix_i}_{key}_d.fits', value)
    # Store model extra
    for key, value in model_extra.items():
        save_model(f'{model_prefix}_extra_{key}.fits', value)
    # Store residual extra (if available)
    for key, value in resid_u_extra.items():
        save_model(f'{resid_u_prefix}_extra_{key}.fits', value)
    for key, value in resid_w_extra.items():
        save_model(f'{resid_w_prefix}_extra_{key}.fits', value)

    #
    # Run performance tests
    #

    if profile > 0:
        _log.info("running performance test...")
        objective.time_stats_reset()
        for i in range(profile):
            if objective_type == 'model':
                objective.model_h(params)
            if objective_type == 'goodness':
                objective.residual_nddata_h(params)
        _log.info("calculating timing statistics...")
        time_stats = _detail.nativify(objective.time_stats())
        _log.info(pd.DataFrame.from_dict(time_stats, orient='index'))
        filename = 'gbkfit_result_time'
        time_stats |= dict(unit='milliseconds')
        json.dump(time_stats, open(f'{filename}.json', 'w+'), indent=2)
        yaml.dump(time_stats, open(f'{filename}.yaml', 'w+'))
