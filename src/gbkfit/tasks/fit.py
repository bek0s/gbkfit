
import json
import logging
import time

import ruamel.yaml as yaml

import gbkfit
import gbkfit.broker
import gbkfit.dataset
import gbkfit.dmodel
import gbkfit.driver
import gbkfit.fitter
import gbkfit.gmodel
import gbkfit.model
import gbkfit.params
import gbkfit.params.descs
from . import _detail


log = logging.getLogger(__name__)


def _prepare_config(config):

    _detail.prepare_config_require_sections(
        config, ['datasets', 'dmodels', 'gmodels', 'fitter', 'objectives', 'params'])

    _detail.prepare_config_listify_sections(
        config, ['brokers', 'drivers', 'datasets', 'dmodels', 'gmodels', 'objectives'])

    _detail.prepare_config_check_sections_length(
        config, ['brokers', 'drivers', 'datasets', 'dmodels', 'gmodels', 'objectives'])


def _prepare_params(info, descs):

    # Remove all param info pairs with invalid keys
    info_keys, info_values, _, _ = \
        gbkfit.params.parse_param_keys(info, descs)
    info = dict(zip(info_keys, info_values))

    param_exprs = {}
    param_infos = {}
    for key, value in info.items():

        if not isinstance(value, dict):
            param_exprs[key] = value
        elif 'expr' in value:
            param_exprs[key] = value['expr']
        else:
            param_infos[key] = value

    # ...
    gbkfit.params.parse_param_exprs(param_exprs, descs)
    foo = gbkfit.params.parse_param_fit_info(param_infos, descs)

    return param_exprs, foo


def fit(config):

    log.info("Initializing gbkfit...")
    gbkfit.init()

    log.info(f"Reading configuration file: '{config}'...")
    config = yaml.YAML().load(open(config))
    _prepare_config(config)

    #
    # Setup optional stuff
    #

    brokers = None
    if config.get('brokers'):
        log.info("Setting up brokers...")
        brokers = gbkfit.broker.parser.load_many(config['brokers'])

    drivers = None
    if config.get('drivers'):
        log.info("Setting up drivers...")
        drivers = gbkfit.driver.parser.load_many(config['drivers'])

    pdescs = None
    if config.get('pdescs'):
        log.info("Setting up pdescs...")
        pdesc_info = _detail.prepare_config_pdescs(config['pdescs'])
        pdesc_keys = pdesc_info.keys()
        pdesc_vals = pdesc_info.values()
        pdesc_list = gbkfit.params.descs.parser.load_many(pdesc_vals)
        pdescs = dict(zip(pdesc_keys, pdesc_list))
    """
    objectives = None
    if config.get('objectives'):
        log.info("Setting up objectives...")
        objectives = gbkfit.fitting.objective.load(config['objectives'])
    """

    #
    # Setup required stuff
    #

    log.info("Setting up datasets...")
    datasets = gbkfit.dataset.parser.load_many(config['datasets'])

    log.info("Setting up dmodels...")
    dmodels = gbkfit.dmodel.parser.load_many(config['dmodels'], datasets)

    log.info("Setting up gmodels...")
    gmodels = gbkfit.gmodel.parser.load_many(config['gmodels'])

    log.info("Setting up models...")
    models, param_descs, param_mappings = _detail.make_models(
        dmodels, gmodels, drivers, brokers, pdescs)

    log.info("Setting up params...")
    param_exprs, param_infos = _prepare_params(config['params'], param_descs)

    log.info("Setting up fitter...")
    fitter = gbkfit.fitter.parser.load_one(config['fitter'])

    exit()
    objectives = [fitter.FOO(dataset, model) for datasets, model in zip(datasets, models)]

    fitter.fit(objectives, objective_weights, descs, exprs, param_info, param_extra)

    fitter.fit((objective), weights)
    fitter.fit((data, model), weights, param_descs, param_exprs, param_infos, param_extra)

    fitter.fit(dict(data=data, model=model), (1,))
    fitter.fit(None, None,)

    exit()

    log.info("Model-fitting started")
    t1 = time.time_ns()

    # fit
    descs = model.descs() + more_descs
    pgroup = ParamGroup(descs, exprs)
    fitter = FitterDynesty()
    params = fitter.parse_params(json['params'], json['params_extra'])
    fitter.fit(data, model, params)

    model = Model()

    problem = Problem(data, model, pdescs, pexprs)

    class ParamFitInfo:
        pass

    class ParamInfoV:
        pass

    class ParamInfoF:
        pass

    result = fitter.fit(datasets, model, param_infos)
    """
    from gbkfit.fitters.dynesty.dynesty import FitterDynestyStaticNestedSampling
    fitter = FitterDynestyStaticNestedSampling(foo='bar')
    fitter.fit(datasets, model, param_infos)
    """

    t2 = time.time_ns()
    t_ms = (t2 - t1) // 1000000
    log.info("Model-fitting completed.")
    log.info(f"Elapsed time: {t_ms} ms.")


import abc
import typing


class FitterParam(abc.ABC):
    @classmethod
    def load(cls, info):
        return cls()

    def __init__(self, **kwargs):
        self._kwargs = kwargs.copy()


class FitterParamDict(dict):

    def __init__(self, param_dict):
        super().__init__(param_dict)

    def get_dict(self, attribs=None):
        dict_ = dict()
        for param in self:
            for attrib in attribs:
                param.get_attribs().get(attrib, None)
        return dict_

    def get_list(self, attribs=None):
        pass


class FitterScipyLeastSquaresParam(FitterParam):
    def __init__(self, minimum=None, maximum=None):
        super().__init__(**dict(minimum=minimum, maximum=maximum))


class FitterScipyLeastSquaresParamDict(FitterParamDict):
    pass





class FitterParamPygmoPSO(FitterParam):
    def __init__(self, minimum=None, maximum=None):
        super().__init__(**dict(minimum=minimum, maximum=maximum))


class FitterParamDynesty(FitterParam):
    def __init__(self, prior):
        super().__init__(**dict(prior=prior))

