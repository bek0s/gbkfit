
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
from . import _detail


log = logging.getLogger(__name__)


def _prepare_config(config):

    _detail.require_config_sections(
        config, ['datasets', 'dmodels', 'gmodels', 'fitter', 'params'])

    _detail.listify_config_sections(
        config, ['brokers', 'drivers', 'datasets', 'dmodels', 'gmodels'])

    _detail.check_config_sections_length(
        config, ['datasets', 'dmodels', 'gmodels'])


def _prepare_params(params):
    param_infos = {}
    param_exprs = {}
    for key, value in params.items():
        if isinstance(value, dict):
            param_infos[key] = value
        else:
            param_exprs[key] = value
    return param_infos, param_exprs


def fit(config):

    log.info("Initializing gbkfit...")
    gbkfit.init()

    log.info(f"Reading configuration file: '{config}'...")
    config = yaml.YAML().load(open(config))
    _prepare_config(config)

    brokers = None
    if config.get('brokers'):
        log.info("Setting up brokers...")
        brokers = gbkfit.broker.parser.load_many(config['brokers'])

    drivers = None
    if config.get('drivers'):
        log.info("Setting up drivers...")
        drivers = gbkfit.driver.parser.load_many(config['drivers'])

    log.info("Setting up datasets...")
    datasets = gbkfit.dataset.parser.load_many(config['datasets'])

    log.info("Setting up dmodels...")
    dmodels = gbkfit.dmodel.parser.load_many(config['dmodels'], datasets)

    log.info("Setting up gmodels...")
    gmodels = gbkfit.gmodel.parser.load_many(config['gmodels'])

    log.info("Setting up model...")
    model = gbkfit.model.Model(dmodels, gmodels, drivers, brokers)

    log.info("Setting up params...")
    param_infos, param_exprs = _prepare_params(config['params'])
    param_info = gbkfit.params.parse_param_fit_info(param_infos, model.get_param_descs())

    print(param_info)

    exit()
    model.set_param_exprs(param_exprs)

    log.info("Setting up fitter...")
    fitter = gbkfit.fitter.parser.load_one(config['fitter'])

    log.info("Model-fitting started")
    t1 = time.time_ns()

    # eval
    model = Model()
    outputs = model.evaluate(params)

    # eval from json expressions
    model = Model()
    params = ParamGroup(model.descs() + descs, exprs).evaluate(json_params)
    outputs = model.evaluate(params)

    # fit
    model = Model()
    lhood = LikelihoodGauss(data, model, descs, exprs)

    lhood1 = LikelihoodGauss(data, model)
    lhood2 = LikelihoodGauss(data, model)
    lhood3 = LikelihoodJoint(lhood1, lhood2)
    result = fitter.fit(lhood3, descs, exprs, params)




    fitter.fit(data, model, descs, exprs, params)

    class ParamInterpreter:
        pass

    model = foo()
    model.add_parameter(gbkfit.params.ParamScalarDesc('foo'))
    model.add_parameters(gbkfit.params.ParamScalarDesc('bar'))
    model.set_expressions(existing, derived)

    descs = model.descs() + more_descs

    pgroup = ParamGroup(descs, exprs)
    fitter = FitterDynesty()
    params = fitter.parse_params(json['params'], json['params_extra'])
    fitter.fit(data, model, params)



    model = Model()

    problem = Problem(data, model, pdescs, pexprs)

    fitter.fit(problem, params)

    model = Model()
    pgroup = ParamGroup(descs, exprs)
    model.evaluate(pgroup.evaluate(values))

    pgroup = ParamGroup()
    pgroup.set(descs=None, exprs=None)
    pgroup.evaluate()





    class ParamGroup:

        def __init__(self, descs, exprs):
            pass

        def add_descs(self):
            pass

        def add_exprs(self):
            pass

        def get_descs(self):
            pass

        def get_exprs(self):
            pass

        def evaluate(
                self, values,
                out_params, out_eparams, out_params_free, out_params_fixed):
            # return params, eparams, eparams_free, eparams_fixed
            pass



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


#params = fitter.parse_params()

#fitter.explore(likelihood, params)


import typing


class Base:
    def foo(self, x: typing.Mapping):
        pass


class Child:
    def foo(self, x: typing.List):
        pass
