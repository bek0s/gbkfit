import abc
import copy
import numbers

from gbkfit.params.interpreter import (
    Interpreter,
    load_exprs_file,
    dump_exprs_file)
from gbkfit.params.utils import parse_param_values_strict
from gbkfit.utils import parseutils

from gbkfit.params import utils as paramutils


def load_params_info_common(cls, info):
    desc = parseutils.make_basic_desc(cls, 'params')
    opts = parseutils.parse_options_for_callable(
        info, desc, cls.__init__, fun_ignore_args=['descs'])
    if 'expressions' in opts:
        opts['expressions'] = load_exprs_file(opts['expressions'])
    return opts


def load_params_info_common_2(
        cls, info,
        ignore_args=None, rename_args=None,
        add_required=None, add_optional=None):
    desc = parseutils.make_basic_desc(cls, 'params')
    opts = parseutils.parse_options_for_callable(
        info, desc, cls.__init__,
        fun_ignore_args=ignore_args, fun_rename_args=rename_args,
        add_required=add_required, add_optional=add_optional)
    opts['value_conversions'] = paramutils.load_param_value_conversions(
        opts.get('value_conversions'))
    return opts


def dump_params_info_common_2(params, value_conversions='conversions.py'):
    info = dict(parameters=params.parameters())
    exprs_func_gen = params.interpreter().exprs_func_gen()
    exprs_func_obj = params.interpreter().exprs_func_obj()
    exprs_func_src = params.interpreter().exprs_func_src()
    if not exprs_func_gen and exprs_func_src:
        info['expressions'] = dump_exprs_file(
            exprs_file, exprs_func_obj, exprs_func_src)
    return info


def dump_params_info_common(params, exprs_file):
    info = dict(parameters=params.parameters())
    exprs_func_gen = params.interpreter().exprs_func_gen()
    exprs_func_obj = params.interpreter().exprs_func_obj()
    exprs_func_src = params.interpreter().exprs_func_src()
    if not exprs_func_gen and exprs_func_src:
        info['expressions'] = dump_exprs_file(
            exprs_file, exprs_func_obj, exprs_func_src)
    return info


class Params(abc.ABC):

    def __init__(self, descs, parameters, expressions, conversions):
        self._descs = copy.deepcopy(descs)
        self._parameters = copy.deepcopy(parameters)
        self._interpreter = Interpreter(descs, expressions, conversions)

    def descs(self):
        return self._descs


class EvaluationParams(parseutils.BasicParserSupport, Params):

    @classmethod
    def load(cls, info, descs):
        desc = parseutils.make_basic_desc(cls, 'params')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['descs'])
        opts['value_conversions'] = paramutils.load_param_value_conversions(
            opts.get('value_conversions'))
        return cls(descs, **opts)

    def dump(self, exprs_file):
        return paramutils.dump_params_info_common(self, exprs_file)

    def __init__(self, descs, parameters, value_conversions=None):
        values, exprs = parse_param_values_strict(descs, parameters, ())
        super().__init__(descs, parameters, exprs, value_conversions)

    def enames(self, tied=True, fixed=True):
        return self._interpreter.enames(free=False, tied=tied, fixed=fixed)

    def evaluate(self, out_eparams):
        return self._interpreter.evaluate(dict(), out_eparams, False)


class FittingParams(Params):

    def __init__(self, descs, parameters, conversions, param_type):
        infos, exprs = parse_param_values_strict(descs, parameters, param_type)
        super().__init__(descs, parameters, exprs, conversions)

    def infos(self):
        return self._infos

    def enames(self, free=True, tied=True, fixed=True):
        return self._interpreter.enames(free=free, tied=tied, fixed=fixed)

    def evaluate(self, in_eparams, out_eparams=None, check=True):
        return self._interpreter.evaluate(in_eparams, out_eparams, check)


class EvalParams(parseutils.BasicParserSupport):

    @classmethod
    def load(cls, info, descs):
        opts = load_params_info_common(cls, info)
        return cls(descs, **opts)

    def dump(self, exprs_file):
        return dump_params_info_common(self, exprs_file)

    def __init__(self, descs, parameters, expressions=None):
        super().__init__()
        value_type = ()
        values, exprs = parse_param_values_strict(descs, parameters, value_type)
        self._descs = copy.deepcopy(descs)
        self._infos = values
        self._parameters = copy.deepcopy(parameters)

        print("values:", values)
        print("exprs:", exprs)

        self._interpreter = Interpreter(descs, values | exprs, expressions)

        exit()

    def descs(self):
        return self._descs

    def infos(self):
        return self._infos

    def parameters(self):
        return self._parameters

    def interpreter(self):
        return self._interpreter


def load_eval_params(info, descs):
    return EvalParams.load(info, descs)


def dump_eval_params(params, exprs_file='gbkfit_config_expressions.py'):
    return params.dump(exprs_file)
