
import copy
import numbers

from gbkfit.params.interpreter import (
    Interpreter,
    load_exprs_file,
    dump_exprs_file)
from gbkfit.params.utils import parse_param_values_strict
from gbkfit.utils import parseutils


def load_params_info_common(cls, info):
    desc = parseutils.make_basic_desc(cls, 'params')
    opts = parseutils.parse_options_for_callable(
        info, desc, cls.__init__, fun_ignore_args=['descs'])
    if 'expressions' in opts:
        opts['expressions'] = load_exprs_file(opts['expressions'])
    return opts


def dump_params_info_common(params, exprs_file):
    info = dict(parameters=params.parameters())
    exprs_func_gen = params.interpreter().exprs_func_gen()
    exprs_func_obj = params.interpreter().exprs_func_obj()
    exprs_func_src = params.interpreter().exprs_func_src()
    if not exprs_func_gen and exprs_func_src:
        info['expressions'] = dump_exprs_file(
            exprs_file, exprs_func_obj, exprs_func_src)
    return info


class EvalParams(parseutils.BasicParserSupport):

    @classmethod
    def load(cls, info, descs):
        opts = load_params_info_common(cls, info)
        return cls(descs, **opts)

    def dump(self, exprs_file):
        return dump_params_info_common(self, exprs_file)

    def __init__(self, descs, parameters, expressions=None):
        super().__init__()
        value_type = (type(None), numbers.Real)
        values, exprs = parse_param_values_strict(descs, parameters, value_type)
        self._descs = copy.deepcopy(descs)
        self._infos = values
        self._parameters = copy.deepcopy(parameters)
        self._interpreter = Interpreter(descs, values | exprs, expressions)

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
