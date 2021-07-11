
import copy
import numbers

from gbkfit.params.expressions import (
    Interpreter,
    load_exprs_file,
    dump_exprs_file)
from gbkfit.params.utils import (
    explode_pdescs,
    explode_pnames,
    parse_param_values)
from gbkfit.utils import parseutils


class EvalParams(parseutils.Serializable):

    @classmethod
    def load(cls, info, descs):
        desc = parseutils.make_basic_desc(cls, 'params')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['descs'])
        if 'expressions' in opts:
            opts['expressions'] = load_exprs_file(opts['expressions'])
        return cls(descs, **opts)

    def dump(self, exprs_file):
        info = dict(parameters=self.parameters())
        exprs_func_gen = self._interpreter.exprs_func_gen()
        exprs_func_obj = self._interpreter.exprs_func_obj()
        exprs_func_src = self._interpreter.exprs_func_src()
        if not exprs_func_gen and exprs_func_src:
            info['expressions'] = dump_exprs_file(
                exprs_file, exprs_func_obj, exprs_func_src)
        return info

    def __init__(self, descs, parameters, expressions=None):
        super().__init__()
        def is_val(x): return isinstance(x, (type(None), numbers.Real))
        names, indices = parse_param_values(
            parameters, descs, is_val)[2:4]
        enames_from_params = explode_pnames(names, indices)
        enames_from_pdescs = explode_pdescs(descs.values(), descs.keys())
        if missing := set(enames_from_pdescs) - set(enames_from_params):
            raise RuntimeError(
                f"information for the following parameters is required "
                f"but not provided: {str(missing)}")
        self._descs = copy.deepcopy(descs)
        self._parameters = copy.deepcopy(parameters)
        self._interpreter = Interpreter(descs, parameters, expressions)

    def descs(self):
        return self._descs

    def parameters(self):
        return self._parameters

    def interpreter(self):
        return self._interpreter


def load_eval_params(info, descs):
    return EvalParams.load(info, descs)


def dump_eval_params(params, exprs_file='gbkfit_config_expressions.py'):
    return params.dump(exprs_file)
