
import copy
import numbers

from gbkfit.params.expressions import (
    Expressions,
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
        desc = 'parameters'
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['descs'])
        if 'expressions' in opts:
            opts['expressions'] = load_exprs_file(opts['expressions'])
        return cls(descs, **opts)

    def dump(self, exprs_file='gbkfit_config_expressions.py'):
        info = dict(parameters=self.parameters())
        exprs = self._expressions
        if exprs.exprs_func_src() and not exprs.exprs_func_gen():
            info['expressions'] = dump_exprs_file(exprs_file, exprs)
        return info

    def __init__(self, descs, parameters, expressions=None):
        super().__init__()
        def is_val(x): return isinstance(x, (type(None), numbers.Number))
        names, indices, values, exprs = parse_param_values(
            parameters, descs, is_val)[2:]
        enames_from_params = explode_pnames(names, indices)
        enames_from_pdescs = explode_pdescs(descs.values(), descs.keys())
        missing = set(enames_from_pdescs).difference(enames_from_params)
        if missing:
            raise RuntimeError(
                f"information for the following parameters is required "
                f"but not provided: {str(missing)}")
        self._parameters = copy.deepcopy(parameters)
        self._expressions = Expressions(descs, parameters, expressions)

    def parameters(self):
        return self._parameters

    def expressions(self):
        return self._expressions
