
import abc
import copy
import numbers

from gbkfit.params.utils import (
    explode_pdescs,
    explode_pnames,
    parse_param_values)

from gbkfit.utils import parseutils
from gbkfit.params.expressions import Expressions


class FitParam(parseutils.BasicParserSupport, abc.ABC):
    pass


class FitParams(parseutils.BasicParserSupport, abc.ABC):

    def __init__(self, descs, params, exprs_func):

        def is_nfo(x): return isinstance(x, FitParam)
        def is_nil(x): return isinstance(x, type(None))
        def is_num(x): return isinstance(x, numbers.Number)
        def is_val(x): return is_nfo(x) or is_nil(x) or is_num(x)
        names, indices, values, exprs = parse_param_values(
            params, descs, is_val)[2:]
        vals_nfo = dict(filter(lambda x: is_nfo(x[1]), values.items()))
        vals_nil = dict(filter(lambda x: is_nil(x[1]), values.items()))
        vals_num = dict(filter(lambda x: is_num(x[1]), values.items()))

        enames_from_params = explode_pnames(names, indices)
        enames_from_pdescs = explode_pdescs(descs.values(), descs.keys())
        missing = set(enames_from_pdescs).difference(enames_from_params)
        if missing:
            raise RuntimeError(
                f"information for the following parameters is required "
                f"but not provided: {str(missing)}")

        self._descs = copy.deepcopy(descs)
        self._infos = vals_nfo
        exprs = {**vals_nil, **vals_num, **exprs}
        self._exprs = Expressions(descs, exprs, exprs_func)

    def descs(self):
        return self._descs

    def infos(self):
        return self._infos

    def exprs(self):
        return self._exprs

    def names(self, free=True, tied=False, fixed=False):
        return self._exprs.enames(free, tied, fixed)

    def expressions(self):
        return self._exprs
