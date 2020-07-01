
import abc
import logging

from gbkfit.params.utils import (
    explode_params, explode_pdescs,
    parse_param_exprs, parse_param_values)


log = logging.getLogger(__name__)


class FitParam(abc.ABC):
    pass


class FitParams(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def load(cls, info, descs):
        pass

    def __init__(self, params, descs):

        # Extract information about the parameters
        keys, values, names, indices, infos, exprs = parse_param_values(
            params, descs, lambda x: isinstance(x, FitParam))

        # Make sure all parameters have been provided
        missing = set(explode_pdescs(descs)).difference(
            explode_params(names, indices))
        if missing:
            raise RuntimeError(
                f"the following parameters are required but not provided: "
                f"{str(missing)}")

        self._descs = descs
        self._infos = infos
        self._exprs = dict(zip(*parse_param_exprs(exprs, descs)[0:2]))

    def descs(self):
        return self._descs

    def exprs(self):
        return self._exprs

    def infos(self):
        return self._infos

    def enames(self):
        pass

    def enames_free(self):
        pass

    def enames_fixed(self):
        pass
