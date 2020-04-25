
import abc

import gbkfit.params
from gbkfit.utils import parseutils


class FitterResultMode:

    def __init__(self):
        pass


class FitterResult:

    @classmethod
    def load(cls, info):
        pass

    def dump(self):
        pass

    def __init__(self):
        pass

    def datasets(self):
        pass


class Fitter(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def type():
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, info):
        pass

    @abc.abstractmethod
    def dump(self):
        pass

    def __init__(self):
        pass

    def parse_params(self, params):
        pass

    def fit(self, objective, extra_pdescs, param_exprs, param_info, param_info_extra=None):


        # validate descs
        # validate exprs
        # validate infos
        result = self._fit_impl(objective, param_info)

        pass


    def fit_(self, dataset, model, param_info):

        param_info = gbkfit.params.parse_param_fit_info(param_info, model.get_param_descs())

        result = self._impl_fit(dataset, model, param_info)

        return result

    @abc.abstractmethod
    def _impl_fit(self, data, model, params):
        pass

    @abc.abstractmethod
    def _fit_impl(self, objective, params):
        pass


parser = parseutils.TypedParser(Fitter)
