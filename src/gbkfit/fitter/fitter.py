
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

    def fit(self, dataset, model, param_info):

        param_info = gbkfit.params.parse_param_fit_info(param_info, model.get_param_descs())

        result = self._impl_fit(dataset, model, param_info)

        return result

    @abc.abstractmethod
    def _impl_fit(self, data, model, params):
        pass


parser = parseutils.TypedParser(Fitter)
