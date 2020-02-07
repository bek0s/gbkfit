
import abc

import gbkfit.params
from gbkfit.utils import parseutils


def _make_list(items):
    return list(items) if isinstance(items, list) else [items]


def _expand_param_vector_value_list(value_list, size):
    values = []
    for i in range(size):
        if i < len(value_list):
            values.append(value_list[i])
        else:
            values.append(value_list[-1])
    return values


def _expand_param_vector_value_dict(value_dict, size):
    values = []
    for i in range(size):
        values.append({})
        for akey, avalue in value_dict.items():
            if akey.startswith('*'):
                akey = akey[1:]
                avalue = _make_list(avalue)
                values[i][akey] = avalue[i] if i < len(avalue) else avalue[-1]
            else:
                values[i][akey] = avalue
    return values

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


        self._fit_impl(dataset, model, param_info)

    @abc.abstractmethod
    def _fit_impl(self):
        pass


parser = parseutils.TypedParser(Fitter)
