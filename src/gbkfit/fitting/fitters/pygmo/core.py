
import abc
import collections.abc
import logging

from gbkfit.fitting import fitutils
from gbkfit.fitting.core import FitParam, FitParams, Fitter
from gbkfit.params import parsers as param_parsers
from gbkfit.utils import iterutils, parseutils


__all__ = [
    'FitParamPygmo'
]


_log = logging.getLogger(__name__)


class FitParamPygmo(FitParam):

    @classmethod
    def load(cls, info):
        desc = parseutils.make_basic_desc(cls, 'fit parameter')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_rename_args=dict(
                initial_value='value',
                initial_width='width',
                minimum='min',
                maximum='max'))
        return cls(**opts)

    def dump(self):
        info = dict(
            value=self.initial_value(),
            width=self.initial_width(),
            min=self.minimum(),
            max=self.maximum())
        info = iterutils.remove_from_mapping_by_value(info, None)
        return info

    def __init__(
            self, minimum, maximum, initial_value=None, initial_width=None):
        super().__init__()
        initial_value, initial_width, initial_value_min, initial_value_max = \
            fitutils.prepare_optional_initial_value_min_max(
                initial_value, initial_width, minimum, maximum)
        self._has_initial = None not in [initial_value, initial_width]
        self._initial_value = initial_value
        self._initial_width = initial_width
        self._initial_value_minimum = initial_value_min
        self._initial_value_maximum = initial_value_max
        self._minimum = minimum
        self._maximum = maximum

    def has_initial(self):
        return self._has_initial

    def initial_value(self):
        return self._initial_value

    def initial_width(self):
        return self._initial_width

    def initial_value_minimum(self):
        return self._initial_value_minimum

    def initial_value_maximum(self):
        return self._initial_value_maximum

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum


class FitParamsPygmo(FitParams):

    @staticmethod
    def load_param(info):
        return FitParamPygmo.load(info)

    @classmethod
    def load(cls, info, pdescs):
        desc = parseutils.make_basic_desc(cls, 'fit params')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['pdescs'])
        opts = param_parsers.load_params_parameters_conversions(
            opts, pdescs, collections.abc.Mapping, cls.load_param)
        return cls(pdescs, **opts)

    def dump(self, conversions_file):
        return param_parsers.dump_params_parameters_conversions(
            self, FitParamPygmo, lambda x: x.dump(), conversions_file)

    def __init__(self, pdescs, parameters, conversions=None):
        super().__init__(pdescs, parameters, conversions, FitParamPygmo)


class FitterPygmo(Fitter, abc.ABC):

    @staticmethod
    def load_params(info, descs):
        return FitParamsPygmo.load(info, descs)

    @classmethod
    def load(cls, info, **kwargs):
        desc = parseutils.make_typed_desc(cls, 'pygmo fitter')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        return dict()

    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)

    def _setup_algorithm(self, options, parameters):
        pass

    def _fit_impl(self, objective, parameters):

        ndim = len(parameters.infos())
        print("OMG")



        pass
