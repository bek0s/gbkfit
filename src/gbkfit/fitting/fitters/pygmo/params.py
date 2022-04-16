
import gbkfit.params.utils as paramutils
from gbkfit.fitting.params import FitParam, FitParams
from gbkfit.utils import parseutils


__all__ = ['FitParamPygmo', 'FitParamsPygmo']


class FitParamPygmo(FitParam):

    @classmethod
    def load(cls, info):
        desc = parseutils.make_basic_desc(cls, 'fit parameter')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_rename_args=dict(
                initial_value='value',
                initial_scale='scale',
                minimum='min',
                maximum='max'))
        return cls(**opts)

    def dump(self):
        return dict(
            initial_value=self.initial_value(),
            initial_scale=self.initial_scale(),
            minimum=self.minimum(),
            maximum=self.maximum())

    def __init__(self, minimum, maximum, initial_value=None, initial_scale=None):
        super().__init__()
        """
        if (initial_value is None) != (initial_scale is None):
            raise RuntimeError(
                "the initial value and initial scale must be defined together")
        """
        self._has_initial = None not in [initial_value, initial_scale]
        self._initial_value = initial_value
        self._initial_scale = initial_scale
        self._minimum = minimum
        self._maximum = maximum

    def has_initial(self):
        return self._has_initial

    def initial_value(self):
        return self._initial_value

    def initial_scale(self):
        return self._initial_scale

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum


class FitParamsPygmo(FitParams):

    @classmethod
    def load(cls, info, descs):
        desc = parseutils.make_basic_desc(cls, 'fit parameters')
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['descs'])
        infos, exprs = paramutils.parse_param_info(
            opts['parameters'], descs)[4:]
        for key, val in infos.items():
            try:
                infos[key] = FitParamPygmo.load(val)
            except RuntimeError as e:
                raise RuntimeError(
                    f"could not parse information for parameter '{key}'; "
                    f"reason: {str(e)}") from e
        parameters = infos | exprs
        expressions = paramutils.load_param_value_conversions(
            opts.get('value_conversions'))
        return cls(descs, parameters, expressions)

    def dump(self):
        return super().dump()

    def __init__(self, descs, parameters, expressions=None):
        super().__init__(descs, parameters, expressions)
