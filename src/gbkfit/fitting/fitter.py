
import abc

from gbkfit.utils import parseutils


class Fitter(parseutils.TypedParserSupport, abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def load_params(info, desc):
        pass

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'fitter')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def __init__(self):
        pass

    def fit(self, objectives, parameters):
        missing = set(objectives.params()).difference(parameters.descs())
        if missing:
            raise RuntimeError(
                f"fitting cannot start because information for the following "
                f"parameters is missing: {', '.join(missing)}")
        result = self._fit_impl(objectives, parameters)
        return result

    @abc.abstractmethod
    def _fit_impl(self, objective, params):
        pass


parser = parseutils.TypedParser(Fitter)
