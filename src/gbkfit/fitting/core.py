
import abc

from gbkfit.utils import parseutils


class Fitter(parseutils.TypedSerializable, abc.ABC):

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, objectives, parameters):
        if missing := set(objectives.pdescs()).difference(parameters.pdescs()):
            raise RuntimeError(
                f"fitting cannot start because information for the following "
                f"parameters is missing: {missing}")
        result = self._fit_impl(objectives, parameters)
        return result

    @abc.abstractmethod
    def _fit_impl(self, objective, params):
        pass


fitter_parser = parseutils.TypedParser(Fitter)
