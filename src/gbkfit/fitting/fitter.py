
import abc

import gbkfit.params
import gbkfit.params.interpreter
import gbkfit.params.utils
from gbkfit.utils import parseutils


class Fitter(parseutils.TypedParserSupport, abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def load_params(info, desc):
        pass

    def __init__(self):
        pass

    def fit(self, objectives, params):
        missing = set(objectives.params()).difference(params.descs())
        if missing:
            raise RuntimeError(
                f"fitting cannot start because information for the following "
                f"parameters is missing: {', '.join(missing)}")
        interpreter = gbkfit.params.interpreter.ParamInterpreter(
            params.descs(), params.exprs())
        result = self._fit_impl(objectives, params, interpreter)
        return result

    @abc.abstractmethod
    def _fit_impl(self, objective, params, interpreter):
        pass


parser = parseutils.TypedParser(Fitter)
