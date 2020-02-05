
import abc

from gbkfit.utils import parseutils


class Fitter(abc.ABC):

    @abc.abstractmethod
    def fit(self, dataset, model, parset):
        pass


parser = parseutils.TypedParser(Fitter)
