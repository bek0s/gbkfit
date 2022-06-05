
import abc


from gbkfit.fitting.core import FitParam, FitParams, Fitter


class FitParamDynestySNS(FitParam, abc.ABC):
    pass


class FitParamsDynestySNS(FitParams, abc.ABC):
    pass


class FitterDynesty(Fitter, abc.ABC):
    pass
