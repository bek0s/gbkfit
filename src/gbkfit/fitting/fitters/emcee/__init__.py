
try:
    import emcee
except ImportError:
    raise ImportError(
        "to use the emcee fitter please install emcee")


class FitterEmcee:

    @staticmethod
    def type():
        return "emcee"
