
try:
    import optuna
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "to use the optuna fitter, please install optuna")

from .fitters import *
