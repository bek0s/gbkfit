import abc
from typing import Any

import optuna

from gbkfit.fitting.core import Fitter
from gbkfit.utils import funcutils, iterutils


def objective(trial: optuna.Trial):
    return 0


class FitterOptuna(Fitter, abc.ABC):

    def load_fitting_params(self, info, pdescs):
        pass

    @classmethod
    def load(cls, info, **kwargs):
        return cls()

    def dump(self):
        return dict()

    def __init__(
            self,
            n_trials: int | None,
            timeout: float | None,
            gc_after_trial: bool,
            sampler_args: dict[str, Any],
    ):
        self._n_trials = n_trials
        self._timeout = timeout
        self._gc_after_trial = gc_after_trial
        self._sampler_args = sampler_args
        super().__init__()

    def sampler_args(self):
        return self._sampler_args

    def fit(self):
        study = optuna.create_study()
        study.optimize(objective, n_trials=100)
        best_params = study.best_params




