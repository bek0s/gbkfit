
import copy
import logging
import os
import pathlib

from dataclasses import dataclass
from typing import Any, Optional, Tuple
from operator import attrgetter

import astropy.io.fits as fits
import numpy as np


log = logging.getLogger(__name__)


def _dump_posterior(params, posterior, prefix=''):
    data = []
    data += posterior.logprobs * bool(posterior.logprobs)
    data += posterior.loglikes * bool(posterior.loglikes)
    data += posterior.logpriors * bool(posterior.logpriors)
    data += posterior.samples
    data = np.column_stack(data)
    width1 = 26
    width2 = 18
    header = ''
    header += 'log_probability'.rjust(width1) * bool(posterior.logprobs)
    header += 'log_likelihood'.rjust(width1) * bool(posterior.loglikes)
    header += 'log_prior'.rjust(width1) * bool(posterior.logpriors)
    header += ''.join(param.rjust(width1) for param in params)
    filename = f'{prefix}posterior.txt'
    np.savetxt(filename, data, fmt=f'%{width1}.{width2}e', header=header)


def load_result(filename):
    pass


def make_unique_path(path):
    i = 0
    base = path
    while os.path.exists(path):
        i += 1
        path = f'{base}_{i}'
    return path


def dump_result(output_dir, result):

    output_dir = make_unique_path(output_dir)
    os.makedirs(output_dir)

    # ...
    for i, dataset in enumerate(result.datasets):
        dataset_prefix = os.path.join(output_dir, f"dataset_{i}_")
        result.datasets[i].dump(prefix=dataset_prefix)

    # ...
    for i, solution in enumerate(result.solutions):

        solution_dir = os.path.join(output_dir, 'solutions', str(i))
        os.makedirs(solution_dir)

        # ...
        for dat, mdl, res in zip(result.datasets, solution.model, solution.residual):

        #   fits.writeto('')
        #   fits.writeto
            pass



        print(solution_dir)


    log.info(f"result saved under {output_dir}")


@dataclass()
class FitterResultPosterior:
    logprobs: np.ndarray = None
    loglikes: np.ndarray = None
    logpriors: np.ndarray = None
    samples: np.ndarray = None


@dataclass()
class FitterResultSolution:
    mode: np.ndarray = None
    mean: np.ndarray = None
    covar: np.ndarray = None
    stddev: np.ndarray = None
    posterior: FitterResultPosterior = None
    model: np.ndarray = None
    residual: np.ndarray = None
    chisqr: float = None
    rchisqr: float = None
    extra: dict = None


@dataclass()
class FitterResult:

    datasets: Any = ()
    param_names: tuple = ()
    param_names_tied: tuple = ()
    param_names_fixed: tuple = ()
    posterior: FitterResultPosterior = None
    extras: dict = None
    solutions: Tuple[FitterResultSolution] = ()

    @property
    def champion(self):
        return max(self.solutions, key=attrgetter('chisqr'))


def make_fitter_result(
        objective, parameters, posterior=None, extra=None, solutions=()):
    # We need to have at least one solution.
    # If not, try to calculate one using the posterior (if exists)
    if not solutions:
        if not posterior:
            raise RuntimeError(
                "at least one solution or a posterior distribution "
                "must be provided")
        solutions = (FitterResultSolution(posterior=posterior),)
    # ...
    params_all = parameters.names(free=True, tied=True, fixed=True)
    params_free = parameters.names(free=True, tied=False, fixed=False)
    params_tied = parameters.names(free=False, tied=True, fixed=False)
    params_fixed = parameters.names(free=False, tied=False, fixed=True)
    dof = 100
    # For each solution, try to generate more information (if possible)
    for s in solutions:
        if s.posterior and s.posterior.logprobs and s.posterior.samples:
            if s.mode is None:
                s.mode = s.posterior.samples[np.argmax(s.posterior.logprobs)]
            if s.mean is None:
                s.mean = np.mean(s.posterior.samples, axis=0)
            if s.covar is None:
                s.covar = None
        params = dict(zip(params_free, s.mode))
        params = parameters.expressions().evaluate(params)
        s.model = objective.model().evaluate_h(params)
        s.residual = objective.residual_nddata(params)
        s.wresidual = objective.residual_nddata(params)
        s.chisqr = 1.0
        s.rchisqr = s.chisqr / (dof - len(params_free))
        s.wchisqr = 1.0
        s.rwchisqr = s.wchisqr / (dof - len(params_free))

    return FitterResult(
        objective.datasets(),
        params_all, params_free, posterior, extra, solutions=solutions)
