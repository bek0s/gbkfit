
import copy
import inspect
import json
import logging
import os
import pathlib
import ruamel.yaml

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple
from operator import attrgetter

import numpy as np
import pandas as pd

import gbkfit.dataset
from gbkfit.utils import iterutils

log = logging.getLogger(__name__)


# Use this object to load and dump yaml
yaml = ruamel.yaml.YAML()


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


def load_result(input_dir):

    #json.load()
    pass


def _dump_object(filename, obj, json_=True, yaml_=True):
    if json_:
        with open(filename + '.json', 'w+') as f:
            json.dump(obj, f, indent=2)
    if yaml_:
        with open(filename + '.yaml', 'w+') as f:
            yaml.dump(obj, f)


def dump_result(output_dir, result):

    os.makedirs(output_dir)

    root_info = dict(
        params_all=result.params_all,
        params_free=result.params_free,
        params_tied=result.params_tied,
        params_fixed=result.params_fixed,
        params_varying=result.params_varying,
        extra=result.extra)

    filename_root = os.path.join(output_dir, 'result')
    with open(filename_root + '.json', 'w+') as f:
        f.write(json.dumps(root_info, indent=2))

    # Dump datasets
    for i, dataset in enumerate(result.datasets):
        for key, data in result.datasets[i].items():
            prefix = os.path.join(output_dir, f'dataset_{i}_{key}')
            filename_d = f'{prefix}_d.fits'
            filename_m = f'{prefix}_m.fits'
            filename_e = f'{prefix}_e.fits'
            data.dump(filename_d, filename_m, filename_e)

    # Dump global posterior
    if result.posterior:
        prefix = os.path.join(output_dir, '')
        _dump_posterior(result.params_varying, result.posterior, prefix)

    # Dump solutions
    for i, sol in enumerate(result.solutions):
        solution_dir = os.path.join(output_dir, 'solutions', str(i))
        os.makedirs(solution_dir)
        # Build parameter data frame
        sol_dict = asdict(sol)
        col_names = []
        for col_name in ['mode', 'mean', 'std']:
            if sol_dict[col_name] is not None:
                col_names.append(col_name)
        df = pd.DataFrame(
            index=result.params_varying,
            columns=col_names)
        for col_name in col_names:
            df[col_name] = sol_dict[col_name]
        # Dump the parameter data frame on various formats
        filename_params = os.path.join(solution_dir, 'parameters')
        with open(filename_params + '.csv', 'w+') as f:
            f.write(df.to_csv())
        with open(filename_params + '.txt', 'w+') as f:
            f.write(df.to_string())
        with open(filename_params + '.json', 'w+') as f:
            json.dump(json.loads(df.to_json(orient='index')), f, indent=2)
        with open(filename_params + '.yaml', 'w+') as f:
            yaml.dump(json.loads(df.to_json(orient='index')), f)
        # Dump model and residual data
        for j, dataset in enumerate(result.datasets):
            model = sol.model[j]
            resid = sol.residual[j]
            for key in dataset:
                filename_mdl = f'bestfit_{j}_mdl_{key}_d.fits'
                filename_res = f'bestfit_{j}_res_{key}_d.fits'
                gbkfit.dataset.Data(model[key]['data']).dump(
                    os.path.join(solution_dir, filename_mdl))
                gbkfit.dataset.Data(resid[key]).dump(
                    os.path.join(solution_dir, filename_res))
        # Dump posterior
        if sol.posterior:
            prefix = os.path.join(solution_dir, '')
            _dump_posterior(result.params_varying, result.posterior, prefix)

        print(df.to_string())


@dataclass()
class FitterResultPosterior:
    logprobs: np.ndarray = None
    loglikes: np.ndarray = None
    logpriors: np.ndarray = None
    samples: np.ndarray = None


@dataclass()
class FitterResultSolution:

    mode: dict = None
    mean: dict = None
    std: dict = None
    covar: np.ndarray = None
    posterior: FitterResultPosterior = None
    model: np.ndarray = None
    residual: np.ndarray = None
    chisqr: float = None
    rchisqr: float = None
    extra: dict = None

    def summary(self):
        summary = inspect.cleandoc(
            f"""
            """)
        return summary


@dataclass
class FitterResult:

    datasets: Any
    params_all: Tuple
    params_free: Tuple
    params_tied: Tuple
    params_fixed: Dict
    params_varying: Tuple
    solutions: Tuple[FitterResultSolution]
    posterior: FitterResultPosterior
    extra: Dict

    @property
    def champion(self):
        return max(self.solutions, key=attrgetter('chisqr'))

    def summary(self):
        summary = inspect.cleandoc(
            f"""
            -------
            summary
            -------
            number of all parameters: {len(self.params_all)}
            number of free parameters: {len(self.params_free)}
            number of tied parameters: {len(self.params_tied)}
            number of fixed parameters: {len(self.params_fixed)}
            number of solutions: {len(self.solutions)}
            """)
        return summary

def make_fitter_result(
        objective, parameters, posterior=None, extra=None, solutions=()):

    # At least one solution or a global posterior is required
    if not solutions:
        if not posterior:
            raise RuntimeError(
                "at least one solution or a global posterior "
                "is required")
        solutions = dict(posterior=posterior)

    # Ensure solutions are iterable for convenience
    solutions = iterutils.tuplify(solutions)

    # Make some arrays with exploded param names for later use
    exprs = parameters.exprs()
    enames_all = exprs.enames(True, True, True)
    enames_free = exprs.enames(True, False, False)
    enames_tied = exprs.enames(False, True, False)
    enames_fixed = exprs.enames(False, False, True)
    enames_varying = exprs.enames(True, True, False)
    params_fixed = exprs.fixed_dict()

    sols = []

    for i, s in enumerate(solutions):

        sol = FitterResultSolution()

        if 'mode' in s:
            eparams_free = dict(zip(enames_free, s['mode']))
            eparams_varying = {p: None for p in enames_varying}
            exprs.evaluate(eparams_free, eparams_varying)
            sol.mode = np.array(list(eparams_varying.values()))

        # Calculate statistical quantities from posterior
        if 'posterior' in s:
            sposterior = s['posterior']
            sol.posterior = FitterResultPosterior()
            sol.posterior.logpriors = sposterior.get('logpriors')
            sol.posterior.loglikes = sposterior.get('loglikes')
            sol.posterior.logprobs = sposterior.get('logprobs')
            sol.posterior.samples = np.full((0, 0), np.nan)
            for j, row in enumerate(sposterior['samples']):
                eparams_free = dict(zip(enames_free, row))
                eparams_varying = {p: None for p in enames_varying}
                exprs.evaluate(eparams_free, eparams_varying)
                sol.posterior.samples[j, :] = eparams_varying.values()
            sol.covar = np.cov(sol.posterior.samples, rowvar=False)
            sol.mean = np.mean(s.posterior.samples, axis=0)
            sol.std = np.std(s.posterior.samples, axis=0)
            if not sol.mode:
                sol.mode = s.posterior.samples[np.argmax(s.posterior.logprobs)]

        # Otherwise, salvage whatever is available
        else:
            if 'mean' in s:
                sol.mean = np.full(len(enames_varying), np.nan)
                for j, param in enumerate(enames_free):
                    sol.mean[enames_varying.index(param)] = s['mean'][j]
            if 'std' in s:
                sol.std = np.full(len(enames_varying), np.nan)
                for j, param in enumerate(enames_free):
                    sol.std[enames_varying.index(param)] = s['std'][j]
            if 'covar' in s:
                pass

        # Mode must be provided or recovered somehow
        if sol.mode is None:
            raise RuntimeError(
                f"mode was not provided or could not be recovered "
                f"for solution {i}")

        # Calculate quantities using the mode
        eparams_varying = dict(zip(enames_varying, sol.mode))
        eparams_free = {n: eparams_varying[n] for n in enames_free}
        print(eparams_free)
        params = parameters.expressions().evaluate(eparams_free)
        dof = 100
        sol.model = objective.models().evaluate_h(params)
        sol.residual = objective.residual_nddata(params)
        sol.wresidual = objective.residual_nddata(params)
        sol.chisqr = 1.0
        sol.rchisqr = sol.chisqr / (dof - len(enames_free))
        sol.wchisqr = 1.0
        sol.rwchisqr = sol.wchisqr / (dof - len(enames_free))

        # ...
        sols.append(sol)

    return FitterResult(
        objective.datasets(),
        enames_all, enames_free, enames_tied, params_fixed, enames_varying, sols, posterior, extra)

