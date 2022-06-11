
import copy
import inspect
import json
import logging
import os
import pathlib
import ruamel.yaml

from dataclasses import dataclass, asdict
from typing import Any, Optional
from operator import attrgetter

import numpy as np
import pandas as pd

import gbkfit.dataset
import gbkfit.model
from gbkfit.utils import iterutils

log = logging.getLogger(__name__)


# Use this object to load and dump yaml
yaml = ruamel.yaml.YAML()


def _dump_posterior(posterior, prefix=''):
    float_format = '%.8f'
    with open(f'{prefix}posterior' + '.csv', 'w+') as f:
        f.write(posterior.df.to_csv(index=False, float_format=float_format))
    with open(f'{prefix}posterior' + '.txt', 'w+') as f:
        f.write(posterior.df.to_string(index=False, float_format=float_format))


def load_result(result_dir):

    # Work with absolute paths
    result_dir = os.path.abspath(result_dir)

    #
    # Result directory contains a 'result.yaml' file with a bunch of
    # important information.
    #

    try:
        result = yaml.load(open(os.path.join(result_dir, 'result.yaml')))
    except Exception as e:
        raise RuntimeError(
            "error while reading result directory; "
            "see preceding exception for additional information") from e

    prefix = os.path.join(result_dir, '')
    datasets = gbkfit.dataset.dataset_parser.load(
        result['datasets'], prefix=[prefix])

    # Discover solution directories
    solution_dirs = sorted([str(path) for path in pathlib.Path(
        os.path.join(result_dir, 'solutions')).glob('*')])

    # Create solutions
    for solution_dir in solution_dirs:
        pass

    enames_all = ()
    enames_free = ()
    enames_free = ()
    enames_fixed = {}
    enames_varying = ()
    sols = None
    posterior = None
    extra = None

    parameters = dict()
    param_names = dict()

    return FitterResult(
        datasets, parameters, param_names, sols, posterior, extra)


def dump_result(output_dir, result):

    os.makedirs(output_dir)

    # parameters = result.parameters.dump()

    parameters_info = dict(
        params_names=dict(

        )
    )
    parameters_filename = os.path.join(output_dir, 'parameters')

    # Dump parameter names

    # Dump parameter configuration
    # parameters = iterutils.nativify(parameters)
    # filename_root = os.path.join(output_dir, 'parameters')
    # with open(f'{filename_root}.json', 'w+') as f:
    #     json.dump(parameters, f, indent=2)
    # with open(f'{filename_root}.yaml', 'w+') as f:
    #     yaml.dump(parameters, f)

    # ...
    root_info = dict(
        datasets=[],
        param_names=result.param_names,
        extra=result.extra)

    # Dump datasets
    for i, dataset in enumerate(result.datasets):
        prefix = os.path.join(output_dir, f'dataset_{i}_')
        root_info['datasets'].append(gbkfit.dataset.dataset_parser.dump(
            dataset, prefix=prefix, dump_full_path=False))

    root_info = iterutils.nativify(root_info)
    filename_root = os.path.join(output_dir, 'result')
    with open(f'{filename_root}.json', 'w+') as f:
        json.dump(root_info, f, indent=2)
    with open(f'{filename_root}.yaml', 'w+') as f:
        yaml.dump(root_info, f)

    # Dump global posterior
    if result.posterior:
        prefix = os.path.join(output_dir, '')
        _dump_posterior(result.posterior, prefix)

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
            index=result.param_names['varying'],
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
                gbkfit.dataset.Data(model[key]['d']).dump(
                    os.path.join(solution_dir, filename_mdl))
                gbkfit.dataset.Data(resid[key]).dump(
                    os.path.join(solution_dir, filename_res))
        # Dump posterior
        if sol.posterior:
            prefix = os.path.join(solution_dir, '')
            _dump_posterior(result.posterior, prefix)

        print(df.to_string())


@dataclass
class FitterResultPosterior:

    logprobs: np.ndarray = None
    loglikes: np.ndarray = None
    logpriors: np.ndarray = None
    samples: np.ndarray = None
    df: pd.DataFrame = None


@dataclass
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

    datasets: list[gbkfit.dataset.Dataset]
    parameters: dict
    param_names: dict
    solutions: tuple[FitterResultSolution]
    posterior: None | FitterResultPosterior
    extra: dict

    @property
    def champion(self):
        return max(self.solutions, key=attrgetter('chisqr'))

    def summary(self):
        summary = inspect.cleandoc(
            f"""
            -------
            summary
            -------
            number of all parameters: {len(self.param_names['all'])}
            number of free parameters: {len(self.param_names['free'])}
            number of tied parameters: {len(self.param_names['tied'])}
            number of fixed parameters: {len(self.param_names['fixed'])}
            number of solutions: {len(self.solutions)}
            """)
        return summary


def make_fitter_result_posterior(posterior, parameters):

    enames_free = parameters.enames(False, False, True)
    enames_varying = parameters.enames(False, True, True)

    samples = posterior.get('samples')
    loglikes = posterior.get('loglikes')

    if samples is None:
        raise RuntimeError("samples must be provided")

    if loglikes is None:
        raise RuntimeError("loglikes must be provided")

    # Evaluate log priors
    logpriors = np.array(
        [parameters.priors().log_prob(dict(zip(enames_free, sample)))
         for sample in samples])

    # Evaluate log probabilities
    logprobs = loglikes + logpriors

    # Evaluate samples for all varying (free+tied) parameters
    samples_varying = np.full((samples.shape[0], len(enames_varying)), np.nan)
    for i, sample in enumerate(samples):
        eparams_free = dict(zip(enames_free, sample))
        eparams_varying = dict.fromkeys(enames_varying, None)
        parameters.evaluate(eparams_free, eparams_varying)
        samples_varying[i, :] = list(eparams_varying.values())

    # Create the final FitterResultPosterior object
    result_posterior = FitterResultPosterior()
    result_posterior.logpriors = logpriors
    result_posterior.loglikes = loglikes
    result_posterior.logprobs = logprobs
    result_posterior.samples = samples_varying
    result_posterior.df = pd.DataFrame(
        np.column_stack([logprobs, loglikes, logpriors, samples]),
        columns=['log_probability', 'log_likelihood', 'log_prior',
                 *enames_varying])

    return result_posterior


def make_fitter_result(
        objective, parameters, posterior=None, extra=None, solutions=None):

    # At least one solution or a global posterior is required
    if not solutions:
        if not posterior:
            raise RuntimeError(
                "at least one solution or a global posterior "
                "is required in order to create a Fitter Result")
        # If we have a global posterior but not solutions,
        # transform the global posterior to a new solution
        solutions = dict(posterior=posterior)
        posterior = None

    # Ensure solutions are iterable for convenience
    solutions = iterutils.tuplify(solutions, False)

    # Process the global posterior
    if posterior:
        posterior = make_fitter_result_posterior(posterior, parameters)

    # Make some arrays with exploded param names for later use
    enames_all = parameters.enames(True, True, True)
    enames_free = parameters.enames(False, False, True)
    enames_tied = parameters.enames(False, True, False)
    enames_fixed = parameters.enames(True, False, False)
    enames_varying = parameters.enames(False, True, True)

    sols = []

    # For each solution, create a FitterResultSolution
    # and try to populate as many of its fields as possible.
    for i, s in enumerate(solutions):
        sol = FitterResultSolution()
        # ...
        if 'mode' in s:
            eparams_free = dict(zip(enames_free, s['mode']))
            eparams_varying = {p: None for p in enames_varying}
            parameters.evaluate(eparams_free, eparams_varying, True)
            sol.mode = np.array(list(eparams_varying.values()))
        # Calculate statistical quantities from posterior
        if 'posterior' in s:
            posterior = make_fitter_result_posterior(s['posterior'], parameters)
            sol.covar = np.cov(posterior.samples, rowvar=False)
            sol.mean = np.mean(posterior.samples, axis=0)
            sol.std = np.std(posterior.samples, axis=0)
            # If no
            sol.mode = posterior.samples[np.argmax(posterior.logprobs), :]
            if sol.mode is None:
                sol.mode = posterior.samples[np.argmax(posterior.logprobs)]

            sol.posterior = posterior

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
        params = parameters.evaluate(eparams_free)
        dof = 100
        sol.model = objective.model_h(params)
        sol.residual = objective.residual_nddata_h(params)
        sol.wresidual = objective.residual_nddata_h(params)
        sol.chisqr = 1.0
        sol.rchisqr = sol.chisqr / (dof - len(enames_free))
        sol.wchisqr = 1.0
        sol.rwchisqr = sol.wchisqr / (dof - len(enames_free))

        # ...
        sols.append(sol)

    # Sort solution by chi-squared
    sols = sorted(sols, key=lambda x: x.chisqr)

    param_names = dict(
        all=enames_all,
        free=enames_free,
        tied=enames_tied,
        fixed=enames_fixed,
        varying=enames_varying)

    return FitterResult(
        objective.datasets(), parameters, param_names, sols, posterior, extra)

