"""
The eval task: evaluate the galaxies of a configuration, or its objective
(the residuals against the data), or draw mock data of its observations,
and write the outputs. Programs that evaluate a configuration themselves
(e.g. at other values of its parameters) set it up with gbkfit.load.
"""

import logging
import os
from typing import Any, Literal

import numpy as np

from gbkfit.config import ConfigError, config_path
from gbkfit.dataset import dataset_parser
import gbkfit.outputs
import gbkfit.setup
from gbkfit.objective import Objective
from gbkfit.observation import observation_parser
from gbkfit.params import ParamValues, ValidRange
from gbkfit.setup import ConfigFile, SetupEvaluation
from gbkfit.utils import numutils
from . import _detail


__all__ = [
    'eval_'
]


_log = logging.getLogger(__name__)


def eval_(
        mode: Literal['model', 'objective', 'mock'],
        config: str,
        profile_iters: int,
        output_dir: str | None,
        output_dir_mode: Literal['terminate', 'overwrite', 'unique'],
        extra_outputs: bool = False
) -> None:
    """
    Evaluate the galaxies of a configuration at its parameters (their
    fixed values, and the 'value' of the free ones), or its objective, and
    write the outputs as a run (see gbkfit.outputs; its option 'mode' is
    the mode): the configuration (config.yaml, kind 'config'), the
    parameters and the derived values (params.json, 'params'), the models
    (and their extra outputs, if asked for), and in the mode 'objective'
    the data and the residuals too (see Setup.outputs), and the log
    (logs/gbkfit.log, 'log'; see run_log); in a project, the sources of
    the configuration too (setup.json, 'setup'; see write_setup).

    The mode 'mock' writes the models (as 'model'), and mock data of the
    observations: their models measured from what their instruments
    record of the models plus noise, with their errors (see
    draw_mocks). Each mock k is the files of the data of each
    observation (in mocks/<k>/observations/<observation>/, kind
    'mock_data') and a configuration that fits them (mocks/<k>/
    config.yaml, 'mock_config'): the configuration with each observation
    paired with its mock data (its section 'datasets', in the order of the
    observations), which its section 'likelihoods' compares with the
    models (see Likelihoods; it can be given without data). The section
    'mock' of the configuration has the options

    - nmocks (1), nsamples (100) and seed (none): the number of mocks,
      the number of draws of the errors of measured data (e.g. moments),
      and the seed of the random numbers (see draw_mocks).

    The noise of the mock data of each observation is its own (its option
    noise; by default, that of the errors of its data, see noise_of). The
    mock data are in the units of the models.

    Parameters
    ----------
    mode : {'model', 'objective', 'mock'}
        What to evaluate.
    config : str
        The configuration file (YAML or JSON), which can name pieces of
        its project (see gbkfit.project); its relative paths are relative
        to its section 'root', or else to the folder of its project, or
        else to its directory.
    profile_iters : int
        The number of evaluations to time after the outputs are written
        (timings.json, kind 'timings'): each at new values of the free
        parameters, as those of a fit (they alternate between their
        'value' and the next floating-point number), not at the values
        of the last, whose models the plan keeps.
    output_dir : str or None
        The directory of the run; by default (None), the folder of the
        runs of the setup in its project (runs/<name of the setup>; see
        gbkfit.project), or else output.
    output_dir_mode : {'terminate', 'overwrite', 'unique'}
        What to do if the directory exists (see gbkfit.outputs.create).
    extra_outputs : bool, optional
        Whether to write the extra outputs of the models too (kind
        'extra'; see Extras): what the evaluation computes on the way to
        them, for diagnostics (e.g. the light and the velocities of each
        component on the grid of the model, gigabytes for thick
        components on a fine grid, and the model cube before it is
        measured); False by default.

    Raises
    ------
    ConfigError
        If the configuration is invalid, the mode is 'mock' and it has no
        section 'mock', or evaluations are to be timed and it has no free
        parameters.
    """
    file = _read_config(config)
    cfg, root = file.config, file.root
    if mode == 'mock' and 'mock' not in cfg:
        raise ConfigError("the mode 'mock' needs the section 'mock'")
    with gbkfit.outputs.create(
            _detail.run_dir(output_dir, file), 'eval', output_dir_mode,
            dict(mode=mode)) as run, _detail.run_log(run):
        _log.info(f"output will be stored under directory: {run.path}")
        _detail.write_config(run, 'config.yaml', 'config', cfg, root)
        _detail.write_setup(run, file)
        with gbkfit.setup.load(file) as setup:
            _eval(mode, cfg, root, setup, profile_iters, extra_outputs, run)


def _read_config(config: str) -> ConfigFile:
    """
    Read the configuration of an evaluation: of any section (see
    setup.read_config), of which it uses those it needs (e.g. a
    configuration of a fit evaluates).
    """
    _log.info(f"reading configuration from file: {config}...")
    return gbkfit.setup.read_config(config)


def _eval(
        mode: Literal['model', 'objective', 'mock'],
        cfg: dict[str, Any],
        root: str,
        setup: gbkfit.setup.Setup,
        profile_iters: int,
        extra_outputs: bool,
        run: gbkfit.outputs.RunWriter
) -> None:
    """
    Evaluate the setup of a configuration, and write the outputs (see
    eval_).
    """
    group_plan, params = setup.group_plan(), setup.params()
    objective = setup.objective() if mode == 'objective' else None
    if mode == 'objective' and objective is None:
        raise ConfigError(
            "the mode 'objective' needs data, and no observation has any")

    _log.info("calculating model parameters...")
    free = params.free_properties()
    if missing := [name for name, info in free.items() if 'value' not in info]:
        with config_path('params'):
            raise ConfigError(
                f"an evaluation is at the 'value' of each free parameter; "
                f"these have none: {missing}")
    if profile_iters > 0 and not free:
        with config_path('params'):
            raise ConfigError(
                "the profile times evaluations at new values of the free "
                "parameters, as those of a fit; there are none: free one")
    # (in the other modes, the solved elements take their values without
    # data)
    evaluation = setup.evaluate() if objective is not None \
        else SetupEvaluation(params.evaluate(), None)
    param_values = evaluation.values
    run.write('params.json', 'params', dict(
        params=param_values, eparams=params.element_values(param_values),
        derived=params.derived_values(param_values, group_plan.functions())))

    _log.info("evaluating...")
    outputs = setup.outputs(
        evaluation, data=objective is not None, extra=extra_outputs)
    if evaluation.objective is not None:
        _log.info(f"chi-square: {evaluation.objective.merits.chi2}")
    _log.info("storing outputs to the filesystem...")
    gbkfit.outputs.write_outputs(run, outputs)
    if mode == 'mock':
        _log.info(
            f"drawing {setup.mock_options().nmocks()} mock datasets...")
        _write_mocks(cfg, root, setup, param_values, run)

    if profile_iters > 0:
        _profile(setup, objective, profile_iters, run)


def _profile(
        setup: gbkfit.setup.Setup,
        objective: Objective | None,
        iters: int,
        run: gbkfit.outputs.RunWriter
) -> None:
    """
    Time evaluations of the models of a setup (of its objective, if
    given), and write the statistics of the times of their steps
    (timings.json, kind 'timings'; see Timers.stats) and log them.

    Each evaluation is at new values of the free parameters, as those of
    a fit: one at the values of the last would give the models it keeps
    (see GroupPlan.model_d), and time nothing. The free parameters
    alternate between their 'value' (that of the outputs) and the next
    floating-point number (toward the inside of its valid range), so that
    every stage that reads one is evaluated again, to the same models;
    those that read none are not, as in a fit.
    """
    _log.info("running performance test...")
    params = setup.params()
    group_plan = setup.group_plan()
    start = {
        name: info['value']
        for name, info in params.free_properties().items()}
    ranges = params.free_valid_ranges()
    moved = {
        name: _next_value(value, ranges.get(name))
        for name, value in start.items()}
    points = [params.evaluate(moved), params.evaluate(start)]
    # Time the profiled evaluations only (the first evaluation, of the
    # outputs, also warmed up the drivers)
    group_plan.timers().clear()
    for k in range(iters):
        values = points[k % 2]
        if objective is None:
            group_plan.model_d(values)
        else:
            objective.chi2(values)
    time_stats = numutils.nativify({
        name: stats.to_dict()
        for name, stats in group_plan.timers().stats().items()})
    _log.info(_timings_table(time_stats))
    run.write('timings.json', 'timings', time_stats)


def _next_value(value: float, valid: ValidRange | None) -> float:
    """
    Return the next floating-point number after a value, upward unless
    that is outside its valid range (if any; e.g. an inclination of 90
    degrees), then downward.
    """
    up = float(np.nextafter(value, np.inf))
    if valid is None or valid.contains(up):
        return up
    return float(np.nextafter(value, -np.inf))


def _timings_table(time_stats: dict[str, dict[str, Any]]) -> str:
    """
    Return the statistics of the timings of steps (see Timers.stats) as a
    table: a row of each step, a column of each statistic.
    """
    columns = list(next(iter(time_stats.values()), {}))
    rows = [[''] + columns] + [
        [name] + [str(stats[column]) for column in columns]
        for name, stats in time_stats.items()]
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return '\n'.join(
        '  '.join([row[0].ljust(widths[0])] + [
            cell.rjust(width) for cell, width in zip(row[1:], widths[1:])])
        for row in rows)


def _write_mocks(
        cfg: dict[str, Any],
        root: str,
        setup: gbkfit.setup.Setup,
        values: ParamValues,
        run: gbkfit.outputs.RunWriter
) -> None:
    """
    Draw the mock data of the observations of a setup at the parameter
    values (see Setup.mock), and write their files and configurations
    (see eval_).
    """
    group = setup.group()
    with config_path('mock'):
        mocks = setup.mock(values)
    sections = {
        k: v for k, v in cfg.items() if k not in ('mock', 'datasets')}
    for k, mocked_observations in enumerate(mocks):
        observations = []
        datasets = []
        for i, (mocked, info) in enumerate(zip(
                mocked_observations, cfg['observations'])):
            name = group.observation_label(i)
            folder = f'mocks/{k}/observations/{name}'
            directory = os.path.join(run.path, *folder.split('/'))
            os.makedirs(directory, exist_ok=True)
            # (the paths of the files of the data are relative to the
            # root of the configuration, as its other paths; they are
            # of this run, whose earlier files a run into the same
            # directory replaces)
            prefix = directory + os.sep
            new_prefix = os.path.relpath(directory, root) + os.sep
            observation = _rebased(
                observation_parser.dump(mocked, prefix, overwrite=True),
                prefix, new_prefix)
            datasets.append(_rebased(
                dataset_parser.dump(mocked.dataset(), prefix, overwrite=True),
                prefix, new_prefix))
            for filename in sorted(os.listdir(directory)):
                run.record(
                    f'{folder}/{filename}', 'mock_data',
                    os.path.splitext(filename)[1].lstrip('.'),
                    mock=str(k), observation=name)
            # (the driver of an observation is the name of one of those
            # of the configuration)
            driver = {} if info.get('driver') is None \
                else dict(driver=info['driver'])
            observations.append(driver | observation)
        # (its datasets after its observations)
        config = {}
        for key, value in sections.items():
            if key == 'observations':
                config |= dict(observations=observations, datasets=datasets)
            else:
                config[key] = value
        _detail.write_config(
            run, f'mocks/{k}/config.yaml', 'mock_config', config, root,
            mock=str(k))


def _rebased(info: Any, prefix: str, new_prefix: str) -> Any:
    """
    Return the options of a configuration (nested dicts and lists) with
    the strings that start with a prefix starting with another one.
    """
    if isinstance(info, str) and info.startswith(prefix):
        return new_prefix + info.removeprefix(prefix)
    if isinstance(info, list):
        return [_rebased(item, prefix, new_prefix) for item in info]
    if isinstance(info, dict):
        return {k: _rebased(v, prefix, new_prefix) for k, v in info.items()}
    return info
