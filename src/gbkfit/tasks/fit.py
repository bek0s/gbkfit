"""
The fit task: fit the galaxies of a configuration to its data, and write
the result.
"""

import contextlib
import dataclasses
import json
import logging
import math
import os
import signal
import threading
import time
from collections.abc import Iterator
from typing import Any, Literal, NamedTuple

import gbkfit.driver
import gbkfit.fitting
import gbkfit.objective
import gbkfit.outputs
import gbkfit.observation
import gbkfit.params
from gbkfit.utils import numutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _detail, _progress


_log = logging.getLogger(__name__)


# The paths of products of a fit in its run (see gbkfit.outputs): the
# configuration and the log of the evaluations, to resume it from, and the
# result, to bootstrap it
_CONFIG = 'config.yaml'
_EVALUATIONS = 'logs/evaluations.bin'
_RESULT = 'fit/result.json'


def fit(
        config: str,
        output_dir: str | None,
        output_dir_mode: Literal['terminate', 'overwrite', 'unique'],
        evaluation_log: bool = True
) -> None:
    """
    Fit the galaxies of a configuration to its data, and write the outputs
    as a run (see gbkfit.outputs): the configuration (config.yaml, kind
    'config'), the result in fit/ (see FitResult.write), the data and the
    models and residuals at the best fit (see data_outputs and
    evaluation_outputs), and fit/restart.yaml ('restart_config'): the
    configuration with the free parameters starting at the best fit, to
    fit again from there (e.g. with another fitter). With a section
    'parallel' ({drivers: [names]}), the batches of points of the fitter
    are evaluated at once on copies of the objective on those drivers
    (see EvaluationPool), e.g. one per GPU. As it fits, it shows its
    progress on the console, writes every report of it to
    logs/progress.jsonl ('progress'), its log to logs/gbkfit.log ('log';
    see run_log), and the configuration at the best
    fit so far to fit/best.yaml ('best_config'; see _progress), from
    which a stopped fit can start again. In a project, it writes the
    sources of the configuration too (setup.json, 'setup'; see
    write_setup).

    With a section 'bootstrap' (see BootstrapOptions), the errors of the
    best fit are then bootstrapped too (see gbkfit.fitting.bootstrap),
    after the outputs of the fit are written, and the result is written
    again with the bootstrap; a fit can be bootstrapped later as well (see
    bootstrap).

    Ctrl-C stops the fit, which ends with the best point so far (and the
    chain so far of an MCMC sampler; see Fitter.fit); a second Ctrl-C
    quits at once. Unless turned off, the evaluations are logged
    (logs/evaluations.bin, 'evaluations'; see EvaluationLog), from which
    a stopped or killed fit resumes (see resume).

    Parameters
    ----------
    config : str
        The configuration file (YAML or JSON), which can name pieces of
        its project (see gbkfit.project); its relative paths are relative
        to its section 'root', or else to the folder of its project, or
        else to its directory.
    output_dir : str or None
        The directory of the run; by default (None), the folder of the
        runs of the setup in its project (runs/<name of the setup>; see
        gbkfit.project), or else output.
    output_dir_mode : {'terminate', 'overwrite', 'unique'}
        What to do if the directory exists (see gbkfit.outputs.create).
    evaluation_log : bool, optional
        Whether to log the evaluations, to resume from (true by default).

    Raises
    ------
    ConfigError
        If the configuration is invalid.
    """
    file = _read_config(config)
    with gbkfit.outputs.create(
            _detail.run_dir(output_dir, file), 'fit', output_dir_mode) \
            as run, _detail.run_log(run):
        _log.info(f"output will be stored under directory: {run.path}")
        _detail.write_config(run, _CONFIG, 'config', file.config, file.root)
        _detail.write_setup(run, file)
        log = None
        if evaluation_log:
            log = gbkfit.fitting.EvaluationLog(run.filename(_EVALUATIONS))
            run.record(_EVALUATIONS, 'evaluations', 'bin')
        _fit(file.config, file.root, run, log)


def resume(output_dir: str) -> None:
    """
    Resume a fit that stopped or was killed, in its run: run it again from
    its configuration, answering the fitter from the log of its
    evaluations while it lasts (see EvaluationLog), and write its outputs
    to the run (see fit). The fit must have had the log; a least-squares
    fit, which has nothing in it, starts again (fit fit/best.yaml
    instead, to start from its best point).

    Parameters
    ----------
    output_dir : str
        The directory of the run of the fit.

    Raises
    ------
    ConfigError
        If the directory is not the run of a fit with a configuration and
        a log, or the configuration is invalid.
    """
    config, evaluations = _products_of_fit(
        output_dir, 'resume', 'config', 'evaluations')
    file = _read_config(config)
    try:
        log = gbkfit.fitting.EvaluationLog.resume(evaluations)
    except ValueError as e:
        raise ConfigError(str(e)) from e
    with gbkfit.outputs.reopen(output_dir) as run, _detail.run_log(run):
        _fit(file.config, file.root, run, log)


def bootstrap(output_dir: str, config: str) -> None:
    """
    Bootstrap the errors of a fit in its run (see
    gbkfit.fitting.bootstrap): rebuild its objective from its
    configuration, refit from its best fit by the section 'bootstrap' of
    a configuration (see BootstrapOptions; the refits use the fitter of
    the fit unless it gives one), and add the bootstrap to the result
    (the entry 'bootstrap' of fit/result.json, and fit/bootstrap.ecsv;
    see FitResult.write), and the configuration, as fit/bootstrap.yaml
    (kind 'bootstrap_config').

    Parameters
    ----------
    output_dir : str
        The directory of the run of the fit.
    config : str
        The configuration file of the bootstrap (YAML or JSON), with only
        the section 'bootstrap'.

    Raises
    ------
    ConfigError
        If the directory is not the run of a fit with a configuration and
        a result, or a configuration is invalid.
    """
    config_file, result_file = _products_of_fit(
        output_dir, 'bootstrap', 'config', 'fit_result')
    file = _read_config(config_file)
    _log.info(f"reading the configuration of the bootstrap: {config}...")
    bootstrap_cfg = _detail.read_config(config, ('bootstrap',)).config
    setup = _set_up(file.config, file.root)
    options = _load_bootstrap(bootstrap_cfg, setup)
    with open(result_file) as f:
        result = json.load(f)
    best = result['solutions'][0]
    with gbkfit.outputs.reopen(output_dir) as run, _detail.run_log(run):
        boot = _bootstrap(setup, best['elements'], options)
        result['bootstrap'] = boot.info()
        run.write(_RESULT, 'fit_result', result)
        run.write('fit/bootstrap.ecsv', 'bootstrap', boot.table())
        run.write('fit/bootstrap.yaml', 'bootstrap_config', bootstrap_cfg)
    _log.info(f"bootstrap:\n{_bootstrap_summary(best, boot)}")


def _products_of_fit(output_dir: str, action: str, *kinds: str) -> list[str]:
    """
    Return the files of the products of the given kinds of the run of a
    fit (to resume it or to bootstrap it: the action).

    Raises
    ------
    ConfigError
        If the directory is not the run of a fit, or it has not one of
        each of the products, or their files are missing.
    """
    try:
        run = gbkfit.outputs.open(output_dir)
        if run.task != 'fit':
            raise ValueError(f"it is a run of the task {run.task!r}")
        files = [run.filename(run.product(kind)) for kind in kinds]
    except (ValueError, LookupError) as e:
        raise ConfigError(f"cannot {action} the fit in {output_dir}: {e}") \
            from e
    if missing := [f for f in files if not os.path.isfile(f)]:
        raise ConfigError(
            f"cannot {action} the fit in {output_dir}: missing {missing}")
    return files


def _read_config(config: str) -> _detail.ConfigFile:
    """Read the configuration of a fit."""
    _log.info(f"reading configuration from file: {config}...")
    return _detail.read_config(
        config, ('galaxies', 'observations', 'params', 'fitter'),
        ('driver', 'drivers', 'lines', 'bands', 'pdescs', 'depth_order',
         'parallel', 'bootstrap'))


class _Setup(NamedTuple):
    """
    What a fit is set up from its configuration: its drivers, its
    observations and objective, its fitter and its parameters.
    """
    drivers: dict[str | None, gbkfit.driver.Driver]
    group: gbkfit.observation.ObservationGroup
    objective: gbkfit.objective.Objective
    fitter: gbkfit.fitting.Fitter
    params: gbkfit.params.ParamSpace


def _set_up(cfg: dict[str, Any], root: str) -> _Setup:
    """
    Set up a fit from its configuration: its observations and objective,
    its fitter and its parameters, checked for the fitter.
    """
    with parseutils.paths_relative_to(root):
        _log.info("setting up galaxies and observations...")
        drivers = _detail.load_drivers(cfg)
        group = _detail.load_observation_group(cfg, drivers)
        _log.info("setting up objective...")
        objective = gbkfit.objective.Objective(group)
        _log.info("setting up fitter...")
        with parseutils.config_path('fitter'):
            fitter = gbkfit.fitting.fitter_parser.load(cfg['fitter'])
        _log.info("setting up params...")
        params = _detail.load_params(
            cfg, objective.pdescs(), objective.constants(),
            group.plan_functions())
        # Check the free properties for the fitter, and the priors on
        # several parameters, before the fit starts
        with parseutils.config_path('params', 'properties'):
            fitter.load_fit_params(params)
        with parseutils.config_path('params'):
            gbkfit.fitting.joint_prior_residuals(params)
    return _Setup(drivers, group, objective, fitter, params)


def _load_bootstrap(
        cfg: dict[str, Any], setup: _Setup
) -> gbkfit.fitting.BootstrapOptions | None:
    """
    Load the options of the bootstrap of a configuration (its section
    'bootstrap'), if it has them, checked for a fit.
    """
    if 'bootstrap' not in cfg:
        return None
    with parseutils.config_path('bootstrap'):
        options = gbkfit.fitting.BootstrapOptions.load(cfg['bootstrap'])
        options.check(setup.objective, setup.params)
    return options


def _bootstrap(
        setup: _Setup,
        start: dict[str, float],
        options: gbkfit.fitting.BootstrapOptions
) -> gbkfit.fitting.Bootstrap:
    """Bootstrap a fit from its best fit (the value of every element)."""
    _log.info(f"bootstrap started ({options.nsamples()} refits)")
    begin = time.perf_counter()
    boot = gbkfit.fitting.bootstrap(
        setup.objective, setup.params, start, setup.fitter, options)
    _log.info(
        f"bootstrap completed ({len(boot.failures)} refits failed; elapsed "
        f"time: {time.perf_counter() - begin:.1f} s)")
    return boot


def _bootstrap_summary(
        solution: dict[str, Any], boot: gbkfit.fitting.Bootstrap
) -> str:
    """
    Return a summary of the bootstrap of a solution (its summary in
    result.json), for humans: the best fit of each free element and
    derived value, and its standard deviation and percentiles over the
    refits.
    """
    std = boot.std()
    quantiles = boot.summary()
    lines = [
        f"{'name':<16} {'best':>12} {'bootstrap std':>13} "
        f"{'bootstrap 16/50/84':>38}"]
    # (the free elements decoded, as the bootstrap gives them)
    free = {name: solution['elements'][name] for name in solution['free']}
    for name, value in (free | solution['derived']).items():
        q = quantiles.get(name, (math.nan,) * 3)
        lines.append(
            f"{name:<16} {value:>12.6g} {std.get(name, math.nan):>13.6g} "
            f"{q[0]:>12.6g} {q[1]:>12.6g} {q[2]:>12.6g}")
    return '\n'.join(lines)


def _fit(
        cfg: dict[str, Any],
        root: str,
        run: gbkfit.outputs.RunWriter,
        log: gbkfit.fitting.EvaluationLog | None
) -> None:
    """Fit a configuration, and write the outputs to its run (see fit)."""
    setup = _set_up(cfg, root)
    group, objective, fitter, params = \
        setup.group, setup.objective, setup.fitter, setup.params
    with parseutils.paths_relative_to(root):
        pool = _detail.load_pool(cfg, objective, setup.drivers)
    options = _load_bootstrap(cfg, setup)

    _log.info("model-fitting started")

    def write_best(free: dict[str, float]) -> None:
        # (the free values of the progress are coded for the parameters
        # with a mode; the configuration starts at those of the elements)
        elements = params.element_values(params.evaluate(free))
        _write_config(
            run, 'fit/best.yaml', 'best_config', cfg, root, params, elements)

    progress = _progress.JsonLinesProgress(
        run.filename('logs/progress.jsonl'))
    run.record('logs/progress.jsonl', 'progress', 'jsonl')
    best = _progress.BestValues(write_best)
    control = gbkfit.fitting.FitControl()
    with pool or contextlib.nullcontext(), _stop_on_interrupt(control), \
            _progress.console_progress(_log) as console:
        result = fitter.fit(
            objective, params, (console, progress, best), control, log, pool)
    _log.info(f"model-fitting completed (elapsed time: {result.time:.1f} s)")

    _log.info(f"saving result under '{run.path}'...")
    result.write(run)
    outputs = _detail.data_outputs(group) | _detail.evaluation_outputs(
        group, objective, result.best().values, True, True)
    _detail.write_outputs(run, outputs)
    _write_config(
        run, 'fit/restart.yaml', 'restart_config', cfg, root, params,
        result.best().elements)
    if options is not None:
        result = dataclasses.replace(result, bootstrap=_bootstrap(
            setup, result.best().elements, options))
        result.write(run)
    _log.info(f"result:\n{result.summary()}")


@contextlib.contextmanager
def _stop_on_interrupt(control: gbkfit.fitting.FitControl) -> Iterator[None]:
    """
    While the block runs, make the first Ctrl-C (SIGINT) stop the fit at
    its next evaluation, and a second one quit at once (only in the main
    thread, which receives the signals).
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def stop(signum: int, frame: Any) -> None:
        signal.signal(signal.SIGINT, previous)
        _log.warning(
            "stopping the fit at its next evaluation (Ctrl-C again to quit "
            "at once)...")
        control.stop()

    previous = signal.signal(signal.SIGINT, stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def _write_config(
        run: gbkfit.outputs.RunWriter,
        path: str,
        kind: str,
        cfg: dict[str, Any],
        root: str,
        params: gbkfit.params.ParamSpace,
        values: dict[str, float]
) -> None:
    """
    Write the configuration with the free parameters starting at the given
    values (by element name) as a product of a run (see
    _detail.write_config).
    """
    try:
        config = params.with_values(values).dump()
    except ConfigError as e:
        _log.warning(f"{path} is not written: {e}")
        return
    _detail.write_config(
        run, path, kind, cfg | dict(params=numutils.nativify(config)), root)
