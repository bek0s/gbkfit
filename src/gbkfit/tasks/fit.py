"""
The fit task: fit the galaxies of a configuration to its data, and write
the result.
"""

import contextlib
import copy
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
import gbkfit.observation
import gbkfit.params
from gbkfit.utils import numutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _detail, _progress


_log = logging.getLogger(__name__)


# The files of a fit in its output directory, to resume it from (and to
# bootstrap it)
_CONFIG = 'config.yaml'
_EVALUATIONS = 'evaluations.bin'
_RESULT = 'result.json'


def fit(
        config: str,
        output_dir: str,
        output_dir_mode: Literal['terminate', 'overwrite', 'unique'],
        evaluation_log: bool = True
) -> None:
    """
    Fit the galaxies of a configuration to its data, and write the result
    (see FitResult.dump), the data and the models and residuals at the
    best fit (FITS; see the plot task), and restart.yaml: the
    configuration with the free parameters starting at the best fit, to
    fit again from there (e.g. with another fitter). With a section
    'parallel' ({drivers: [names]}), the batches of points of the fitter
    are evaluated at once on copies of the objective on those drivers
    (see EvaluationPool), e.g. one per GPU. As it fits, it shows its
    progress on the console, writes every report of it to
    progress.jsonl, and the configuration at the best fit so far to
    best.yaml (see _progress), from which a stopped fit can start again.

    With a section 'bootstrap' (see BootstrapOptions), the errors of the
    best fit are then bootstrapped too (see gbkfit.fitting.bootstrap),
    after the outputs of the fit are written, which are written again
    with the bootstrap (see FitResult.dump); a fit can be bootstrapped
    later as well (see bootstrap).

    Ctrl-C stops the fit, which ends with the best point so far (and the
    chain so far of an MCMC sampler; see Fitter.fit); a second Ctrl-C
    quits at once. The configuration (config.yaml) and, unless turned
    off, the log of the evaluations (evaluations.bin; see EvaluationLog)
    go to the output directory, from which a stopped or killed fit
    resumes (see resume).

    Parameters
    ----------
    config : str
        The configuration file (YAML or JSON); its relative paths are
        relative to its directory (or to its section 'root').
    output_dir : str
        The output directory.
    output_dir_mode : {'terminate', 'overwrite', 'unique'}
        What to do if the output directory exists (see make_output_dir).
    evaluation_log : bool, optional
        Whether to log the evaluations, to resume from (true by default).

    Raises
    ------
    ConfigError
        If the configuration is invalid.
    """
    cfg, root = _read_config(config)
    output_dir = _detail.make_output_dir(output_dir, output_dir_mode)
    _log.info(f"output will be stored under directory: {output_dir}")
    _detail.write_yaml(
        dict(root=root) | copy.deepcopy(cfg),
        os.path.join(output_dir, _CONFIG))
    log = gbkfit.fitting.EvaluationLog(
        os.path.join(output_dir, _EVALUATIONS)) if evaluation_log else None
    _fit(cfg, root, output_dir, log)


def resume(output_dir: str) -> None:
    """
    Resume a fit that stopped or was killed, in its output directory: run
    it again from its configuration (config.yaml), answering the fitter
    from the log of its evaluations (evaluations.bin) while it lasts (see
    EvaluationLog), and write its outputs there (see fit). The fit must
    have had the log; a least-squares fit, which has nothing in it, starts
    again (fit best.yaml instead, to start from its best point).

    Parameters
    ----------
    output_dir : str
        The output directory of the fit.

    Raises
    ------
    ConfigError
        If the directory has no configuration or log of a fit, or the
        configuration is invalid.
    """
    paths = [
        os.path.join(output_dir, name) for name in (_CONFIG, _EVALUATIONS)]
    if missing := [path for path in paths if not os.path.isfile(path)]:
        raise ConfigError(
            f"cannot resume the fit in {output_dir}: missing {missing}")
    cfg, root = _read_config(paths[0])
    try:
        log = gbkfit.fitting.EvaluationLog.resume(paths[1])
    except ValueError as e:
        raise ConfigError(str(e)) from e
    _fit(cfg, root, output_dir, log)


def bootstrap(output_dir: str, config: str) -> None:
    """
    Bootstrap the errors of a fit in its output directory (see
    gbkfit.fitting.bootstrap): rebuild its objective from its
    configuration (config.yaml), refit from its best fit (result.json)
    by the section 'bootstrap' of a configuration (see BootstrapOptions;
    the refits use the fitter of the fit unless it gives one), and add
    the bootstrap to the result (the entry 'bootstrap' of result.json,
    and bootstrap.csv; see FitResult.dump), and the configuration, as
    bootstrap.yaml.

    Parameters
    ----------
    output_dir : str
        The output directory of the fit.
    config : str
        The configuration file of the bootstrap (YAML or JSON), with only
        the section 'bootstrap'.

    Raises
    ------
    ConfigError
        If the directory has no configuration or result of a fit, or a
        configuration is invalid.
    """
    paths = [
        os.path.join(output_dir, name) for name in (_CONFIG, _RESULT)]
    if missing := [path for path in paths if not os.path.isfile(path)]:
        raise ConfigError(
            f"cannot bootstrap the fit in {output_dir}: missing {missing}")
    cfg, root = _read_config(paths[0])
    _log.info(f"reading the configuration of the bootstrap: {config}...")
    bootstrap_cfg, _ = _detail.read_config(config, ('bootstrap',))
    setup = _set_up(cfg, root)
    options = _load_bootstrap(bootstrap_cfg, setup)
    with open(paths[1]) as f:
        result = json.load(f)
    best = result['solutions'][0]
    boot = _bootstrap(setup, best['elements'], options)
    result['bootstrap'] = boot.info()
    with open(paths[1], 'w') as f:
        json.dump(numutils.nativify(result), f, indent=2)
    boot.dump(output_dir)
    _detail.write_yaml(
        bootstrap_cfg, os.path.join(output_dir, 'bootstrap.yaml'))
    _log.info(f"bootstrap:\n{_bootstrap_summary(best, boot)}")


def _read_config(config: str) -> tuple[dict[str, Any], str]:
    """Read the configuration of a fit, and its root."""
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
    for name, value in (solution['free'] | solution['derived']).items():
        q = quantiles.get(name, (math.nan,) * 3)
        lines.append(
            f"{name:<16} {value:>12.6g} {std.get(name, math.nan):>13.6g} "
            f"{q[0]:>12.6g} {q[1]:>12.6g} {q[2]:>12.6g}")
    return '\n'.join(lines)


def _fit(
        cfg: dict[str, Any],
        root: str,
        output_dir: str,
        log: gbkfit.fitting.EvaluationLog | None
) -> None:
    """Fit a configuration, and write the outputs (see fit)."""
    setup = _set_up(cfg, root)
    group, objective, fitter, params = \
        setup.group, setup.objective, setup.fitter, setup.params
    with parseutils.paths_relative_to(root):
        pool = _detail.load_pool(cfg, objective, setup.drivers)
    options = _load_bootstrap(cfg, setup)

    _log.info("model-fitting started")

    def write_best(values: dict[str, float]) -> None:
        _write_config(
            cfg, root, params, values, os.path.join(output_dir, 'best.yaml'))

    progress = _progress.JsonLinesProgress(
        os.path.join(output_dir, 'progress.jsonl'))
    best = _progress.BestValues(write_best)
    control = gbkfit.fitting.FitControl()
    with pool or contextlib.nullcontext(), _stop_on_interrupt(control), \
            _progress.console_progress(_log) as console:
        result = fitter.fit(
            objective, params, (console, progress, best), control, log, pool)
    _log.info(f"model-fitting completed (elapsed time: {result.time:.1f} s)")

    _log.info(f"saving result under '{output_dir}'...")
    result.dump(output_dir)
    outputs = _detail.data_outputs(group) | _detail.evaluation_outputs(
        group, objective, result.best().values, True, True)
    _detail.write_outputs(output_dir, outputs, 'gbkfit_fit_extra')
    _write_restart(cfg, root, params, result, output_dir)
    if options is not None:
        result = dataclasses.replace(result, bootstrap=_bootstrap(
            setup, result.best().elements, options))
        result.dump(output_dir)
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


def _write_restart(
        cfg: dict[str, Any],
        root: str,
        params: gbkfit.params.ParamSpace,
        result: gbkfit.fitting.FitResult,
        output_dir: str
) -> None:
    """
    Write restart.yaml: the configuration with the free parameters
    starting at the best fit (see _write_config).
    """
    _write_config(
        cfg, root, params, result.best().elements,
        os.path.join(output_dir, 'restart.yaml'))


def _write_config(
        cfg: dict[str, Any],
        root: str,
        params: gbkfit.params.ParamSpace,
        values: dict[str, float],
        filename: str
) -> None:
    """
    Write the configuration with the free parameters starting at the given
    values (by element name), and its directory as its root (so that it
    runs from anywhere): to a temporary file first, then renamed, so that
    the file is never half written.
    """
    try:
        config = params.with_values(values).dump()
    except ConfigError as e:
        _log.warning(f"{os.path.basename(filename)} is not written: {e}")
        return
    temporary = filename + '.tmp'
    _detail.write_yaml(
        dict(root=root) | cfg | dict(params=numutils.nativify(config)),
        temporary)
    os.replace(temporary, filename)
