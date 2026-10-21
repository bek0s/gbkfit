"""
The eval task: evaluate the galaxies of a configuration, or its objective
(the residuals against the data), or draw mock data of its observations,
and write the outputs. Programs that evaluate a configuration themselves
(e.g. at other values of its parameters) set it up with load.
"""

import logging
import os
from typing import Any, Literal, NamedTuple

import pandas as pd

from gbkfit.objective import Objective
import gbkfit.outputs
from gbkfit.observation import (
    Likelihood, Noise, ObservationGroup, likelihood_parser, noise_parser)
from gbkfit.params import ParamSpace, ParamValues
from gbkfit.utils import iterutils, numutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _detail
from ._detail import Output


__all__ = [
    'Evaluation',
    'Output',
    'eval_',
    'load'
]


_log = logging.getLogger(__name__)


class Evaluation(NamedTuple):
    """
    What evaluates a configuration (see load): the observations of its
    galaxies, its parameters, and its objective, if every observation
    has data (else None).
    """
    group: ObservationGroup
    params: ParamSpace
    objective: Objective | None

    def outputs(
            self,
            values: ParamValues,
            data: bool = False,
            extra: bool = False
    ) -> dict[str, Output]:
        """
        Return the outputs of an evaluation, as the task writes them (see
        eval_), by their paths in a run.

        Parameters
        ----------
        values : ParamValues
            The values of the parameters (see ParamSpace.evaluate).
        data : bool, optional
            Whether to add the data of the observations ('data', 'error'
            and 'data_image'), which do not change with the values.
        extra : bool, optional
            Whether to add the extra outputs of the models ('extra'; e.g.
            their cubes at high resolution).

        Returns
        -------
        dict
            The outputs: the models ('model', and their images,
            'model_image'), and with the objective the residuals too
            ('residual', 'wresidual' and 'residual_image').
        """
        residuals = self.objective is not None
        outputs = _detail.evaluation_outputs(
            self.group, self.objective, values, True, residuals, extra)
        if data:
            outputs |= _detail.data_outputs(self.group)
        return outputs


class _Mock(NamedTuple):
    """The options of the mock data (the section 'mock'; see eval_)."""
    noises: list[Noise]
    likelihoods: list[Likelihood | None]
    nmocks: int
    nsamples: int
    seed: int | None


def eval_(
        mode: Literal['model', 'objective', 'mock'],
        config: str,
        profile_iters: int,
        output_dir: str | None,
        output_dir_mode: Literal['terminate', 'overwrite', 'unique']
) -> None:
    """
    Evaluate the galaxies of a configuration at its parameters (their
    fixed values, and the 'value' of the free ones), or its objective,
    and write the outputs as a run (see gbkfit.outputs; its option 'mode'
    is the mode): the configuration (config.yaml, kind 'config'), the
    parameters and the derived values (params.json, 'params'), the
    models and their extra outputs, and in the mode 'objective' the data
    and the residuals too (see data_outputs and evaluation_outputs), and
    the log (logs/gbkfit.log, 'log'; see run_log); in a project, the
    sources of the configuration too (setup.json, 'setup'; see
    write_setup).

    The mode 'mock' writes the models (as 'model'), and mock data of the
    observations: their models measured from what their instruments
    record of the models plus noise, with their errors (see
    ObservationGroup.mock). Each mock k is the files of the data of each
    observation (in mocks/<k>/observations/<observation>/, kind
    'mock_data') and a configuration that fits them (mocks/<k>/
    config.yaml, 'mock_config'): the configuration with each observation
    of its mock data. The section 'mock' of the configuration has the
    options

    - noise: the noise of every observation, or a list of one for each
      (see Noise);
    - nmocks (1), nsamples (100) and seed (none): the number of mocks,
      the number of draws of the errors of measured data (e.g. moments),
      and the seed of the random numbers (see ObservationGroup.mock);
    - likelihood (optional): the likelihood of the mock data of every
      observation, or a list of one (or null) for each, in the mock
      configurations; by default that of its data, if it has data, or
      that of data without one (see Observation).

    The mock data are in the units of the models.

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
        The number of evaluations to time (timings.json, kind 'timings').
    output_dir : str or None
        The directory of the run; by default (None), the folder of the
        runs of the setup in its project (runs/<name of the setup>; see
        gbkfit.project), or else output.
    output_dir_mode : {'terminate', 'overwrite', 'unique'}
        What to do if the directory exists (see gbkfit.outputs.create).

    Raises
    ------
    ConfigError
        If the configuration is invalid, or the mode is 'mock' and it has
        no section 'mock'.
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
        _eval(mode, cfg, root, profile_iters, run)


def load(config: str) -> Evaluation:
    """
    Read a configuration file, and set up the observations of its galaxies,
    its parameters, and its objective if every observation has data, to
    evaluate them (e.g. at other values of the parameters; see
    ParamSpace.evaluate, Evaluation.outputs and ObservationGroup.model_h).

    Parameters
    ----------
    config : str
        The configuration file (see eval_), or that of a fit (see
        gbkfit.tasks.fit.fit; its sections of the fit are left out).

    Returns
    -------
    Evaluation
        The observations, the parameters and the objective.

    Raises
    ------
    ConfigError
        If the configuration is invalid.
    """
    file = _read_config(config, _OPTIONAL + _FIT)
    with parseutils.paths_relative_to(file.root):
        group, params = _set_up(file.config)
    objective = None
    if all(obs.data() is not None for obs in group.observations()):
        _log.info("setting up objective...")
        objective = Objective(group)
    return Evaluation(group, params, objective)


# The sections of the configurations of the task, and those of fits, which
# load reads too (without them)
_REQUIRED = ('galaxies', 'observations', 'params')
_OPTIONAL = (
    'driver', 'drivers', 'lines', 'bands', 'pdescs', 'depth_order', 'mock')
_FIT = ('fitter', 'parallel', 'bootstrap')


def _read_config(
        config: str, optional: tuple[str, ...] = _OPTIONAL
) -> _detail.ConfigFile:
    """Read the configuration of an evaluation."""
    _log.info(f"reading configuration from file: {config}...")
    return _detail.read_config(config, _REQUIRED, optional)


def _set_up(cfg: dict[str, Any]) -> tuple[ObservationGroup, ParamSpace]:
    """
    Set up the observations and the parameters of a configuration (with
    the paths relative to its root).
    """
    _log.info("setting up galaxies and observations...")
    group = _detail.load_observation_group(cfg)
    _log.info("setting up params...")
    params = _detail.load_params(
        cfg, group.pdescs(), group.constants(), group.plan_functions())
    return group, params


def _eval(
        mode: Literal['model', 'objective', 'mock'],
        cfg: dict[str, Any],
        root: str,
        profile_iters: int,
        run: gbkfit.outputs.RunWriter
) -> None:
    """Evaluate a configuration, and write the outputs (see eval_)."""
    with parseutils.paths_relative_to(root):
        group, params = _set_up(cfg)
        objective = None
        if mode == 'objective':
            _log.info("setting up objective...")
            objective = Objective(group)
        mock = None
        if mode == 'mock':
            with parseutils.config_path('mock'):
                mock = _load_mock(cfg['mock'], group.nobservations())

    _log.info("calculating model parameters...")
    param_values = params.evaluate()
    run.write('params.json', 'params', dict(
        params=param_values, eparams=params.element_values(param_values),
        derived=params.derived_values(param_values)))

    _log.info("evaluating...")
    outputs = _detail.evaluation_outputs(
        group, objective, param_values, True, mode == 'objective')
    if objective is not None:
        outputs |= _detail.data_outputs(group)
        _log.info(f"chi-square: {objective.chi2(param_values)}")
    _log.info("storing outputs to the filesystem...")
    _detail.write_outputs(run, outputs)
    if mock is not None:
        _log.info(f"drawing {mock.nmocks} mock datasets...")
        _write_mocks(cfg, root, group, param_values, mock, run)

    if profile_iters > 0:
        _log.info("running performance test...")
        # Time the profiled evaluations only (the first evaluation, above,
        # also warmed up the drivers)
        group.timers().clear()
        for _ in range(profile_iters):
            if objective is None:
                group.model_d(param_values)
            else:
                objective.chi2(param_values)
        time_stats = numutils.nativify({
            name: stats.to_dict()
            for name, stats in group.timers().stats().items()})
        _log.info(pd.DataFrame.from_dict(time_stats, orient='index'))
        run.write('timings.json', 'timings', time_stats)


def _load_mock(info: dict[str, Any], nobservations: int) -> _Mock:
    """
    Load the options of the mock data (see eval_), with a noise and a
    likelihood (or None) for each of the observations.
    """
    if not iterutils.is_mapping(info):
        raise ConfigError("section 'mock' must be a dictionary")
    options = parseutils.parse_options(
        info, required={'noise'},
        optional={'likelihood', 'nmocks', 'nsamples', 'seed'})

    def per_observation(key: str, parser: parseutils.Parser) -> list[Any]:
        value = options.get(key)
        if not iterutils.is_sequence(value):
            value = [value] * nobservations
        if len(value) != nobservations:
            raise ConfigError(
                f"{key} is one, or a list of one for each of the "
                f"{nobservations} observations; it is a list of "
                f"{len(value)}")
        with parseutils.config_path(key):
            return [
                None if item is None else parser.load(item) for item in value]

    noises = per_observation('noise', noise_parser)
    if None in noises:
        raise ConfigError("every observation needs a noise")
    return _Mock(
        noises, per_observation('likelihood', likelihood_parser),
        options.get('nmocks', 1), options.get('nsamples', 100),
        options.get('seed'))


def _write_mocks(
        cfg: dict[str, Any],
        root: str,
        group: ObservationGroup,
        values: ParamValues,
        mock: _Mock,
        run: gbkfit.outputs.RunWriter
) -> None:
    """
    Draw the mock data of the observations of a group at the parameter
    values, and write their files and configurations (see eval_).
    """
    with parseutils.config_path('mock'):
        datasets = group.mock(
            values, mock.noises, mock.nmocks, mock.nsamples, mock.seed)
    sections = {k: v for k, v in cfg.items() if k != 'mock'}
    for k in range(mock.nmocks):
        observations = []
        for i, (observation, info) in enumerate(zip(
                group.observations(), cfg['observations'])):
            name = group.observation_label(i)
            folder = f'mocks/{k}/observations/{name}'
            directory = os.path.join(run.path, *folder.split('/'))
            os.makedirs(directory, exist_ok=True)
            with parseutils.config_path('mock'):
                mocked = observation.with_data(
                    datasets[i][k], mock.likelihoods[i])
            # (the paths of the files of the data are relative to the
            # root of the configuration, as its other paths)
            prefix = directory + os.sep
            mocked_info = _rebased(
                mocked.dump(prefix), prefix,
                os.path.relpath(directory, root) + os.sep)
            for filename in sorted(os.listdir(directory)):
                run.record(
                    f'{folder}/{filename}', 'mock_data',
                    os.path.splitext(filename)[1].lstrip('.'),
                    mock=str(k), observation=name)
            # (the driver of an observation is the name of one of those
            # of the configuration)
            driver = {} if info.get('driver') is None \
                else dict(driver=info['driver'])
            observations.append(driver | mocked_info)
        _detail.write_config(
            run, f'mocks/{k}/config.yaml', 'mock_config',
            sections | dict(observations=observations), root, mock=str(k))


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
