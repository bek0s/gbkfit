"""
The eval task: evaluate the galaxies of a configuration, or its objective
(the residuals against the data), or draw mock data of its observations,
and write the outputs.
"""

import logging
import os
from typing import Any, Literal, NamedTuple

import pandas as pd

import gbkfit.objective
from gbkfit.observation import (
    Likelihood, Noise, ObservationGroup, likelihood_parser, noise_parser)
from gbkfit.params import ParamValues
from gbkfit.utils import iterutils, numutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _detail


_log = logging.getLogger(__name__)


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
        output_dir: str,
        output_dir_mode: Literal['terminate', 'overwrite', 'unique']
) -> None:
    """
    Evaluate the galaxies of a configuration at its parameters (their
    fixed values, and the 'value' of the free ones), or its objective,
    and write the outputs: the parameters (gbkfit_eval_params), the
    models, or the data and the residuals (FITS), and their extra
    outputs.

    The mode 'mock' writes the models (as 'model'), and mock data of the
    observations: their models measured from what their instruments
    record of the models plus noise, with their errors (see
    ObservationGroup.mock). Each mock is the files of the data of each
    observation (mock<k>_<observation prefix>...) and a configuration
    that fits them, mock<k>.yaml: the configuration with each
    observation of its mock data. The section 'mock' of the
    configuration has the options

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
        The configuration file (YAML or JSON); its relative paths are
        relative to its directory (or to its section 'root').
    profile_iters : int
        The number of evaluations to time (gbkfit_eval_timings).
    output_dir : str
        The output directory.
    output_dir_mode : {'terminate', 'overwrite', 'unique'}
        What to do if the output directory exists (see make_output_dir).

    Raises
    ------
    ConfigError
        If the configuration is invalid, or the mode is 'mock' and it has
        no section 'mock'.
    """
    _log.info(f"reading configuration from file: {config}...")
    cfg, root = _detail.read_config(
        config, ('galaxies', 'observations', 'params'),
        ('driver', 'drivers', 'lines', 'bands', 'pdescs', 'depth_order',
         'mock'))
    if mode == 'mock' and 'mock' not in cfg:
        raise ConfigError("the mode 'mock' needs the section 'mock'")
    output_dir = _detail.make_output_dir(output_dir, output_dir_mode)
    _log.info(f"output will be stored under directory: {output_dir}")

    with parseutils.paths_relative_to(root):
        _log.info("setting up galaxies and observations...")
        group = _detail.load_observation_group(cfg)
        objective = None
        if mode == 'objective':
            _log.info("setting up objective...")
            objective = gbkfit.objective.Objective(group)
        _log.info("setting up params...")
        params = _detail.load_params(
            cfg, group.pdescs(), group.constants(), group.plan_functions())
        mock = None
        if mode == 'mock':
            with parseutils.config_path('mock'):
                mock = _load_mock(cfg['mock'], group.nobservations())

    _log.info("calculating model parameters...")
    param_values = params.evaluate()
    _detail.dump_dict(
        numutils.nativify(dict(
            params=param_values,
            eparams=params.element_values(param_values))),
        os.path.join(output_dir, 'gbkfit_eval_params'))

    _log.info("evaluating...")
    outputs = _detail.evaluation_outputs(
        group, objective, param_values, mode != 'objective',
        mode == 'objective')
    if objective is not None:
        outputs |= _detail.data_outputs(group)
        _log.info(f"chi-square: {objective.chi2(param_values)}")
    _log.info("storing outputs to the filesystem...")
    _detail.write_outputs(output_dir, outputs, 'gbkfit_eval_extra')
    if mock is not None:
        _log.info(f"drawing {mock.nmocks} mock datasets...")
        _write_mocks(cfg, root, group, param_values, mock, output_dir)

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
        _detail.dump_dict(
            time_stats, os.path.join(output_dir, 'gbkfit_eval_timings'))



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
        output_dir: str
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
            prefix = os.path.join(
                output_dir, f'mock{k}_{group.extra_prefix(i)}')
            with parseutils.config_path('mock'):
                mocked = observation.with_data(
                    datasets[i][k], mock.likelihoods[i])
            # (the driver of an observation is the name of one of those
            # of the configuration)
            driver = {} if info.get('driver') is None \
                else dict(driver=info['driver'])
            observations.append(driver | mocked.dump(prefix))
        # (relative paths stay relative to the directory of the
        # configuration, so that it runs from anywhere)
        _detail.write_yaml(
            dict(root=root) | sections | dict(observations=observations),
            os.path.join(output_dir, f'mock{k}.yaml'))
