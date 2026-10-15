"""
The fit task: fit the galaxies of a configuration to its data, and write
the result.
"""

import logging
import os
from typing import Any, Literal

import gbkfit.fitting
import gbkfit.objective
import gbkfit.params
from gbkfit.utils import numutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _detail, _progress


_log = logging.getLogger(__name__)


def fit(
        config: str,
        output_dir: str,
        output_dir_mode: Literal['terminate', 'overwrite', 'unique']
) -> None:
    """
    Fit the galaxies of a configuration to its data, and write the result
    (see FitResult.dump), the data and the models and residuals at the
    best fit (FITS; see the plot task), and restart.yaml: the
    configuration with the free parameters starting at the best fit, to
    fit again from there (e.g. with another fitter). As it fits, it shows
    its progress on the console, writes every report of it to
    progress.jsonl, and the configuration at the best fit so far to
    best.yaml (see _progress), from which a stopped fit can start again.

    Parameters
    ----------
    config : str
        The configuration file (YAML or JSON); its relative paths are
        relative to its directory (or to its section 'root').
    output_dir : str
        The output directory.
    output_dir_mode : {'terminate', 'overwrite', 'unique'}
        What to do if the output directory exists (see make_output_dir).

    Raises
    ------
    ConfigError
        If the configuration is invalid.
    """
    _log.info(f"reading configuration from file: {config}...")
    cfg, root = _detail.read_config(
        config, ('galaxies', 'observations', 'params', 'fitter'),
        ('driver', 'drivers', 'lines', 'bands', 'pdescs', 'depth_order'))
    output_dir = _detail.make_output_dir(output_dir, output_dir_mode)
    _log.info(f"output will be stored under directory: {output_dir}")

    with parseutils.paths_relative_to(root):
        _log.info("setting up galaxies and observations...")
        group = _detail.load_observation_group(cfg)
        _log.info("setting up objective...")
        objective = gbkfit.objective.Objective(group)
        _log.info("setting up fitter...")
        with parseutils.config_path('fitter'):
            fitter = gbkfit.fitting.fitter_parser.load(cfg['fitter'])
        _log.info("setting up params...")
        params = _detail.load_params(
            cfg, objective.pdescs(), objective.constants())
        # Check the free properties for the fitter before the fit starts
        with parseutils.config_path('params', 'properties'):
            fitter.load_fit_params(params)

    _log.info("model-fitting started")

    def write_best(values: dict[str, float]) -> None:
        _write_config(
            cfg, root, params, values, os.path.join(output_dir, 'best.yaml'))

    progress = _progress.JsonLinesProgress(
        os.path.join(output_dir, 'progress.jsonl'))
    best = _progress.BestValues(write_best)
    with _progress.console_progress(_log) as console:
        result = fitter.fit(objective, params, (console, progress, best))
    _log.info(f"model-fitting completed (elapsed time: {result.time:.1f} s)")

    _log.info(f"saving result under '{output_dir}'...")
    result.dump(output_dir)
    outputs = _detail.data_outputs(group) | _detail.evaluation_outputs(
        group, objective, result.best().values, True, True)
    _detail.write_outputs(output_dir, outputs, 'gbkfit_fit_extra')
    _write_restart(cfg, root, params, result, output_dir)
    _log.info(f"result:\n{result.summary()}")


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
