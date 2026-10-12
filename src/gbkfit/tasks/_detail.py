"""
Helpers shared by the tasks: reading configurations, the output
directory, loading the parts of a configuration, and writing outputs.
"""

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

import astropy.io.fits as fits
import numpy as np
import ruamel.yaml

import gbkfit.model
import gbkfit.observation
from gbkfit.driver import Driver, driver_parser
from gbkfit.objective import Objective
from gbkfit.observation import ObservationGroup
from gbkfit.params import (
    ParamDesc, ParamSpace, ParamValues, load_pdescs_dict, param_space_parser)
from gbkfit.utils import (
    fitsutils, gridutils, iterutils, miscutils, numutils, parseutils)
from gbkfit.utils.parseutils import ConfigError


_log = logging.getLogger(__name__)

# Reads and writes YAML (round trip: dicts keep their order)
_yaml = ruamel.yaml.YAML()


def read_config(
        filename: str,
        required: tuple[str, ...],
        optional: tuple[str, ...] = ()
) -> tuple[dict[str, Any], str]:
    """
    Read a configuration file (YAML or JSON), and check its sections (see
    prepare_config).

    Parameters
    ----------
    filename : str
        The file.
    required, optional : tuple of str
        The names of the required and the optional sections.

    Returns
    -------
    dict
        The configuration.
    str
        The directory its relative paths are relative to: its optional
        section 'root' (itself relative to the file), or by default the
        directory of the file.

    Raises
    ------
    ConfigError
        If the file cannot be read, or its sections are invalid.
    """
    filename = os.path.abspath(filename)
    try:
        with open(filename) as f:
            cfg = _yaml.load(f)
    except (OSError, ruamel.yaml.YAMLError) as e:
        raise ConfigError(
            f"cannot read the configuration file {filename}: {e}") from e
    root = cfg.pop('root', None) if isinstance(cfg, dict) else None
    if root is not None and not isinstance(root, str):
        raise ConfigError(
            f"section 'root' must be a directory; it is {root!r}")
    directory = os.path.dirname(filename)
    if root is not None:
        directory = os.path.normpath(os.path.join(directory, root))
    return prepare_config(cfg, required, optional), directory


def prepare_config(
        config: dict[str, Any] | None,
        required: tuple[str, ...] = (),
        optional: tuple[str, ...] = ()
) -> dict[str, Any]:
    """
    Check the sections of a configuration, and return it as plain JSON
    values: unknown sections are reported (see parseutils.report_unknown)
    and left out, null sections are empty, and the sections 'galaxies' and
    'observations' are lists.

    Raises
    ------
    ConfigError
        If a required section is missing, or a section is of the wrong
        type.
    """
    config = {} if config is None else dict(config)
    known = required + optional
    if unknown := [s for s in config if s not in known]:
        parseutils.report_unknown(
            "unknown configuration sections", unknown, known)
    config = {s: config[s] for s in config if s in known}
    if missing := [s for s in required if s not in config]:
        raise ConfigError(
            f"the configuration needs the sections {missing}")
    # (empty sections are loaded as such, for clearer messages)
    config = {s: {} if v is None else v for s, v in config.items()}
    for s in ('driver', 'drivers', 'lines', 'bands', 'pdescs', 'params',
              'fitter'):
        if s in config and not iterutils.is_mapping(config[s]):
            raise ConfigError(f"section '{s}' must be a dictionary")
    for s in ('galaxies', 'observations'):
        if s in config:
            if not iterutils.is_sequence_or_mapping(config[s]):
                raise ConfigError(
                    f"section '{s}' must be a dictionary or a list")
            config[s] = iterutils.listify(config[s])
    return json.loads(json.dumps(config))


def make_output_dir(
        path: str,
        mode: Literal['terminate', 'overwrite', 'unique']
) -> str:
    """
    Create the output directory at the given path and return its absolute
    path. If it exists: an error (terminate), use it (overwrite), or create
    the first free path with a numeric suffix (unique, e.g. 'out_2').
    Creating a directory is atomic, so two runs cannot both create the
    same one.
    """
    path_obj = Path(path).absolute()
    try:
        path_obj.mkdir(parents=True)
        return str(path_obj)
    except FileExistsError:
        pass
    if not path_obj.is_dir():
        raise RuntimeError(f"path '{str(path_obj)}' exists as a file")
    if mode == 'terminate':
        raise RuntimeError(f"path '{str(path_obj)}' already exists")
    if mode == 'overwrite':
        return str(path_obj)
    while True:
        candidate = miscutils.make_unique_path(path_obj)
        try:
            candidate.mkdir(parents=True)
            return str(candidate)
        except FileExistsError:
            # Another run created it first: try the next one
            continue


def dump_dict(info: Any, filename: str) -> None:
    """Write a dict (of JSON values) to <filename>.json and .yaml."""
    path = Path(filename)
    with path.with_suffix('.json').open('w') as f:
        json.dump(info, f, indent=2)
    with path.with_suffix('.yaml').open('w') as f:
        _yaml.dump(info, f)


def write_yaml(info: Any, filename: str) -> None:
    """Write a dict (of JSON values) to a YAML file."""
    with open(filename, 'w') as f:
        _yaml.dump(info, f)


def merge_pdescs(
        pdescs1: dict[str, ParamDesc],
        pdescs2: dict[str, ParamDesc]
) -> dict[str, ParamDesc]:
    """
    Return the descriptions of the parameters of the galaxies (pdescs1) and
    those of a configuration (pdescs2), which must have other names.

    Raises
    ------
    ConfigError
        If names are in both.
    """
    if conflicting := sorted(set(pdescs1) & set(pdescs2)):
        raise ConfigError(
            f"these parameters of the configuration are parameters of the "
            f"galaxies: {conflicting}")
    return pdescs1 | pdescs2


def load_drivers(cfg: dict[str, Any]) -> dict[str | None, Driver]:
    """
    Return the drivers of a configuration, by name: the one of its
    section 'driver' (named None), those of its section 'drivers' (a dict
    of drivers by name), or none.

    Raises
    ------
    ConfigError
        If it has both sections, a name is invalid, or a driver has an
        error.
    """
    if 'driver' in cfg and 'drivers' in cfg:
        raise ConfigError(
            "the configuration must have either a section 'driver' (one "
            "driver) or a section 'drivers' (several, by name), not both")
    if 'driver' in cfg:
        with parseutils.config_path('driver'):
            return {None: driver_parser.load(cfg['driver'])}
    drivers = {}
    for name, info in cfg.get('drivers', {}).items():
        with parseutils.config_path('drivers', name):
            parseutils.check_name(name)
            drivers[name] = driver_parser.load(info)
    return drivers


def load_observation_group(cfg: dict[str, Any]) -> ObservationGroup:
    """
    Return the galaxies and observations (with their data, if any) of a
    configuration, as an ObservationGroup. The galaxies and observations
    name lines and bands of the catalogue of GBKFIT and of the sections
    'lines' and 'bands' of the configuration (see LineCatalogue), and the
    observations refer to the drivers of the configuration by name (see
    load_drivers).
    """
    drivers = load_drivers(cfg)
    catalogue = gbkfit.model.LineCatalogue(cfg.get('lines'), cfg.get('bands'))
    with parseutils.config_path('galaxies'):
        galaxies = gbkfit.model.galaxy_parser.load(
            cfg['galaxies'], catalogue=[catalogue] * len(cfg['galaxies']))
    count = len(cfg['observations'])
    with parseutils.config_path('observations'):
        observations = gbkfit.observation.observation_parser.load(
            cfg['observations'], drivers=[drivers] * count,
            catalogue=[catalogue] * count)
    return gbkfit.observation.ObservationGroup(galaxies, observations)


def load_params(
        cfg: dict[str, Any],
        pdescs: dict[str, ParamDesc],
        constants: dict[str, Any]
) -> ParamSpace:
    """
    Return the parameters of a configuration (its section 'params'), with
    the descriptions of the parameters of the galaxies and those of its
    optional section 'pdescs'.
    """
    if 'pdescs' in cfg:
        with parseutils.config_path('pdescs'):
            pdescs = merge_pdescs(pdescs, load_pdescs_dict(cfg['pdescs']))
    with parseutils.config_path('params'):
        return param_space_parser.load(
            cfg['params'], pdescs=pdescs, constants=constants)


def data_outputs(group: ObservationGroup) -> dict[str, Any]:
    """
    Return the data of the observations as outputs, by name, in the form
    their observables give them: their values (data_<i>_<item>_d, NaN
    where masked) and errors (data_<i>_<item>_e).
    """
    outputs = {}
    for i, observation in enumerate(group.observations()):
        if (dataset := observation.data()) is None:
            continue
        observable = observation.observable()
        for key in observable.keys():
            item = dataset[key]
            outputs[f'data_{i}_{key}_d'] = observable.output(item.data())
            if item.error() is not None:
                outputs[f'data_{i}_{key}_e'] = observable.output(
                    item.error())
    return outputs


def evaluation_outputs(
        group: ObservationGroup,
        objective: Objective | None,
        values: ParamValues,
        model: bool,
        residuals: bool
) -> dict[str, Any]:
    """
    Return the outputs of an evaluation, by name, in the form their
    observables give them: the models (model_<i>_<item>_<d|m|w>), the
    residuals ((model - data) / error) without and with the weights of
    the data items (residual_<item>_d and wresidual_<item>_d, with _<i>
    after the prefix if there are several observations; these need the
    objective), and the extra outputs
    (<prefix>_extra_<name>; those of the models once, under the first
    prefix: each residual evaluation evaluates the models again).
    """
    outputs = {}
    model_extra: dict[str, Any] = {}
    if model:
        for i, data_i in enumerate(group.model_h(values, model_extra)):
            observable = group.observations()[i].observable()
            for key, value in data_i.items():
                for kind in ('d', 'm', 'w'):
                    if value.get(kind) is not None:
                        outputs[f'model_{i}_{key}_{kind}'] = \
                            observable.output(value[kind])
    extras = [('model', model_extra)]
    if residuals:
        if objective is None:
            raise ValueError("the residuals need an objective")
        for prefix, weighted in (('residual', False), ('wresidual', True)):
            extra: dict[str, Any] = {}
            # (the objective reuses its residual arrays, so copy them)
            data = copy.deepcopy(objective.residual_nddata_h(
                values, weighted, None if model or weighted else extra))
            for i, data_i in enumerate(data):
                prefix_i = prefix + f'_{i}' * (objective.nobservations() > 1)
                observable = group.observations()[i].observable()
                for key, value in data_i.items():
                    outputs[f'{prefix_i}_{key}_d'] = observable.output(value)
            extras.append((prefix, extra))
    for prefix, extra in extras:
        for key, value in extra.items():
            outputs[f'{prefix}_extra_{key}'] = value
    return outputs


def write_outputs(
        output_dir: str,
        outputs: dict[str, Any],
        extra_filename: str
) -> None:
    """
    Write each output by its type: data on a grid (GridData) and spectra
    (SpectraData) to a FITS file with world coordinates, other arrays to
    a FITS file without, and the other values (numbers, strings, lists
    and dicts) together to <extra_filename>.json and .yaml.
    """
    values = {}
    for name, value in outputs.items():
        filename = os.path.join(output_dir, f'{name}.fits')
        if isinstance(value, gridutils.GridData):
            fitsutils.write_data(
                filename, value.data, value.coords, value.spectral_axis,
                overwrite=True)
        elif isinstance(value, gridutils.SpectraData):
            fitsutils.write_spectra(
                filename, value.data, value.coords, overwrite=True)
        elif isinstance(value, np.ndarray):
            fits.writeto(filename, value, overwrite=True)
        elif isinstance(value, (bool, int, float, str, list, tuple, dict,
                                np.generic)):
            values[name] = numutils.nativify(value)
        else:
            raise TypeError(
                f"output {name} has an unsupported type: "
                f"{type(value).__name__}")
    if values:
        dump_dict(values, os.path.join(output_dir, extra_filename))
