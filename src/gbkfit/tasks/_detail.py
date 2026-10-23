"""
Helpers shared by the tasks: reading configurations, the output
directory, loading the parts of a configuration, and writing outputs.
"""

import contextlib
import copy
import json
import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

from astropy.table import Table
import numpy as np
import ruamel.yaml

import gbkfit.fitting
import gbkfit.model
import gbkfit.observation
import gbkfit.outputs
import gbkfit.project
from gbkfit.driver import Driver, driver_by_name, driver_parser
from gbkfit.model import Extra, Extras
from gbkfit.objective import Objective
from gbkfit.observation import DepthGroup, Observation, ObservationGroup
from gbkfit.params import (
    ParamDesc, ParamFunctions, ParamSpace, ParamValues, load_pdescs_dict,
    param_space_parser)
from gbkfit.utils import (
    gridutils, iterutils, miscutils, parseutils, unitutils)
from gbkfit.utils.parseutils import ConfigError


_log = logging.getLogger(__name__)

# Reads YAML (round trip: dicts keep their order)
_yaml = ruamel.yaml.YAML()

# The log of a run, and the form of its messages
_LOG = 'logs/gbkfit.log'
_LOG_FORMAT = (
    '{asctime}:{processName}({process}):{threadName}({thread}):'
    '{levelname}:{name}:{funcName}:{lineno}:{message}')


class ConfigFile(NamedTuple):
    """
    A configuration read from its file (see read_config): the
    configuration, the directory its relative paths are relative to (its
    root), and the project it is in, if any (see gbkfit.project), with the
    name of its setup (none out of the folder setups) and its sources (see
    gbkfit.project.Resolved).
    """
    config: dict[str, Any]
    root: str
    project: gbkfit.project.Project | None
    setup: str | None
    sources: dict[str, Any]


def read_config(
        filename: str,
        required: tuple[str, ...],
        optional: tuple[str, ...] = (),
        texts: Mapping[str, str] | None = None
) -> ConfigFile:
    """
    Read a configuration file (YAML or JSON), with the pieces of its
    project that it names in place (see gbkfit.project.resolve), and check
    its sections (see prepare_config).

    Parameters
    ----------
    filename : str
        The file.
    required, optional : tuple of str
        The names of the required and the optional sections.
    texts : Mapping of str to str, optional
        Texts to read in place of files, by their absolute paths: of the
        configuration, or of pieces of its project (e.g. edits not saved
        yet).

    Returns
    -------
    ConfigFile
        The configuration. Its root is its optional section 'root' (itself
        relative to the file), or by default the folder of its project, or
        else the directory of the file.

    Raises
    ------
    ConfigError
        If the file cannot be read, its project is invalid, or its pieces
        or its sections are invalid.
    """
    filename = os.path.abspath(filename)
    texts = texts or {}
    try:
        if filename in texts:
            cfg = _yaml.load(texts[filename])
        else:
            with open(filename) as f:
                cfg = _yaml.load(f)
    except (OSError, ruamel.yaml.YAMLError) as e:
        raise ConfigError(
            f"cannot read the configuration file {filename}: {e}") from e
    root = cfg.pop('root', None) if isinstance(cfg, dict) else None
    if root is not None and not isinstance(root, str):
        raise ConfigError(
            f"section 'root' must be a directory; it is {root!r}")
    try:
        project = gbkfit.project.find(filename)
    except ValueError as e:
        raise ConfigError(str(e)) from e
    if project is not None and texts:
        project = gbkfit.project.Project(project.path, texts)
    directory = os.path.dirname(filename) if project is None \
        else project.path
    if root is not None:
        directory = os.path.normpath(
            os.path.join(os.path.dirname(filename), root))
    setup, sources = None, {}
    if project is not None:
        setup = project.setup_name(filename)
        _log.info(f"in the project {project.path}")
    if isinstance(cfg, dict):
        cfg, sources = gbkfit.project.resolve(cfg, filename, project)
    return ConfigFile(
        prepare_config(cfg, required, optional), directory, project, setup,
        sources)


def run_dir(output_dir: str | None, config: ConfigFile) -> str:
    """
    Return the directory of the run of a configuration: the one given, or
    by default the folder of the runs of its setup in its project (see
    gbkfit.project.Project.run_path), or else 'output'.
    """
    if output_dir is not None:
        return output_dir
    if config.setup is not None:
        return config.project.run_path(config.setup)
    return 'output'


def write_setup(run: gbkfit.outputs.RunWriter, config: ConfigFile) -> None:
    """
    Write the sources of a configuration in a project as a product of a
    run (setup.json, kind 'setup'; see gbkfit.project.Resolved), with the
    folder of the project relative to the run ('project').
    """
    if config.project is None:
        return
    run.write(
        'setup.json', 'setup',
        dict(project=os.path.relpath(config.project.path, run.path))
        | config.sources)


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
              'fitter', 'parallel'):
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


def load_observation_group(
        cfg: dict[str, Any],
        drivers: Mapping[str | None, Driver] | None = None
) -> ObservationGroup:
    """
    Return the galaxies and observations (with their data, if any) of a
    configuration, as an ObservationGroup, with the order of the galaxies
    along the line of sight of its section 'depth_order'. The galaxies
    and observations name lines and bands of the catalogue of GBKFIT and
    of the sections 'lines' and 'bands' of the configuration (see
    LineCatalogue), and the observations and the galaxies refer to the
    drivers of the configuration by name: the given ones, or by default
    those it declares (see load_drivers).
    """
    if drivers is None:
        drivers = load_drivers(cfg)
    catalogue = gbkfit.model.LineCatalogue(cfg.get('lines'), cfg.get('bands'))
    ngalaxies = len(cfg['galaxies'])
    with parseutils.config_path('galaxies'):
        galaxies = gbkfit.model.galaxy_parser.load(
            cfg['galaxies'], catalogue=[catalogue] * ngalaxies,
            drivers=[drivers] * ngalaxies)
    nobservations = len(cfg['observations'])
    with parseutils.config_path('observations'):
        observations = gbkfit.observation.observation_parser.load(
            cfg['observations'], drivers=[drivers] * nobservations,
            catalogue=[catalogue] * nobservations)
    with parseutils.config_path('depth_order'):
        depth_order = load_depth_order(cfg.get('depth_order'))
    return gbkfit.observation.ObservationGroup(
        galaxies, observations, depth_order)


def load_pool(
        cfg: dict[str, Any],
        objective: Objective,
        drivers: Mapping[str | None, Driver]
) -> gbkfit.fitting.EvaluationPool | None:
    """
    Return the pool of the section 'parallel' of a configuration, if it
    has one: {drivers: [names]}, copies of the objective on those of the
    drivers of the configuration (see load_drivers and EvaluationPool).

    Raises
    ------
    ConfigError
        If the section has an error, or a name is not one of the drivers.
    """
    if 'parallel' not in cfg:
        return None
    with parseutils.config_path('parallel'):
        info = cfg['parallel']
        if unknown := [key for key in info if key != 'drivers']:
            parseutils.report_unknown("unknown options", unknown, ['drivers'])
        names = info.get('drivers')
        if not isinstance(names, list) or not names:
            raise ConfigError(
                "option 'drivers' must be a list of the names of drivers of "
                "the configuration (section 'drivers')")
        with parseutils.config_path('drivers'):
            pool_drivers = [
                driver_by_name(name, drivers, 'parallel evaluations')
                for name in names]
    return gbkfit.fitting.EvaluationPool(objective, pool_drivers)


def load_depth_order(
        info: Sequence[str | dict[str, Sequence[str]]] | None
) -> list[str | DepthGroup] | None:
    """
    Return the depth order of a configuration (see ObservationGroup):
    names of galaxies, and groups of them as {name: [galaxies]}.

    Raises
    ------
    ConfigError
        If an item is neither a name nor a group of one name.
    """
    if info is None:
        return None
    if not iterutils.is_sequence(info):
        raise ConfigError("depth_order must be a list")
    order = []
    for item in info:
        if iterutils.is_mapping(item):
            if len(item) != 1:
                raise ConfigError(
                    f"a group of depth_order is {{name: [galaxies]}}; "
                    f"{item} is not")
            (name, galaxies), = item.items()
            if not iterutils.is_sequence(galaxies):
                raise ConfigError(
                    f"the galaxies of the group {name!r} of depth_order "
                    f"must be a list")
            order.append(DepthGroup(name, tuple(galaxies)))
        else:
            order.append(item)
    return order


def load_params(
        cfg: dict[str, Any],
        pdescs: dict[str, ParamDesc],
        constants: dict[str, Any],
        functions: ParamFunctions | None = None
) -> ParamSpace:
    """
    Return the parameters of a configuration (its section 'params'), with
    the descriptions of the parameters of the galaxies and those of its
    optional section 'pdescs', and the functions of the galaxies that its
    derived values can use.
    """
    if 'pdescs' in cfg:
        with parseutils.config_path('pdescs'):
            pdescs = merge_pdescs(pdescs, load_pdescs_dict(cfg['pdescs']))
    with parseutils.config_path('params'):
        return param_space_parser.load(
            cfg['params'], pdescs=pdescs, constants=constants,
            functions=functions)


class Output(NamedTuple):
    """
    An output of a task, to write as a product of its run (see
    gbkfit.outputs): its path in the run, kind, value and ids, and its
    title and description, if any.
    """
    path: str
    kind: str
    value: Any
    ids: dict[str, str]
    title: str | None = None
    description: str | None = None


# The folders of the extra outputs by their ids, in this order (see
# _extra_output)
_EXTRA_FOLDERS = dict(
    observation='observations', galaxy='galaxies', group='groups',
    component='components')


def data_outputs(group: ObservationGroup) -> dict[str, Output]:
    """
    Return the data of the observations as outputs, by path, in the form
    the observations give them (see Observation.output): their values
    (kind 'data', NaN where masked) and errors ('error'), in their units,
    of each data item, and the images of their values and of their PSF
    that the observations make ('data_image'; e.g. the dirty image and
    beam of visibilities, see Observation.images and psf_images), in
    observations/<observation>/<kind>/<item or image>.
    """
    outputs = []
    for i, observation in enumerate(group.observations()):
        if (dataset := observation.data()) is None:
            continue
        name = group.observation_label(i)
        for key in observation.keys():
            item = dataset[key]
            outputs.append(_observation_output(
                name, 'data', key, _with_unit(
                    observation.output(item.data()), observation, key)))
            if item.error() is not None:
                outputs.append(_observation_output(
                    name, 'error', key, _with_unit(
                        observation.output(item.error()), observation, key)))
        weights = _weights(dataset)
        images = observation.images(
            {key: dataset[key].data() for key in observation.keys()},
            weights, _units(dataset)) | observation.psf_images(weights)
        outputs += [
            _observation_output(name, 'data_image', image, x, 'image')
            for image, x in images.items()]
    return {output.path: output for output in outputs}


def _units(dataset: Any) -> dict[str, Any] | None:
    """Return the units of the data items of a dataset; None without data."""
    return None if dataset is None \
        else {key: x.unit() for key, x in dataset.items()}


def _weights(dataset: Any) -> dict[str, np.ndarray] | None:
    """
    Return the weights of the values of the data items of a dataset, 1 /
    error^2 (0 where they have none); None without data or errors.
    """
    if dataset is None or any(x.error() is None for x in dataset.values()):
        return None
    with np.errstate(divide='ignore'):
        return {
            key: np.where(np.isfinite(x.error()), 1 / x.error() ** 2, 0)
            for key, x in dataset.items()}


def _with_unit(output: Any, observation: Observation, key: str) -> Any:
    """
    Return an output of a data item of an observation (or of its model,
    which is in the unit of its data) with the unit of the item, if it
    has one and the output takes it (see GridData, SpectraData).
    """
    data = observation.data()
    if data is None or data[key].unit() is None or not isinstance(
            output, (gridutils.GridData, gridutils.SpectraData)):
        return output
    return output._replace(unit=unitutils.format_unit(data[key].unit()))


def evaluation_outputs(
        group: ObservationGroup,
        objective: Objective | None,
        values: ParamValues,
        model: bool,
        residuals: bool,
        extra: bool = True
) -> dict[str, Output]:
    """
    Return the outputs of an evaluation, by path, in the form the
    observations give them, in observations/<observation>/<kind>/<item>
    of each data item: the models (kind 'model'; 'model_mask' and
    'model_weight'), and the residuals ((model - data) / error) without
    and with the weights of the data items ('residual' and 'wresidual';
    these need the objective). With the residuals, the free amplitudes of
    spectra (see LikelihoodGaussian) are 'amplitudes'
    (observations/<observation>/amplitudes), and with the models too, the
    model of the spectra is that which the residuals compare, times the
    amplitudes ('model'), and the model without them is
    'model_unscaled'. With the models, the images that the observations
    make of them ('model_image', in place of the item: the name of the
    image; see Observation.images), and of the data minus them
    ('residual_image'), with the weights of the data. Unless left out
    (extra), the extra outputs of the models are 'extra', with their
    titles and descriptions (see _extra_output).
    """
    outputs = []
    extras = Extras() if extra else None
    if model:
        for i, data_i in enumerate(group.model_h(values, extras)):
            observation = group.observations()[i]
            name = group.observation_label(i)
            for key, value in data_i.items():
                for kind, part in (
                        ('model', 'd'), ('model_mask', 'm'),
                        ('model_weight', 'w')):
                    if value.get(part) is None:
                        continue
                    output = observation.output(value[part])
                    if part == 'd':
                        output = _with_unit(output, observation, key)
                    outputs.append(
                        _observation_output(name, kind, key, output))
            outputs += _images(
                observation, name, {k: v['d'] for k, v in data_i.items()})
    if residuals:
        if objective is None:
            raise ValueError("the residuals need an objective")
        for kind, weighted in (('residual', False), ('wresidual', True)):
            # (the objective reuses its residual arrays, so copy them;
            # without the models, the extra outputs are those of the
            # first residual evaluation, which evaluates the models)
            data = copy.deepcopy(objective.residual_nddata_h(
                values, weighted, None if model or weighted else extras))
            for i, data_i in enumerate(data):
                name = group.observation_label(i)
                observation = group.observations()[i]
                outputs += [
                    _observation_output(
                        name, kind, key, observation.output(value))
                    for key, value in data_i.items()]
    outputs += [_extra_output(x) for x in extras or ()]
    outputs = {output.path: output for output in outputs}
    if residuals:
        for i, amplitudes in enumerate(objective.amplitudes()):
            if amplitudes is None:
                continue
            name = group.observation_label(i)
            output = Output(
                f'observations/{name}/amplitudes.fits', 'amplitudes',
                amplitudes, dict(observation=name))
            outputs[output.path] = output
            if model:
                key = group.observations()[i].spectra_item()
                path = f'observations/{name}/model/{key}.fits'
                unscaled = outputs[path]
                outputs[path] = unscaled._replace(
                    value=_scaled(unscaled.value, amplitudes))
                output = _observation_output(
                    name, 'model_unscaled', key, unscaled.value)
                outputs[output.path] = output
    return outputs


def _images(
        observation: Observation,
        name: str,
        model: Mapping[str, np.ndarray]
) -> list[Output]:
    """
    Return the images that an observation (of the given name in the
    outputs) makes of its model ('model_image') and of its data minus the
    model ('residual_image'; see Observation.images).
    """
    dataset = observation.data()
    weights = _weights(dataset)
    units = _units(dataset)
    images = [
        _observation_output(name, 'model_image', image, x, 'image')
        for image, x in observation.images(model, weights, units).items()]
    # (an observation that makes no images of its model makes none of the
    # residual either)
    if images and dataset is not None:
        residual = {key: dataset[key].data() - model[key] for key in model}
        images += [
            _observation_output(name, 'residual_image', image, x, 'image')
            for image, x in observation.images(
                residual, weights, units).items()]
    return images


def _extra_output(extra: Extra) -> Output:
    """
    Return an extra output of a model (kind 'extra', with its ids and its
    name as the id 'name'), in the folders of its ids (see
    _EXTRA_FOLDERS; e.g. observations/<observation>/galaxies/<galaxy>/
    components/<component>/<name>), or in observations/<observation>/
    extra/<name> if it is of the observation alone (e.g. the model cube;
    its folder holds the other outputs of the observation by kind).
    """
    folders = []
    for key, folder in _EXTRA_FOLDERS.items():
        if key in extra.ids:
            folders += [folder, extra.ids[key]]
    if set(extra.ids) == {'observation'}:
        folders.append('extra')
    return Output(
        '/'.join(folders + [f'{extra.name}{_extension(extra.value)}']),
        'extra', extra.value, extra.ids | dict(name=extra.name),
        extra.title, extra.description)


def _observation_output(
        observation: str,
        kind: str,
        name: str,
        value: Any,
        id_name: str = 'item'
) -> Output:
    """
    Return an output of a kind of an observation, of a data item (or of
    another id: e.g. 'image'), in observations/<observation>/<kind>/
    <name>.
    """
    return Output(
        f'observations/{observation}/{kind}/{name}{_extension(value)}', kind,
        value, {'observation': observation, id_name: name})


def _extension(value: Any) -> str:
    """
    Return the extension of the file of an output by its type: FITS for
    arrays (data on a grid and spectra with world coordinates), ECSV for
    tables, and JSON for other values (numbers, strings, lists and dicts).

    Raises
    ------
    TypeError
        If the type is not of these.
    """
    if isinstance(
            value, (gridutils.GridData, gridutils.SpectraData, np.ndarray)):
        return '.fits'
    if isinstance(value, Table):
        return '.ecsv'
    if isinstance(value, (bool, int, float, str, list, tuple, dict,
                          np.generic)):
        return '.json'
    raise TypeError(f"outputs of type {type(value).__name__} are unsupported")


def _scaled(output: Any, amplitudes: np.ndarray) -> Any:
    """
    Return an output of spectra (along the first axis of its data) with
    each spectrum times its amplitude.
    """
    if isinstance(output, np.ndarray):
        return output * amplitudes
    return output._replace(data=output.data * amplitudes)


@contextlib.contextmanager
def run_log(run: gbkfit.outputs.RunWriter) -> Iterator[None]:
    """
    While the block runs, copy the log of GBKFIT (the logger 'gbkfit', at
    its level) to the log of a run (logs/gbkfit.log, kind 'log'; appended
    to, e.g. when a fit resumes), with the time, the process, the thread
    and the place of each message.
    """
    handler = logging.FileHandler(run.filename(_LOG))
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, style='{'))
    run.record(_LOG, 'log', 'log')
    logger = logging.getLogger('gbkfit')
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()


def write_outputs(
        run: gbkfit.outputs.RunWriter,
        outputs: Mapping[str, Output]
) -> None:
    """Write outputs as products of a run (see RunWriter.write)."""
    for output in outputs.values():
        run.write(
            output.path, output.kind, output.value, title=output.title,
            description=output.description, **output.ids)


def write_config(
        run: gbkfit.outputs.RunWriter,
        path: str,
        kind: str,
        cfg: Mapping[str, Any],
        root: str,
        **ids: str
) -> None:
    """
    Write a configuration as a product of a run (YAML), with its root (the
    directory its relative paths are relative to) relative to the
    directory of its file.
    """
    directory = os.path.dirname(os.path.join(run.path, path))
    run.write(
        path, kind,
        dict(root=os.path.relpath(root, directory)) | copy.deepcopy(cfg),
        **ids)
