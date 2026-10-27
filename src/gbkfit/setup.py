"""
The front door of GBKFIT: a configuration set up as the objects it
describes (see load and Setup).

A configuration (a YAML or JSON file, or a dictionary) has the sections
'galaxies', 'observations' and 'params', and can have the sections
'driver' or 'drivers', 'lines', 'bands', 'pdescs', 'depth_order',
'fitter' and 'parallel', and those of the tasks ('mock' and 'bootstrap';
see gbkfit.tasks), which load leaves to them.
"""

import dataclasses
import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Self

import ruamel.yaml

import gbkfit.migration
import gbkfit.model
import gbkfit.observation
import gbkfit.outputs
import gbkfit.project
from gbkfit.driver import Driver, driver_by_name, driver_parser
from gbkfit.fitting import (
    EvaluationLog, EvaluationPool, FitControl, Fitter, FitResult,
    ProgressObserver, fitter_parser)
from gbkfit.objective import Evaluation, Objective
from gbkfit.observation import DepthGroup, GroupPlan, ObservationGroup
from gbkfit.params import (
    ParamDesc, ParamFunctions, ParamSpace, ParamValues,
    joint_prior_residuals, load_pdescs_dict, param_space_parser)
from gbkfit.utils import iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'ConfigFile',
    'Setup',
    'SetupEvaluation',
    'load',
    'read_config'
]


_log = logging.getLogger(__name__)

# Reads YAML (round trip: dicts keep their order)
_yaml = ruamel.yaml.YAML()

# The sections of configurations: the required ones, and the others (of
# load and of the tasks)
_REQUIRED = ('galaxies', 'observations', 'params')
_OPTIONAL = (
    'driver', 'drivers', 'lines', 'bands', 'pdescs', 'depth_order',
    'fitter', 'parallel', 'mock', 'bootstrap')


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


@dataclasses.dataclass(frozen=True)
class SetupEvaluation:
    """
    An evaluation of a setup at values of its free elements (see
    Setup.evaluate).

    Attributes
    ----------
    values : ParamValues
        The values of all parameters, with those of the solved elements
        that the objective solves there (without an objective, their
        values without data; see Solve.default).
    objective : Evaluation or None
        The evaluation of the objective there (its merits, the parts of
        its chi-square, and the solved values), or None without an
        objective.
    """
    values: ParamValues
    objective: Evaluation | None


@dataclasses.dataclass(frozen=True)
class Setup:
    """
    What a configuration sets up (see load): the plan of the observations
    of its galaxies on its drivers (see ObservationGroup.plan), its
    parameters, its objective if every observation has data, its fitter
    if it has one (its section 'fitter'), and its pool of evaluations if
    it has one (its section 'parallel'; see EvaluationPool).

    It holds the memory of its objective and the threads of its pool
    until it is closed (see close; e.g. at the end of a with block).
    """
    group_plan: GroupPlan
    params: ParamSpace
    objective: Objective | None
    fitter: Fitter | None
    pool: EvaluationPool | None

    def evaluate(
            self, free: Mapping[str, float] | Sequence[float] | None = None
    ) -> SetupEvaluation:
        """
        Evaluate the setup at values of its free elements: the values of
        all parameters, with those of the solved elements, and the
        evaluation of its objective there, if it has one.

        Parameters
        ----------
        free : Mapping of str to float or Sequence of float, optional
            The values of the free elements (see ParamSpace.evaluate); by
            default their option 'value'.

        Returns
        -------
        SetupEvaluation
            The evaluation.

        Raises
        ------
        ValueError
            If values of free elements are missing or unknown.
        ParamEvaluationError
            If the parameters cannot be evaluated there.
        """
        values = self.params.evaluate(free)
        if self.objective is None:
            return SetupEvaluation(values, None)
        evaluation = self.objective.evaluate(values)
        return SetupEvaluation(
            self.params.evaluate(free, evaluation.solved), evaluation)

    def outputs(
            self,
            evaluation: SetupEvaluation,
            data: bool = False,
            extra: bool = False
    ) -> dict[str, gbkfit.outputs.Output]:
        """
        Return the outputs of an evaluation, as the tasks write them, by
        their paths in a run (see gbkfit.outputs.evaluation_outputs).

        Parameters
        ----------
        evaluation : SetupEvaluation
            The evaluation (see evaluate).
        data : bool, optional
            Whether to add the data of the observations ('data', 'error'
            and 'data_image'; see gbkfit.outputs.data_outputs), which do
            not change with the values.
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
        outputs = gbkfit.outputs.evaluation_outputs(
            self.group_plan, self.objective, evaluation.values, True,
            residuals, extra)
        if data:
            outputs |= gbkfit.outputs.data_outputs(self.group_plan)
        return outputs

    def fit(
            self,
            observers: Sequence[ProgressObserver] = (),
            control: FitControl | None = None,
            log: EvaluationLog | None = None
    ) -> FitResult:
        """
        Fit the data of its objective with its fitter, on its pool if it
        has one (see Fitter.fit).

        Parameters
        ----------
        observers : Sequence of ProgressObserver, optional
            Who receives the progress of the fit as it runs (see
            FitProgress).
        control : FitControl, optional
            The control of the fit, to stop it.
        log : EvaluationLog, optional
            The log of the evaluations: a new one, or that of a fit to
            resume (with the same objective, parameters and fitter).

        Returns
        -------
        FitResult
            The result.

        Raises
        ------
        ConfigError
            If it has no fitter, or no objective (an observation has no
            data), or the fit cannot start (see Fitter.fit).
        """
        if self.fitter is None:
            raise ConfigError("a fit needs a fitter (section 'fitter')")
        if self.objective is None:
            raise ConfigError("a fit needs data for every observation")
        return self.fitter.fit(
            self.objective, self.params, observers, control, log, self.pool)

    def close(self) -> None:
        """
        Release the memory of its objective (allocated again by its next
        evaluation; see Objective.close), and stop the threads of its pool.
        """
        if self.pool is not None:
            self.pool.close()
        if self.objective is not None:
            self.objective.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_config(
        filename: str,
        required: tuple[str, ...] = _REQUIRED,
        optional: tuple[str, ...] = _OPTIONAL,
        texts: Mapping[str, str] | None = None
) -> ConfigFile:
    """
    Read a configuration file (YAML or JSON), with the pieces of its
    project that it names in place (see gbkfit.project.resolve), and check
    its sections.

    Parameters
    ----------
    filename : str
        The file.
    required, optional : tuple of str, optional
        The names of the required and the optional sections; by default
        those of every configuration, and all the others (see above).
    texts : Mapping of str to str, optional
        Texts to read in place of files, by their absolute paths: of the
        configuration, or of pieces of its project (e.g. edits not saved
        yet).

    Returns
    -------
    ConfigFile
        The configuration, as plain JSON values (see load). Its root is its
        optional section 'root' (itself relative to the file), or by
        default the folder of its project, or else the directory of the
        file.

    Raises
    ------
    ConfigError
        If the file cannot be read, its project is invalid, a required
        section is missing, a section is unknown (see
        parseutils.report_unknown) or of the wrong type, or an item has an
        old form (see gbkfit.migration).
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
    root = _root(cfg)
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
        _prepare_config(cfg, required, optional), directory, project, setup,
        sources)


def load(
        config: str | ConfigFile | Mapping[str, Any],
        drivers: Mapping[str | None, Driver] | None = None
) -> Setup:
    """
    Set up a configuration: the plan of the observations of its galaxies
    on its drivers, its parameters, its objective if every observation
    has data, its fitter if it has one, and its pool of evaluations if it
    has one (see Setup). A configuration with a fitter or a pool is of a
    fit: its observations need data.

    Parameters
    ----------
    config : str, ConfigFile or Mapping
        The configuration: its file (see read_config), the configuration
        read from its file, or a dictionary, whose relative paths are
        relative to its optional section 'root' (itself relative to the
        current directory), or else to the current directory.
    drivers : Mapping of str or None to Driver, optional
        The drivers to evaluate on, by name, in place of those that the
        configuration declares (its section 'driver', of the name None, or
        'drivers'; e.g. {None: DriverCuda()}), which its galaxies,
        observations and pool name.

    Returns
    -------
    Setup
        What it sets up.

    Raises
    ------
    ConfigError
        If the configuration is invalid, or it has a fitter or a pool and
        an observation has no data.

    Examples
    --------
    >>> import gbkfit
    >>> with gbkfit.load('config.yaml') as setup:
    ...     chi2 = setup.evaluate().objective.merits.chi2
    ...     result = setup.fit()
    """
    if isinstance(config, str):
        config = read_config(config)
    elif not isinstance(config, ConfigFile):
        config = _config_file(config)
    cfg = config.config
    with parseutils.paths_relative_to(config.root):
        _log.info("setting up galaxies and observations...")
        if drivers is None:
            drivers = _load_drivers(cfg)
        group = _load_observation_group(cfg)
        group_plan = _load_group_plan(cfg, group, drivers)
        _log.info("setting up params...")
        params = _load_params(
            cfg, group.pdescs(), group.constants(),
            group_plan.plan_functions())
        objective = None
        if 'fitter' in cfg or 'parallel' in cfg or all(
                obs.data() is not None for obs in group.observations()):
            _log.info("setting up objective...")
            objective = Objective(group_plan, params.solves())
        fitter = _load_fitter(cfg, params)
        pool = _load_pool(cfg, objective, drivers)
    return Setup(group_plan, params, objective, fitter, pool)


def _root(config: Any) -> str | None:
    """
    Remove the section 'root' of a configuration, and return it, if it has
    one.

    Raises
    ------
    ConfigError
        If it is not a directory.
    """
    root = config.pop('root', None) if isinstance(config, dict) else None
    if root is not None and not isinstance(root, str):
        raise ConfigError(
            f"section 'root' must be a directory; it is {root!r}")
    return root


def _config_file(config: Mapping[str, Any]) -> ConfigFile:
    """
    Return a configuration given as a dictionary as if read from its file
    (see read_config and load).
    """
    config = dict(config)
    root = os.path.abspath(_root(config) or os.curdir)
    return ConfigFile(_prepare_config(config), root, None, None, {})


def _prepare_config(
        config: dict[str, Any] | None,
        required: tuple[str, ...] = _REQUIRED,
        optional: tuple[str, ...] = _OPTIONAL
) -> dict[str, Any]:
    """
    Check the sections of a configuration, and return it as plain JSON
    values: unknown sections are reported (see parseutils.report_unknown)
    and left out, null sections are empty, and the sections 'galaxies' and
    'observations' are lists.

    Raises
    ------
    ConfigError
        If a required section is missing, a section is of the wrong type,
        or an item has an old form (see gbkfit.migration).
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
    _refuse_old_forms(config)
    return json.loads(json.dumps(config))


def _refuse_old_forms(config: dict[str, Any]) -> None:
    """
    Refuse the old forms of the items of the sections of a configuration
    (see gbkfit.migration.refuse_old_forms).
    """
    for section in ('galaxies', 'observations'):
        for i, item in enumerate(config.get(section, [])):
            with parseutils.config_path(section, i):
                gbkfit.migration.refuse_old_forms(section, item)
    for section in ('params', 'fitter'):
        if section in config:
            with parseutils.config_path(section):
                gbkfit.migration.refuse_old_forms(section, config[section])


def _merge_pdescs(
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


def _load_drivers(cfg: dict[str, Any]) -> dict[str | None, Driver]:
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


def _load_observation_group(cfg: dict[str, Any]) -> ObservationGroup:
    """
    Return the galaxies and observations (with their data, if any) of a
    configuration, as an ObservationGroup, with the order of the galaxies
    along the line of sight of its section 'depth_order'. The galaxies
    and observations name lines and bands of the catalogue of GBKFIT and
    of the sections 'lines' and 'bands' of the configuration (see
    LineCatalogue). Their option driver places them on a driver of the
    configuration (see _load_group_plan).
    """
    catalogue = gbkfit.model.LineCatalogue(cfg.get('lines'), cfg.get('bands'))
    ngalaxies = len(cfg['galaxies'])
    with parseutils.config_path('galaxies'):
        galaxies = gbkfit.model.galaxy_parser.load(
            _without_driver(cfg['galaxies']),
            catalogue=[catalogue] * ngalaxies)
    nobservations = len(cfg['observations'])
    with parseutils.config_path('observations'):
        observations = gbkfit.observation.observation_parser.load(
            _without_driver(cfg['observations']),
            catalogue=[catalogue] * nobservations)
    with parseutils.config_path('depth_order'):
        depth_order = _load_depth_order(cfg.get('depth_order'))
    return gbkfit.observation.ObservationGroup(
        galaxies, observations, depth_order)


def _without_driver(infos: Sequence[Any]) -> list[Any]:
    """
    Return the configurations of galaxies or observations without their
    option driver (see _load_group_plan).
    """
    return [
        {key: value for key, value in info.items() if key != 'driver'}
        if isinstance(info, dict) else info for info in infos]


def _load_group_plan(
        cfg: dict[str, Any],
        group: ObservationGroup,
        drivers: Mapping[str | None, Driver]
) -> GroupPlan:
    """
    Return the plan of the galaxies and observations of a configuration
    (see _load_observation_group and ObservationGroup.plan) on its
    drivers.

    The option driver of a galaxy or an observation is not a driver but
    the name of one of the drivers of the configuration, which declares
    them at its top level. An observation, and a galaxy with mass, can
    leave it out when there is one driver, and have the host when there
    are none. A galaxy without mass and without it has the driver of the
    observations that see it, if they have one, and else the host: its
    stage is for them alone.

    Raises
    ------
    ConfigError
        If an option driver is not the name of one of the drivers, or is
        left out while there are several.
    """
    # (imported here: the host driver needs its native module)
    from gbkfit.driver.drivers.host import DriverHost
    host = DriverHost()

    def driver_of(section: str, index: int, owner: str) -> Driver:
        info = cfg[section][index]
        with parseutils.config_path(section, index, 'driver'):
            driver = driver_by_name(info.get('driver'), drivers, owner)
        return host if driver is None else driver

    observation_drivers = [
        driver_of('observations', i, 'observation')
        for i in range(len(cfg['observations']))]
    galaxy_drivers = []
    for g, info in enumerate(cfg['galaxies']):
        if 'driver' in info or info.get('mass'):
            galaxy_drivers.append(driver_of('galaxies', g, 'galaxy'))
            continue
        seen_on = {
            id(driver): driver
            for i, driver in enumerate(observation_drivers)
            if g in group.seen_galaxies(i)}
        galaxy_drivers.append(
            next(iter(seen_on.values())) if len(seen_on) == 1 else host)
    return GroupPlan(group, galaxy_drivers, observation_drivers)


def _load_params(
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
            pdescs = _merge_pdescs(pdescs, load_pdescs_dict(cfg['pdescs']))
    with parseutils.config_path('params'):
        return param_space_parser.load(
            cfg['params'], pdescs=pdescs, constants=constants,
            functions=functions)


def _load_fitter(cfg: dict[str, Any], params: ParamSpace) -> Fitter | None:
    """
    Return the fitter of a configuration (its section 'fitter'), if it
    has one, with the free properties and the priors on several
    parameters checked for it (before a fit starts).
    """
    if 'fitter' not in cfg:
        return None
    _log.info("setting up fitter...")
    with parseutils.config_path('fitter'):
        fitter = fitter_parser.load(cfg['fitter'])
    with parseutils.config_path('params', 'properties'):
        fitter.load_fit_params(params)
    with parseutils.config_path('params'):
        joint_prior_residuals(params)
    return fitter


def _load_pool(
        cfg: dict[str, Any],
        objective: Objective | None,
        drivers: Mapping[str | None, Driver]
) -> EvaluationPool | None:
    """
    Return the pool of the section 'parallel' of a configuration, if it
    has one: {drivers: [names]}, copies of its objective (which it has,
    see load) on those of the drivers of the configuration (see
    EvaluationPool).

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
    return EvaluationPool(objective, pool_drivers)


def _load_depth_order(
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
