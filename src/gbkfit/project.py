"""
Projects: the folder of a study, whose files hold the pieces of its
setups, one piece per file:

    gbkfit-project.yaml       the marker of the project (its format)
    galaxies/<name>.yaml      a galaxy (an item of the section 'galaxies')
    observations/<name>.yaml  an observation (of 'observations')
    params/<name>.yaml        parameters (a section 'params')
    fitters/<name>.yaml       a fitter (a section 'fitter')
    setups/<name>.yaml        the configurations of the tasks (eval, fit)
    data/                     the data, in any files and folders
    runs/                     the runs of the setups (see gbkfit.outputs)

The name of a piece is the path of its file in the folder of its kind,
without .yaml, and can have folders (e.g. 'ngc1313/hi'). In the sections
galaxies, observations, params and fitter of a configuration in a
project, a string names a piece, and a dictionary is one in place; the
pieces never name others. The section 'params' can also be a list of
them, merged in order (see merge_params). The relative paths of the
configurations in a project, and of its pieces, are relative to its
folder.
"""

import copy
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import ruamel.yaml

from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'FORMAT',
    'FORMAT_VERSION',
    'MARKER',
    'Project',
    'Resolved',
    'create',
    'find',
    'merge_params',
    'open',
    'resolve'
]


FORMAT = 'gbkfit-project'
FORMAT_VERSION = 1
MARKER = 'gbkfit-project.yaml'

# The folders of the pieces, by the sections of the configurations that
# name them
_FOLDERS = dict(
    galaxies='galaxies', observations='observations', params='params',
    fitter='fitters')

_yaml = ruamel.yaml.YAML(typ='safe')


class Project:
    """
    A project (see the module): its folder, its pieces, its setups and
    their runs.

    Parameters
    ----------
    path : str
        The folder of the project.
    texts : Mapping of str to str, optional
        Texts to read in place of files of the project, by their absolute
        paths (e.g. edits not saved yet).
    """

    def __init__(self, path: str, texts: Mapping[str, str] | None = None):
        self._path = os.path.abspath(path)
        self._texts = dict(texts or {})

    @property
    def path(self) -> str:
        """The folder of the project (absolute)."""
        return self._path

    def piece(self, section: str, name: str) -> dict[str, Any]:
        """
        Return a piece of a section of the configurations by name.

        Parameters
        ----------
        section : {'galaxies', 'observations', 'params', 'fitter'}
            The section.
        name : str
            The name of the piece.

        Returns
        -------
        dict
            The piece.

        Raises
        ------
        ConfigError
            If the name is not a path in the folder of the pieces, or the
            piece does not exist, cannot be read, or is not a dictionary.
        """
        filename = self.piece_file(section, name)
        text = self._texts.get(filename)
        try:
            piece = _yaml.load(Path(filename) if text is None else text)
        except FileNotFoundError:
            raise ConfigError(
                f"the project has no piece '{name}' of {section} "
                f"({os.path.relpath(filename, self._path)})") from None
        except (OSError, ruamel.yaml.YAMLError) as e:
            raise ConfigError(f"cannot read the piece {filename}: {e}") \
                from e
        if not isinstance(piece, dict):
            raise ConfigError(
                f"the piece {os.path.relpath(filename, self._path)} must be "
                f"a dictionary")
        return piece

    def piece_file(self, section: str, name: str) -> str:
        """
        Return the file of a piece of a section by name (absolute).

        Raises
        ------
        ConfigError
            If the name is not a path in the folder of the pieces.
        """
        folder = os.path.join(self._path, _FOLDERS[section])
        filename = os.path.normpath(os.path.join(folder, f'{name}.yaml'))
        if os.path.isabs(name) or not filename.startswith(folder + os.sep):
            raise ConfigError(
                f"the name of a piece must be a path in {_FOLDERS[section]}/; "
                f"it is '{name}'")
        return filename

    def setup_name(self, filename: str) -> str | None:
        """
        Return the name of a setup by its file: its path in the folder
        setups, without .yaml; None for a file elsewhere.
        """
        path = os.path.relpath(
            os.path.abspath(filename), os.path.join(self._path, 'setups'))
        if path.startswith(os.pardir + os.sep):
            return None
        return os.path.splitext(path)[0]

    def run_path(self, name: str) -> str:
        """Return the folder of the runs of a setup by name (runs/<name>)."""
        return os.path.join(self._path, 'runs', name)


def create(path: str) -> Project:
    """
    Create a project: its folder, if needed, and its marker.

    Parameters
    ----------
    path : str
        The folder of the project.

    Returns
    -------
    Project
        The project.

    Raises
    ------
    FileExistsError
        If the folder is a project already.
    """
    os.makedirs(path, exist_ok=True)
    with Path(path, MARKER).open('x') as f:
        _yaml.dump(dict(format=FORMAT, version=FORMAT_VERSION), f)
    return Project(path)


def open(path: str) -> Project:
    """
    Open the project of a folder.

    Parameters
    ----------
    path : str
        The folder of the project.

    Returns
    -------
    Project
        The project.

    Raises
    ------
    ValueError
        If the folder has no marker of a project, or one of another format
        or of a newer version.
    """
    try:
        marker = _yaml.load(Path(path, MARKER))
    except FileNotFoundError:
        raise ValueError(f"{path} is not a project (no {MARKER})") from None
    if not isinstance(marker, dict) or marker.get('format') != FORMAT:
        raise ValueError(f"{os.path.join(path, MARKER)} is not of {FORMAT}")
    if marker.get('version', 0) > FORMAT_VERSION:
        raise ValueError(
            f"the project {path} is of version {marker['version']} of "
            f"{FORMAT}, newer than this version of gbkfit reads "
            f"({FORMAT_VERSION})")
    return Project(path)


def find(path: str) -> Project | None:
    """
    Return the project of a file or a folder: of the nearest folder with
    the marker of a project, from it up; None if there is none.

    Raises
    ------
    ValueError
        If the marker found is invalid (see open).
    """
    folder = os.path.abspath(path)
    if not os.path.isdir(folder):
        folder = os.path.dirname(folder)
    while True:
        if os.path.isfile(os.path.join(folder, MARKER)):
            return open(folder)
        parent = os.path.dirname(folder)
        if parent == folder:
            return None
        folder = parent


class Resolved(NamedTuple):
    """
    A configuration with its pieces in place (see resolve), and its
    sources, relative to its project: its file ('file'), the file of each
    item of its sections galaxies, observations and params (a list, of the
    layers), and of its fitter, and the file of the layer that gives the
    properties of each parameter ('parameters', by name).
    """
    config: dict[str, Any]
    sources: dict[str, Any]


def resolve(
        config: Mapping[str, Any],
        filename: str,
        project: Project | None
) -> Resolved:
    """
    Put the pieces that a configuration names in place, and merge the
    layers of its section 'params', if it is a list (see merge_params).

    Parameters
    ----------
    config : Mapping
        The configuration.
    filename : str
        Its file, the source of what it has in place.
    project : Project or None
        The project it is in, if any.

    Returns
    -------
    Resolved
        The configuration, and its sources (relative to the folder of the
        file outside a project).

    Raises
    ------
    ConfigError
        If it names pieces outside a project, or unknown pieces, or the
        layers of 'params' are invalid.
    """
    base = os.path.dirname(filename) if project is None else project.path
    config = dict(config)
    sources: dict[str, Any] = dict(file=os.path.relpath(filename, base))

    def source(section: str, item: Any) -> str:
        path = filename if not isinstance(item, str) \
            else project.piece_file(section, item)
        return os.path.relpath(path, base)

    for section in ('galaxies', 'observations', 'params'):
        items = config.get(section)
        if isinstance(items, str):
            items = [items]
        if not isinstance(items, list):
            continue
        loaded = [
            _piece(project, item, section, i) for i, item in enumerate(items)]
        sources[section] = [source(section, item) for item in items]
        if section != 'params':
            config[section] = loaded
            continue
        with parseutils.config_path('params'):
            config['params'], owners = merge_params(loaded)
        sources['parameters'] = {
            name: sources['params'][i] for name, i in owners.items()}
    if 'fitter' in config:
        item = config['fitter']
        config['fitter'] = _piece(project, item, 'fitter')
        sources['fitter'] = source('fitter', item)
    return Resolved(config, sources)


def _piece(
        project: Project | None,
        item: Any,
        section: str,
        *index: int
) -> Any:
    """
    Return an item of a section of a configuration: the piece it names,
    or itself.
    """
    if not isinstance(item, str):
        return item
    with parseutils.config_path(section, *index):
        if project is None:
            raise ConfigError(
                f"'{item}' names a piece of a project, but the "
                f"configuration is not in one (no {MARKER} above it)")
        return project.piece(section, item)


def merge_params(
        layers: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, int]]:
    """
    Merge layers of parameters (sections 'params'), in order.

    The properties of a parameter in a layer (by its name, with or
    without subscripts) replace all those of the parameter in the layers
    before it, as do its mode, and a derived value of a name. The priors
    of the layers add up. Any other option is that of the last layer
    that has it (e.g. transforms).

    Parameters
    ----------
    layers : Sequence of Mapping
        The layers.

    Returns
    -------
    dict
        The parameters.
    dict
        The layer of the properties of each parameter (its index), by
        name.

    Raises
    ------
    ConfigError
        If a layer is not a dictionary, or its options 'properties',
        'modes' or 'derived' are not dictionaries, or 'priors' is not a
        list.
    """
    merged: dict[str, Any] = {}
    owners: dict[str, int] = {}
    for i, layer in enumerate(layers):
        with parseutils.config_path(i):
            if not isinstance(layer, Mapping):
                raise ConfigError("a layer of parameters must be a dictionary")
            for key, value in layer.items():
                with parseutils.config_path(key):
                    merged[key] = _merge_option(key, merged.get(key), value)
            names = {_parameter(key) for key in layer.get('properties', {})}
            owners |= dict.fromkeys(names, i)
    return copy.deepcopy(merged), owners


def _merge_option(key: str, merged: Any, value: Any) -> Any:
    """Return an option of the merged layers with that of the next one."""
    if key == 'priors':
        if not isinstance(value, list):
            raise ConfigError("must be a list")
        return (merged or []) + value
    if key not in ('properties', 'modes', 'derived'):
        return value
    if not isinstance(value, Mapping):
        raise ConfigError("must be a dictionary")
    # (the properties of a parameter go by its name, without subscripts)
    name_of = _parameter if key == 'properties' else str
    replaced = {name_of(name) for name in value}
    kept = {
        name: v for name, v in (merged or {}).items()
        if name_of(name) not in replaced}
    return kept | dict(value)


def _parameter(key: str) -> str:
    """Return the name of the parameter of a key of the properties."""
    return key.split('[', 1)[0].strip()
