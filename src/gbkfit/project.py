"""
Projects: the folder of a study, whose files hold the pieces of its
setups, one piece per file:

    gbkfit-project.yaml       the marker of the project (its format)
    galaxies/<name>.yaml      a galaxy (an item of the section 'galaxies')
    observations/<name>.yaml  an observation (of 'observations')
    datasets/<name>.yaml      a dataset (the option 'data' of an observation)
    psfs/<name>.yaml          a PSF (the option 'psf' of an instrument)
    lsfs/<name>.yaml          an LSF (the option 'lsf' of an instrument)
    noise/<name>.yaml         a noise (the option 'noise' of the mock data)
    params/<name>.yaml        parameters (a section 'params')
    fitters/<name>.yaml       a fitter (a section 'fitter')
    setups/<name>.yaml        the configurations of the tasks (eval, fit)
    data/                     the data, in any files and folders
    runs/                     the runs of the setups (see gbkfit.outputs)

The name of a piece is the path of its file in the folder of its kind,
without .yaml, and can have folders (e.g. 'ngc1313/hi'). In the sections
galaxies, observations, params and fitter of a configuration in a
project, in the options data, instrument.psf and instrument.lsf of its
observations, and in the option noise of its section mock, a string
names a piece, and a dictionary is one in place. Of the pieces, only
observations name others (their dataset, PSF and LSF). The section
'params' can also be a list of pieces, merged in order (see
merge_params), and the option noise one for each observation. The
relative paths of the configurations in a project, and of its pieces,
are relative to its folder.
"""

import copy
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import ruamel.yaml

from gbkfit.config import FORMAT_VERSION, ConfigError, config_path


__all__ = [
    'FORMAT',
    'MARKER',
    'TEMPLATES',
    'Project',
    'Resolved',
    'create',
    'find',
    'merge_params',
    'open',
    'piece_folder',
    'resolve'
]


FORMAT = 'gbkfit-project'
MARKER = 'gbkfit-project.yaml'

# The folders of the pieces, by the sections and options of the
# configurations that name them (see the module)
_FOLDERS = dict(
    galaxies='galaxies', observations='observations', data='datasets',
    psf='psfs', lsf='lsfs', noise='noise', params='params',
    fitter='fitters')


def piece_folder(section: str) -> str:
    """
    Return the folder of the pieces of a section or an option of the
    configurations that name them (e.g. 'lsfs', of 'lsf'; see the module).

    Raises
    ------
    KeyError
        If no pieces are of the section.
    """
    return _FOLDERS[section]


def _yaml() -> ruamel.yaml.YAML:
    """Return a new YAML that reads and writes plain values."""
    return ruamel.yaml.YAML(typ='safe')


def _yaml_out() -> ruamel.yaml.YAML:
    """
    Return a new YAML of the pieces written with their content: lists and
    dictionaries of plain values on one line (e.g. data: {file:
    data/cube.fits, hdu: SCI}).
    """
    yaml = ruamel.yaml.YAML(typ='safe')
    yaml.default_flow_style = None
    yaml.sort_base_mapping_type_on_output = False
    return yaml


def _yaml_rt() -> ruamel.yaml.YAML:
    """
    Return a new YAML of the pieces rewritten in place (e.g. as a piece
    they name is renamed): their comments and layout stay.
    """
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    return yaml


# The templates of new pieces and setups, by folder (see
# Project.create_piece): a disk seen in a cube; a setup names the first
# piece of each folder ({galaxies}, ...)
TEMPLATES = {
    'galaxies': """\
# A galaxy: here a thin disk with an exponential surface brightness, an
# arctan rotation curve and a uniform velocity dispersion, seen in an
# emission line. Its parameters are named after its component (disk_*).
components:
- type: thin_disk
  name: disk
  loose: false
  tilted: false
  rnodes: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
  density:
    ptraits: {type: exponential}
  kinematics:
    type: traits
    vptraits: {type: arctan, direction: tan}
    dptraits: {type: uniform}
  emission: {type: lines}
""",
    'observations': """\
# An observation: here a cube of 48 x 48 spaxels of 1 arcsec and 40
# channels of 10 km/s of radio velocities (as of HI; optical for lines of
# ionised gas), with Gaussian PSF and LSF. To fit data, name a dataset of
# the project instead of the grid (data: <name>); the PSF and the LSF can
# be named too (psf: <name>, lsf: <name>).
type: sky
observable:
  type: pixel_spectra
  size: [48, 48]
  step: [1, 1]
  spectral: {size: 40, step: 10, convention: radio}
line_of_sight: {type: flat}
instrument:
  psf: {type: gauss, sigma: 1.5}
  lsf: {type: gauss, sigma: 15 km/s}
""",
    'datasets': """\
# A dataset: here a cube and its errors (FITS files; paths relative to the
# project), of the type of the observable of the observations that name it
# (data: <name>).
type: pixel_spectra
data: data/cube.fits
error: data/error.fits
""",
    'psfs': """\
# A PSF: here a Gaussian of dispersion 1.5 arcsec. Observations name it in
# their instrument (psf: <name>).
type: gauss
sigma: 1.5
""",
    'lsfs': """\
# An LSF: here a Gaussian of dispersion 15 km/s (of the light; a width in
# Angstrom, kHz or pixels of the data, or its fwhm instead of sigma, as
# the instrument gives it). Observations name it in their instrument (lsf:
# <name>).
type: gauss
sigma: 15 km/s
""",
    'noise': """\
# A noise, of mock data: here Gaussian, of dispersion 0.01 at every value.
# The section mock of a setup names it (noise: <name>).
type: gauss
rms: 0.01
""",
    'params': """\
# Parameters: fixed values, and free ones (value, min, max). Setups list
# layers of them; a later layer replaces each parameter it gives.
properties:
  disk_xpos: 0
  disk_ypos: 0
  disk_vsys: 0
  disk_posa: {value: 30, min: 0, max: 180}
  disk_incl: {value: 60, min: 5, max: 85}
  disk_pt_a: 1
  disk_pt_s: 5
  disk_vpt_rt: 3
  disk_vpt_vt: {value: 180, min: 50, max: 300}
  disk_dpt_a: 20
""",
    'fitters': """\
# A fitter: here least squares (scipy), the quickest; samplers give
# posteriors (e.g. type: nautilus).
type: scipy.least_squares
""",
    'setups': """\
# A setup: the pieces of the project it is made of, by name (their paths
# in their folders, without .yaml), or written in place. Preview it, or
# evaluate or fit it (a fit needs data and a fitter).
driver: {{type: host}}
galaxies: [{galaxies}]
observations: [{observations}]
params: [{params}]
# fitter: {fitters}
""",
}


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

    def version(self) -> int:
        """
        Return the version of the format of its pieces and setups (see
        FORMAT_VERSION), from its marker.
        """
        return _marker(self._path)['version']

    def piece(self, section: str, name: str) -> dict[str, Any]:
        """
        Return a piece of a section (or an option) of the configurations
        by name.

        Parameters
        ----------
        section : str
            The section, or the option, that names it (see the module):
            'galaxies', 'observations', 'data', 'psf', 'lsf', 'noise',
            'params' or 'fitter'.
        name : str
            The name of the piece.

        Returns
        -------
        dict
            The piece.

        Raises
        ------
        ValueError
            If the name is not a path in the folder of the pieces.
        ConfigError
            If the piece does not exist (a configuration names it), is
            not YAML, or is not a dictionary.
        OSError
            If the piece cannot be read.
        """
        filename = self.piece_file(section, name)
        text = self._texts.get(filename)
        try:
            piece = _yaml().load(Path(filename) if text is None else text)
        except FileNotFoundError:
            raise ConfigError(
                f"the project has no piece '{name}' of {_FOLDERS[section]} "
                f"({os.path.relpath(filename, self._path)})") from None
        except ruamel.yaml.YAMLError as e:
            raise ConfigError(f"the piece {filename} is not YAML: {e}") \
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
        ValueError
            If the name is not a path in the folder of the pieces.
        """
        return self._file(_FOLDERS[section], name)

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

    def names(self, folder: str) -> list[str]:
        """Return the names of the pieces (or setups) of a folder."""
        return sorted(
            path.relative_to(Path(self._path, folder)).with_suffix('')
            .as_posix()
            for path in Path(self._path, folder).rglob('*.yaml'))

    def create_piece(
            self,
            folder: str,
            name: str,
            content: Mapping[str, Any] | None = None
    ) -> str:
        """
        Create a piece (or a setup) with its content, or from the template
        of its folder (see TEMPLATES), and return its file. A setup made
        from its template names the first piece of each folder, if there
        is one.

        Parameters
        ----------
        folder : str
            The folder: one of pieces (e.g. 'galaxies', 'psfs'), or
            'setups' (see TEMPLATES).
        name : str
            The name of the piece (e.g. 'ngc1313/hi').
        content : Mapping, optional
            The content of the piece (its options); by default, that of
            the template.

        Returns
        -------
        str
            The file of the piece (absolute).

        Raises
        ------
        ValueError
            If the folder is not one of pieces or setups, or the name is
            not a path in it.
        FileExistsError
            If the piece exists.
        """
        if folder not in TEMPLATES:
            raise ValueError(
                f"pieces are in the folders {list(TEMPLATES)}; not {folder!r}")
        filename = self._file(folder, name)
        text = TEMPLATES[folder]
        if folder == 'setups':
            text = text.format(**{
                f: next(iter(self.names(f)), '') for f in _FOLDERS.values()})
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with Path(filename).open('x') as f:
            if content is None:
                f.write(text)
            else:
                _yaml_out().dump(dict(content), f)
        return filename

    def update_piece(
            self, folder: str, name: str, content: Mapping[str, Any]
    ) -> str:
        """
        Write the content of a piece (or a setup) that exists, and return
        its file: its options that change are written anew, and the rest
        of its text (e.g. its comments, and the options that stay) as it
        is; but the comment of its template, which says what it is (e.g.
        'here a Gaussian of dispersion 1.5 arcsec'), goes as it changes.

        Raises
        ------
        ValueError
            If the folder is not one of pieces or setups, or the name is
            not a path in it.
        FileNotFoundError
            If the piece does not exist.
        """
        filename = self._piece_file(folder, name)
        try:
            text = Path(filename).read_text()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"the project has no piece '{name}' of {folder}") from None
        try:
            old = _yaml_rt().load(text)
        except ruamel.yaml.YAMLError:
            old = None
        # (the comment of its template, which says what it is, e.g. 'here
        # a Gaussian of dispersion 1.5 arcsec', goes as it changes)
        header = _header(TEMPLATES.get(folder, ''))
        if isinstance(old, Mapping) and header and text.startswith(header) \
                and _plain(old) != dict(content):
            old = _yaml_rt().load(text[len(header):])
        if not isinstance(old, Mapping):
            with Path(filename).open('w') as f:
                _yaml_out().dump(dict(content), f)
            return filename
        for key in [key for key in old if key not in content]:
            del old[key]
        for key, value in content.items():
            if key not in old or _plain(old[key]) != value:
                old[key] = value
        with Path(filename).open('w') as f:
            _yaml_rt().dump(old, f)
        return filename

    def references(self, folder: str, name: str) -> list[tuple[str, str]]:
        """
        Return the pieces and setups that name a piece (see the module):
        their folders and names, in order. A file that cannot be read
        names none.

        Parameters
        ----------
        folder : str
            The folder of the piece (e.g. 'datasets').
        name : str
            The name of the piece.
        """
        found = []
        for other in ('observations', 'setups'):
            for other_name in self.names(other):
                content = _load_quietly(self._file(other, other_name))
                if any(target == folder and parent[key] == name
                       for parent, key, target in _named(other, content)):
                    found.append((other, other_name))
        return found

    def rename_piece(self, folder: str, name: str, new: str) -> str:
        """
        Rename a piece (or a setup), and the pieces and setups that name
        it with it (their other text, e.g. comments, stays as it is); and
        return its new file. The folders its old name leaves empty go.

        Raises
        ------
        ValueError
            If the folder is not one of pieces or setups, or a name is
            not a path in it.
        FileNotFoundError
            If the piece does not exist.
        FileExistsError
            If a piece of the new name exists.
        """
        source = self._piece_file(folder, name)
        target = self._file(folder, new)
        if not os.path.isfile(source):
            raise FileNotFoundError(f"the project has no piece '{name}' "
                                    f"of {folder}")
        if os.path.exists(target):
            raise FileExistsError(f"the project has a piece '{new}' of "
                                  f"{folder}")
        referring = self.references(folder, name)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, target)
        self._prune(folder, source)
        for other, other_name in referring:
            filename = self._file(other, other_name)
            content = _yaml_rt().load(Path(filename))
            for parent, key, target_folder in _named(other, content):
                if target_folder == folder and parent[key] == name:
                    parent[key] = new
            with Path(filename).open('w') as f:
                _yaml_rt().dump(content, f)
        return target

    def delete_piece(self, folder: str, name: str) -> None:
        """
        Delete a piece (or a setup); the folders its name leaves empty go.
        The pieces and setups that name it still do (see references).

        Raises
        ------
        ValueError
            If the folder is not one of pieces or setups, or the name is
            not a path in it.
        FileNotFoundError
            If the piece does not exist.
        """
        filename = self._piece_file(folder, name)
        try:
            os.remove(filename)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"the project has no piece '{name}' of {folder}") from None
        self._prune(folder, filename)

    def _piece_file(self, folder: str, name: str) -> str:
        """
        Return the file of a piece (or a setup) by its folder and name.

        Raises
        ------
        ValueError
            If the folder is not one of pieces or setups, or the name is
            not a path in it.
        """
        if folder not in TEMPLATES:
            raise ValueError(
                f"pieces are in the folders {list(TEMPLATES)}; not {folder!r}")
        return self._file(folder, name)

    def _prune(self, folder: str, filename: str) -> None:
        """
        Remove the folders of a file gone from a folder of the project
        that it leaves empty, up to that folder.
        """
        base = Path(self._path, folder)
        parent = Path(filename).parent
        while parent != base and parent.is_relative_to(base):
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    def _file(self, folder: str, name: str) -> str:
        """
        Return the file of a name in a folder of the project (absolute).

        Raises
        ------
        ValueError
            If the name is not a path in the folder.
        """
        base = os.path.join(self._path, folder)
        filename = os.path.normpath(os.path.join(base, f'{name}.yaml'))
        if os.path.isabs(name) or not filename.startswith(base + os.sep):
            raise ValueError(
                f"the name of a piece must be a path in {folder}/; it is "
                f"'{name}'")
        return filename


def create(path: str) -> Project:
    """
    Create a project: its folder, if needed, its marker, and the folders
    of its pieces, setups, data and runs.

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
        _yaml().dump(dict(format=FORMAT, version=FORMAT_VERSION), f)
    for folder in (*TEMPLATES, 'data', 'runs'):
        os.makedirs(os.path.join(path, folder), exist_ok=True)
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
    if (version := _marker(path)['version']) > FORMAT_VERSION:
        raise ValueError(
            f"the project {path} is of version {version} of {FORMAT}, "
            f"newer than this version of gbkfit reads ({FORMAT_VERSION})")
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
    layers), and of its fitter; the file of the dataset, the PSF and the
    LSF of each observation ('data', 'psf' and 'lsf': lists, None for an
    observation without one), if any has one; the file of the noise of its
    mock data ('noise'; a list if it is one for each observation); and the
    file of the layer that gives the properties of each parameter
    ('parameters', by name).
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
        if section == 'observations':
            parts = [
                _parts(project, item, i, sources[section][i], base)
                for i, item in enumerate(loaded)]
            loaded = [observation for observation, _ in parts]
            for key in ('data', 'psf', 'lsf'):
                if any(key in found for _, found in parts):
                    sources[key] = [found.get(key) for _, found in parts]
        if section != 'params':
            config[section] = loaded
            continue
        with config_path('params'):
            config['params'], owners = merge_params(loaded)
        sources['parameters'] = {
            name: sources['params'][i] for name, i in owners.items()}
    if 'fitter' in config:
        item = config['fitter']
        config['fitter'] = _piece(project, item, 'fitter')
        sources['fitter'] = source('fitter', item)
    if isinstance(mock := config.get('mock'), Mapping) and 'noise' in mock:
        mock = config['mock'] = dict(mock)
        noise = mock['noise']
        with config_path('mock'):
            if isinstance(noise, list):
                mock['noise'] = [
                    _piece(project, item, 'noise', i)
                    for i, item in enumerate(noise)]
                sources['noise'] = [
                    None if item is None else source('noise', item)
                    for item in noise]
            else:
                mock['noise'] = _piece(project, noise, 'noise')
                sources['noise'] = source('noise', noise)
    return Resolved(config, sources)


def _parts(
        project: Project | None,
        observation: Any,
        index: int,
        source: str,
        base: str
) -> tuple[Any, dict[str, str]]:
    """
    Return an observation (the item of the section observations at an
    index) with the pieces that its options data, instrument.psf and
    instrument.lsf name in place, and the source of each of those options
    that it has, by name: the file of its piece, or that of the
    observation (source) if it is in place.
    """
    if not isinstance(observation, Mapping):
        return observation, {}
    observation, found = dict(observation), {}

    def part(options: dict[str, Any], key: str) -> None:
        value = options[key]
        options[key] = _piece(project, value, key)
        found[key] = source if not isinstance(value, str) \
            else os.path.relpath(project.piece_file(key, value), base)

    with config_path('observations', index):
        if 'data' in observation:
            part(observation, 'data')
        if isinstance(observation.get('instrument'), Mapping):
            instrument = observation['instrument'] = dict(
                observation['instrument'])
            with config_path('instrument'):
                for key in ('psf', 'lsf'):
                    if key in instrument:
                        part(instrument, key)
    return observation, found


def _piece(
        project: Project | None,
        item: Any,
        section: str,
        *index: int
) -> Any:
    """
    Return an item of a section (or an option) of a configuration: the
    piece it names, or itself.
    """
    if not isinstance(item, str):
        return item
    with config_path(section, *index):
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
        with config_path(i):
            if not isinstance(layer, Mapping):
                raise ConfigError("a layer of parameters must be a dictionary")
            for key, value in layer.items():
                with config_path(key):
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


def _marker(path: str) -> dict[str, Any]:
    """
    Return the marker of the project of a folder.

    Raises
    ------
    ValueError
        If the folder has no marker, or one that is not of the format of
        projects or has no version.
    """
    try:
        marker = _yaml().load(Path(path, MARKER))
    except FileNotFoundError:
        raise ValueError(f"{path} is not a project (no {MARKER})") from None
    if not isinstance(marker, dict) or marker.get('format') != FORMAT \
            or not isinstance(marker.get('version'), int):
        raise ValueError(
            f"{os.path.join(path, MARKER)} is not a marker of {FORMAT}")
    return marker


def _named(folder: str, content: Any) -> Iterator[tuple[Any, Any, str]]:
    """
    Yield the places of the names of pieces in the content of a piece or
    a setup of a folder (see the module), as (parent, key, folder): the
    parent (a mapping or a list) whose item of the key is a name, and the
    folder of the piece it names.
    """
    if not isinstance(content, Mapping):
        return

    def names(parent: Any, key: Any, target: str) -> Iterator[Any]:
        value = parent.get(key) if isinstance(parent, Mapping) else None
        if isinstance(value, str):
            yield parent, key, target
        elif isinstance(value, list):
            yield from (
                (value, i, target) for i, item in enumerate(value)
                if isinstance(item, str))

    def observation(options: Any) -> Iterator[Any]:
        if not isinstance(options, Mapping):
            return
        yield from names(options, 'data', 'datasets')
        for key, target in (('psf', 'psfs'), ('lsf', 'lsfs')):
            yield from names(options.get('instrument'), key, target)

    if folder == 'observations':
        yield from observation(content)
    elif folder == 'setups':
        for section in ('galaxies', 'observations', 'params'):
            yield from names(content, section, _FOLDERS[section])
        observations = content.get('observations')
        for item in observations if isinstance(observations, list) \
                else [observations]:
            yield from observation(item)
        yield from names(content, 'fitter', 'fitters')
        yield from names(content.get('mock'), 'noise', 'noise')


def _load_quietly(filename: str) -> Any:
    """Return the content of a YAML file; None if it cannot be read."""
    try:
        return _yaml().load(Path(filename))
    except (OSError, ruamel.yaml.YAMLError):
        return None


def _plain(value: Any) -> Any:
    """
    Return a value read in round trip (e.g. a CommentedMap) as plain
    values (dict, list, str, int, float, bool, None).
    """
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    for kind in (int, float, str):
        if isinstance(value, kind):
            return kind(value)
    return value


def _header(text: str) -> str:
    """Return the lines of comments that a text starts with."""
    lines = text.splitlines(keepends=True)
    count = next(
        (i for i, line in enumerate(lines) if not line.startswith('#')),
        len(lines))
    return ''.join(lines[:count])
