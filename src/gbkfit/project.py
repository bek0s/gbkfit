"""
Projects: the folder of a study, whose files hold the pieces of its
setups, one piece per file:

    gbkfit-project.yaml       the marker of the project (its format)
    galaxies/<name>.yaml      a galaxy (an item of the section 'galaxies')
    lenses/<name>.yaml        a lens (of 'lenses')
    observations/<name>.yaml  an observation (of 'observations')
    datasets/<name>.yaml      a dataset (an item of the section 'datasets',
                              the data of the observation at its place)
    psfs/<name>.yaml          a PSF (the option 'psf' of an instrument)
    lsfs/<name>.yaml          an LSF (the option 'lsf' of an instrument)
    noise/<name>.yaml         a noise (the option 'noise' of an observation)
    params/<name>.yaml        parameters (a section 'params')
    fitters/<name>.yaml       a fitter (a section 'fitter')
    setups/<name>.yaml        the configurations of the tasks (eval, fit)
    data/                     the data, in any files and folders
    runs/<setup>/<n>/         the runs of the setups, numbered (see
                              gbkfit.outputs)

The name of a piece is the path of its file in the folder of its kind,
without .yaml, and can have folders (e.g. 'ngc1313/hi'). In the sections
galaxies, lenses, observations, datasets, params and fitter of a
configuration in a project, and in the options noise, instrument.psf and
instrument.lsf of its observations, a string names a piece, and a
dictionary is one in place. Of the pieces, only
observations name others (their noise, PSF and LSF). The section
'params' can also be a
list of pieces, merged in order (see gbkfit.params.merge_layers); see
resolve for the pieces that items of a setup use under its names, and
the parameters of an item. The relative paths of
the configurations in a project, and of its pieces, are relative to its
folder.
"""

import io
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import ruamel.yaml

from gbkfit.config import (
    FORMAT_VERSION, ConfigError, config_path, plain_values)
from gbkfit.params import merge_layers, prefixed_params


__all__ = [
    'FORMAT',
    'MARKER',
    'TEMPLATES',
    'Project',
    'Resolved',
    'create',
    'find',
    'open',
    'piece_folder',
    'piece_sections',
    'piece_text',
    'resolve'
]


FORMAT = 'gbkfit-project'
MARKER = 'gbkfit-project.yaml'

# The folders of the pieces, by the sections and options of the
# configurations that name them (see the module)
_FOLDERS = dict(
    galaxies='galaxies', lenses='lenses', observations='observations',
    datasets='datasets', psf='psfs', lsf='lsfs', noise='noise',
    params='params', fitter='fitters')


# The sections of configurations whose items name pieces, as one or a list
# (see resolve)
_LISTED = ('galaxies', 'lenses', 'observations', 'params')


def piece_sections() -> tuple[str, ...]:
    """
    Return the sections and the options of the configurations whose items
    are pieces (see piece_folder).
    """
    return tuple(_FOLDERS)


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


def piece_text(content: Mapping[str, Any]) -> str:
    """
    Return the text of a piece (or a setup) of its content (its options),
    as a project writes it (see create_piece): YAML in block style, with
    lists and dictionaries of plain values on one line (e.g. data: {file:
    data/cube.fits, hdu: SCI}).
    """
    stream = io.StringIO()
    _yaml_out().dump(dict(content), stream)
    return stream.getvalue()


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
  rptraits: {type: exponential}
  vptraits: {type: arctan, direction: tan}
  dptraits: {type: uniform}
  emission: {type: lines}
""",
    'lenses': """\
# A gravitational lens: here a singular isothermal ellipsoid and an
# external shear at the redshift of its plane, which bend the light of the
# galaxies behind it (every galaxy then needs a redshift). Its Einstein
# radius is that of a source at source_redshift. Its parameters are named
# after it and its pieces (lens_main_*, lens_shear_*).
name: lens
redshift: 0.3
source_redshift: 1.5
pieces:
- {name: main, type: sie}
- {name: shear, type: shear}
""",
    'observations': """\
# An observation: here a cube of 48 x 48 spaxels of 1 arcsec and 40
# channels of 10 km/s of radio velocities (as of HI; optical for lines of
# ionised gas), with Gaussian PSF and LSF. To fit data, a setup pairs it
# with a dataset of the project (its section datasets, at the place of
# the observation), whose grid then replaces the grid here; the PSF and
# the LSF can be named too (psf: <name>, lsf: <name>).
type: pixel_spectra
size: [48, 48]
step: [1, 1]
spectral: {size: 40, step: 10, convention: radio}
instrument:
  psf: {type: gauss, sigma: 1.5}
  lsf: {type: gauss, sigma: 15 km/s}
""",
    'datasets': """\
# A dataset: here a cube and its errors (FITS files; paths relative to the
# project), of the type of the observations that setups pair with it
# (their section datasets, at the places of the observations).
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
# Observations name it (noise: <name>).
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
  disk_rpt_a: 1
  disk_rpt_s: 5
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
# evaluate or fit it (a fit needs a fitter, and fits the observations
# paired with data under their likelihoods: Gaussian by default, or one
# for all as below, or likelihoods by observation name). The datasets are
# the data of the observations, one for each, in their order (null for
# one without data). With several galaxies, name each in the setup and
# give each its own parameters, e.g.
#   galaxies: [{{name: source, use: ngc1}}, {{name: deflector, use: ngc2}}]
#   params: [{{of: source, use: ngc1}}, {{of: deflector, use: ngc2}}]
driver: {{type: host}}
galaxies: [{galaxies}]
# lenses: [{lenses}]
observations: [{observations}]
# datasets: [{datasets}]
params: [{params}]
# likelihoods:
#   type: gauss
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
            'galaxies', 'lenses', 'observations', 'datasets', 'psf', 'lsf',
            'noise', 'params' or 'fitter'.
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

    def runs_path(self, name: str) -> str:
        """Return the folder of the runs of a setup by name (runs/<name>)."""
        return os.path.join(self._path, 'runs', name)

    def next_run_path(self, name: str) -> str:
        """
        Return the folder of the next run of a setup by name: runs/<name>/
        <n>, n one more than the greatest number of its runs (1 for the
        first; see runs_path). It is not made: the task that writes the run
        makes it (see gbkfit.outputs.create).
        """
        folder = Path(self.runs_path(name))
        numbers = [
            int(child.name) for child in folder.glob('*')
            if child.is_dir() and child.name.isdigit()] \
            if folder.is_dir() else []
        return str(folder / str(max(numbers, default=0) + 1))

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
            f.write(text if content is None else piece_text(content))
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
                and plain_values(old) != dict(content):
            old = _yaml_rt().load(text[len(header):])
        if not isinstance(old, Mapping):
            Path(filename).write_text(piece_text(content))
            return filename
        for key in [key for key in old if key not in content]:
            del old[key]
        for key, value in content.items():
            if key not in old or plain_values(old[key]) != value:
                old[key] = value
        with Path(filename).open('w') as f:
            _yaml_rt().dump(old, f)
        return filename

    def links(self) -> list[tuple[tuple[str, str], tuple[str, str]]]:
        """
        Return the links of the pieces and setups of the project (see the
        module): each piece or setup that names a piece, with the piece it
        names, as ((folder, name), (folder, name)), once each, in the order
        of the folders and of the names. A file that cannot be read names
        none; a piece can be named and not exist (e.g. once deleted).
        """
        found: dict[tuple[tuple[str, str], tuple[str, str]], None] = {}
        for folder in ('observations', 'setups'):
            for name in self.names(folder):
                content = _load_quietly(self._file(folder, name))
                for parent, key, target in _named(folder, content):
                    found[(folder, name), (target, parent[key])] = None
        return list(found)

    def references(self, folder: str, name: str) -> list[tuple[str, str]]:
        """
        Return the pieces and setups that name a piece (see links): their
        folders and names, in order.

        Parameters
        ----------
        folder : str
            The folder of the piece (e.g. 'datasets').
        name : str
            The name of the piece.
        """
        return list(dict.fromkeys(
            source for source, target in self.links()
            if target == (folder, name)))

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
    item of its sections galaxies, lenses, observations and params (a
    list, of the layers; of a layer of an item from a list of pieces,
    their files), of each dataset of its section datasets ('datasets', a
    list in the order of the observations, None for an observation
    without data), and of its fitter; the file of the noise, the PSF
    and the LSF of each observation ('noise', 'psf' and 'lsf': lists,
    None for an observation without one), if any has one; and the file of
    the layer that gives the properties of each parameter ('parameters',
    by name).
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
    layers of its section 'params', if it is a list (see merge_layers).

    An item of the sections galaxies, lenses and observations is the
    piece it names, or {name: <name>, use: <piece>}: the piece under a
    name of the configuration (e.g. a galaxy that is the deflector of a
    lens in one setup, and alone in another). An item of the section
    datasets is the piece it names, one in place, or null (an observation
    without data). A layer of params is the
    piece it names or one in place, in the names of the configuration,
    or the parameters of an item of it: {of: <item>, use: <piece or
    list of pieces>}, or {of: <item>, ...} with the options of params in
    place, in the names of the item (e.g. gas_vsys), which its prefix
    makes those of the configuration (e.g. deflector_gas_vsys; see
    prefixed_params): the parameters of a galaxy are the same in every
    setup it is in.

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
        If it names pieces outside a project, or unknown pieces, a piece
        used under another name than its own, a layer of params of an
        item that the configuration has not (by name), or the layers of
        'params' are invalid.
    """
    base = os.path.dirname(filename) if project is None else project.path
    config = dict(config)
    sources: dict[str, Any] = dict(file=os.path.relpath(filename, base))

    def source(section: str, item: Any) -> str:
        path = filename if not isinstance(item, str) \
            else project.piece_file(section, item)
        return os.path.relpath(path, base)

    for section in _LISTED:
        items = config.get(section)
        # (one item: a name, or a dictionary but of params, whose one
        # dictionary is the section itself)
        if isinstance(items, str) or (
                section != 'params' and isinstance(items, Mapping)):
            items = [items]
        if not isinstance(items, list):
            continue
        if section == 'params':
            config['params'], sources['params'], sources['parameters'] \
                = _layers(project, items, _prefixes(config), source)
            continue
        loaded = [
            _named_piece(project, item, section, i)
            for i, item in enumerate(items)]
        sources[section] = [
            source(section, item.get('use') if _uses(item) else item)
            for item in items]
        if section == 'observations':
            parts = [
                _parts(project, item, i, sources[section][i], base)
                for i, item in enumerate(loaded)]
            loaded = [observation for observation, _ in parts]
            for key in ('noise', 'psf', 'lsf'):
                if any(key in found for _, found in parts):
                    sources[key] = [found.get(key) for _, found in parts]
        config[section] = loaded
    if (datasets := config.get('datasets')) is not None:
        # (one, or one for each observation, None for one without data)
        if not isinstance(datasets, list):
            datasets = [datasets]
        config['datasets'] = [
            None if item is None else _piece(project, item, 'datasets', i)
            for i, item in enumerate(datasets)]
        sources['datasets'] = [
            None if item is None else source('datasets', item)
            for item in datasets]
    if 'fitter' in config:
        item = config['fitter']
        config['fitter'] = _piece(project, item, 'fitter')
        sources['fitter'] = source('fitter', item)
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
    index) with the pieces that its options noise, instrument.psf and
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
        if 'noise' in observation:
            part(observation, 'noise')
        if isinstance(observation.get('instrument'), Mapping):
            instrument = observation['instrument'] = dict(
                observation['instrument'])
            with config_path('instrument'):
                for key in ('psf', 'lsf'):
                    if key in instrument:
                        part(instrument, key)
    return observation, found


def _uses(item: Any) -> bool:
    """
    Check whether an item of a section uses a piece under a name of the
    configuration ({name: <name>, use: <piece>}; see resolve).
    """
    return isinstance(item, Mapping) and 'use' in item


def _named_piece(
        project: Project | None,
        item: Any,
        section: str,
        index: int
) -> Any:
    """
    Return an item of a section of a configuration: the piece it names,
    the piece it uses under a name of the configuration (see resolve), or
    itself.
    """
    if not _uses(item):
        return _piece(project, item, section, index)
    with config_path(section, index):
        if set(item) != {'name', 'use'} or not isinstance(item['use'], str):
            raise ConfigError(
                "an item that uses a piece gives only the piece (use) and "
                "its name (name)")
        piece = _piece(project, item['use'], section, index)
        if not isinstance(piece, Mapping):
            return piece
        own = piece.get('name')
        if own is not None and own != item['name']:
            raise ConfigError(
                f"the piece {item['use']!r} is named {own!r}; it cannot be "
                f"used as {item['name']!r} (take the name out of the piece, "
                f"for the setups to name it)")
        return dict(piece) | dict(name=item['name'])


def _prefixes(config: Mapping[str, Any]) -> dict[str, str]:
    """
    Return the prefixes of the names of the parameters of the named items
    of a configuration (its galaxies, lenses and observations, their
    pieces in place), by their names: each its name and '_' (see
    ObservationGroup).
    """
    return {
        item['name']: f"{item['name']}_"
        for section in ('galaxies', 'lenses', 'observations')
        for item in _listed_items(config.get(section))
        if isinstance(item, Mapping) and isinstance(item.get('name'), str)}


def _listed_items(value: Any) -> list[Any]:
    """Return the items of a section: a list, or one item."""
    return value if isinstance(value, list) else [value]


def _layers(
        project: Project | None,
        items: Sequence[Any],
        prefixes: Mapping[str, str],
        source: Any
) -> tuple[dict[str, Any], list[Any], dict[str, str]]:
    """
    Return the merged layers of the section params of a configuration
    (see resolve and merge_layers), the source of each item (a file, or of
    the parameters of an item from pieces, their files), and the source
    of the properties of each parameter, by name.
    """
    layers, files, sources = [], [], []
    for i, item in enumerate(items):
        if not (isinstance(item, Mapping) and 'of' in item):
            layers.append(_piece(project, item, 'params', i))
            files.append(source('params', item))
            sources.append(files[-1])
            continue
        with config_path('params', i):
            of = item['of']
            if of not in prefixes:
                raise ConfigError(
                    f"the parameters are of {of!r}, which names no galaxy, "
                    f"lens or observation of the configuration (by name: "
                    f"{sorted(prefixes)})")
            rest = {k: v for k, v in item.items() if k not in ('of', 'use')}
            if 'use' in item and rest:
                raise ConfigError(
                    "the parameters of an item are from pieces (use) or in "
                    "place, not both")
            uses = item.get('use')
            uses = [uses] if isinstance(uses, str) else uses
            if uses is not None and not (
                    isinstance(uses, list)
                    and all(isinstance(u, str) for u in uses)):
                raise ConfigError(
                    "use names a params piece, or a list of them")
            blocks = [rest] if uses is None else [
                _piece(project, use, 'params', i) for use in uses]
            used = [source('params', None)] if uses is None else [
                source('params', use) for use in uses]
            for block, file in zip(blocks, used):
                if not isinstance(block, Mapping):
                    raise ConfigError(
                        "a layer of parameters must be a dictionary")
                try:
                    layers.append(prefixed_params(block, prefixes[of]))
                except ValueError as e:
                    raise ConfigError(str(e)) from None
                sources.append(file)
            files.append(
                used if isinstance(item.get('use'), list) else used[0])
    with config_path('params'):
        merged, owners = merge_layers(layers)
    return merged, files, {
        name: sources[layer] for name, layer in owners.items()}


def _piece(
        project: Project | None,
        item: Any,
        section: str,
        *index: int | str
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
        elif isinstance(value, Mapping):
            yield from names(value, 'use', target)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, str):
                    yield value, i, target
                elif isinstance(item, Mapping):
                    # (a piece under a name of the setup, or parameters of
                    # an item from pieces: see resolve)
                    yield from names(item, 'use', target)

    def observation(options: Any) -> Iterator[Any]:
        if not isinstance(options, Mapping):
            return
        yield from names(options, 'noise', 'noise')
        for key, target in (('psf', 'psfs'), ('lsf', 'lsfs')):
            yield from names(options.get('instrument'), key, target)

    if folder == 'observations':
        yield from observation(content)
    elif folder == 'setups':
        for section in _LISTED:
            yield from names(content, section, _FOLDERS[section])
        observations = content.get('observations')
        for item in observations if isinstance(observations, list) \
                else [observations]:
            yield from observation(item)
        yield from names(content, 'datasets', 'datasets')
        yield from names(content, 'fitter', 'fitters')


def _load_quietly(filename: str) -> Any:
    """Return the content of a YAML file; None if it cannot be read."""
    try:
        return _yaml().load(Path(filename))
    except (OSError, ruamel.yaml.YAMLError):
        return None


def _header(text: str) -> str:
    """Return the lines of comments that a text starts with."""
    lines = text.splitlines(keepends=True)
    count = next(
        (i for i, line in enumerate(lines) if not line.startswith('#')),
        len(lines))
    return ''.join(lines[:count])
