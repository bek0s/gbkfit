"""
The workspace: the folder that the server serves, with the projects under
it (see gbkfit.project), their files and their runs (see gbkfit.outputs).
The paths of the requests are relative to it, and stay in it.
"""

import dataclasses
import datetime
import filecmp
import itertools
import json
import os
import shutil
import threading
import uuid
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import astropy.units
import astropy.wcs
import astropy.wcs.utils
import numpy as np
import ruamel.yaml
from astropy.io import fits

import gbkfit.dataset
import gbkfit.outputs
import gbkfit.project
from gbkfit.utils import unitutils

from .errors import Invalid, NotFound


__all__ = [
    'FolderInfo',
    'ProjectInfo',
    'Workspace'
]


# The folders not searched for projects: of their data and runs, and
# hidden ones
_SKIPPED = ('data', 'runs', 'node_modules', '__pycache__')

# The endings of the names of the files of data that GBKFIT reads (FITS)
_DATA_FILES = ('.fits', '.fit', '.fts', '.fits.gz')

# The record of how the files of a folder of data were prepared (see
# Workspace.record_preparation), in that folder
_PREPARED = '.prepared.json'

# The options of prep that name files of data (a path, or a path and an
# HDU, {file, hdu}; of moment maps, a list of them)
_PREP_FILES = ('file_d', 'file_e', 'file_m')

# The folders of the pieces and of the setups of a project
_PIECES = (
    'galaxies', 'observations', 'datasets', 'psfs', 'lsfs', 'noise',
    'params', 'fitters', 'setups')


@dataclasses.dataclass
class ProjectInfo:
    """
    A project: its path, the names of its pieces and setups by folder,
    and the paths of its runs.
    """
    path: str
    pieces: dict[str, list[str]]
    runs: list[str]


@dataclasses.dataclass
class FolderInfo:
    """
    A folder: its path, and the names of its folders and the names and
    sizes (bytes) of its files ({name, size}), but the hidden ones.
    """
    path: str
    folders: list[str]
    files: list[dict[str, Any]]


class Workspace:
    """
    The folder that the server serves.

    Parameters
    ----------
    root : str
        The folder.

    Raises
    ------
    NotADirectoryError
        If the folder does not exist.
    """

    def __init__(self, root: str):
        self._root = os.path.realpath(root)
        if not os.path.isdir(self._root):
            raise NotADirectoryError(f"{root} is not a folder")
        # (the records of preparations are read and written one at a time)
        self._records = threading.Lock()

    @property
    def root(self) -> str:
        """The folder (absolute)."""
        return self._root

    def path(self, relative: str) -> str:
        """
        Return the absolute path of a path relative to the workspace.

        Raises
        ------
        Invalid
            If the path is out of the workspace.
        """
        path = os.path.realpath(os.path.join(self._root, relative))
        if path != self._root and not path.startswith(self._root + os.sep):
            raise Invalid(f"the path '{relative}' is out of the workspace")
        return path

    def relative(self, path: str) -> str:
        """Return a path in the workspace relative to it (with '/')."""
        return Path(os.path.relpath(path, self._root)).as_posix()

    def projects(self) -> list[str]:
        """Return the paths of the projects in the workspace."""
        found = []
        for folder, folders, files in os.walk(self._root):
            if gbkfit.project.MARKER in files:
                found.append(self.relative(folder))
                folders.clear()
                continue
            folders[:] = sorted(
                name for name in folders
                if name not in _SKIPPED and not name.startswith('.'))
        return sorted(found)

    def project(self, relative: str) -> ProjectInfo:
        """
        Return a project.

        Raises
        ------
        NotFound
            If the folder is not a project.
        """
        path = self.path(relative)
        try:
            gbkfit.project.open(path)
        except ValueError as e:
            raise NotFound(str(e)) from e
        pieces = {
            folder: sorted(
                file.relative_to(Path(path, folder)).with_suffix('')
                .as_posix()
                for file in Path(path, folder).rglob('*.yaml'))
            for folder in _PIECES}
        runs = sorted(
            self.relative(manifest.parent)
            for manifest in Path(path, 'runs').rglob(gbkfit.outputs.MANIFEST))
        return ProjectInfo(self.relative(path), pieces, runs)

    def create_project(self, relative: str) -> ProjectInfo:
        """
        Create a project (see gbkfit.project.create).

        Raises
        ------
        Invalid
            If the folder is a project, or in one.
        """
        path = self.path(relative)
        try:
            if (outer := gbkfit.project.find(path)) is not None:
                raise Invalid(
                    f"'{relative}' is in the project "
                    f"'{self.relative(outer.path)}'")
            gbkfit.project.create(path)
        except (OSError, ValueError) as e:
            raise Invalid(f"cannot create the project '{relative}': {e}") \
                from e
        return self.project(relative)

    def create_piece(
            self,
            project: str,
            folder: str,
            name: str,
            content: dict[str, Any] | None = None
    ) -> str:
        """
        Create a piece (or a setup) of a project with its content, or from
        its template (see gbkfit.project.Project.create_piece), and return
        its path.

        Raises
        ------
        NotFound
            If the project does not exist.
        Invalid
            If the folder or the name is invalid, or the piece exists.
        """
        try:
            target = gbkfit.project.open(self.path(project))
        except ValueError as e:
            raise NotFound(str(e)) from e
        try:
            return self.relative(target.create_piece(folder, name, content))
        except (ValueError, OSError) as e:
            raise Invalid(f"cannot create the piece '{name}': {e}") from e

    def rename_project(self, relative: str, to: str) -> ProjectInfo:
        """
        Move a project to another folder of the workspace (its pieces and
        setups name their files relative to it, so they follow).

        Raises
        ------
        NotFound
            If the project does not exist.
        Invalid
            If it is the workspace itself, or the new folder exists, or is
            in a project.
        """
        source = self._project_path(relative)
        target = self.path(to)
        if source == self._root:
            raise Invalid("the workspace itself cannot move")
        if os.path.exists(target):
            raise Invalid(f"'{to}' exists")
        try:
            outer = gbkfit.project.find(os.path.dirname(target))
        except ValueError as e:
            raise Invalid(str(e)) from e
        if outer is not None:
            raise Invalid(
                f"'{to}' is in the project '{self.relative(outer.path)}'")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.rename(source, target)
        return self.project(self.relative(target))

    def delete_project(self, relative: str) -> None:
        """
        Delete a project, and all its files: its data and runs too.

        Raises
        ------
        NotFound
            If the project does not exist.
        Invalid
            If it is the workspace itself.
        """
        path = self._project_path(relative)
        if path == self._root:
            raise Invalid("the workspace itself cannot be deleted")
        shutil.rmtree(path)

    def references(self, relative: str) -> list[dict[str, str]]:
        """
        Return the pieces and setups of a project that name one of its
        pieces (see gbkfit.project.Project.references), by its path: their
        folders and names ({folder, name}).

        Raises
        ------
        NotFound
            If the piece is not one of a project.
        """
        target, folder, name = self._piece_of(relative)
        return [
            dict(folder=other, name=other_name)
            for other, other_name in target.references(folder, name)]

    def rename_piece(self, relative: str, name: str) -> str:
        """
        Rename a piece (or a setup) of a project by its path, and the
        pieces and setups that name it with it (see
        gbkfit.project.Project.rename_piece); return its new path.

        Raises
        ------
        NotFound
            If the piece is not one of a project.
        Invalid
            If the name is invalid, or a piece of it exists.
        """
        target, folder, old = self._piece_of(relative)
        try:
            return self.relative(target.rename_piece(folder, old, name))
        except (ValueError, FileExistsError) as e:
            raise Invalid(f"cannot rename the piece '{old}': {e}") from e

    def update_piece(self, relative: str, content: dict[str, Any]) -> None:
        """
        Write the content of a piece (or a setup) of a project by its path
        (see gbkfit.project.Project.update_piece: its comments stay).

        Raises
        ------
        NotFound
            If the piece is not one of a project.
        """
        target, folder, name = self._piece_of(relative)
        target.update_piece(folder, name, content)

    def delete_piece(self, relative: str) -> None:
        """
        Delete a piece (or a setup) of a project by its path; its files of
        data stay.

        Raises
        ------
        NotFound
            If the piece is not one of a project.
        """
        target, folder, name = self._piece_of(relative)
        target.delete_piece(folder, name)

    def _project_path(self, relative: str) -> str:
        """
        Return the folder of a project (absolute).

        Raises
        ------
        NotFound
            If the folder is not a project.
        """
        path = self.path(relative)
        try:
            gbkfit.project.open(path)
        except ValueError as e:
            raise NotFound(str(e)) from e
        return path

    def _piece_of(
            self, relative: str
    ) -> tuple[gbkfit.project.Project, str, str]:
        """
        Return the project of a piece (or a setup) by its path, and its
        folder and name.

        Raises
        ------
        NotFound
            If the file is not a piece of a project.
        """
        path = self.path(relative)
        try:
            target = gbkfit.project.find(path)
        except ValueError as e:
            raise NotFound(str(e)) from e
        parts = Path(os.path.relpath(path, target.path)).parts \
            if target is not None else ()
        if len(parts) < 2 or parts[0] not in _PIECES \
                or not parts[-1].endswith('.yaml') \
                or not os.path.isfile(path):
            raise NotFound(f"no piece '{relative}'")
        name = Path(*parts[1:]).with_suffix('').as_posix()
        return target, parts[0], name

    def read_text(self, relative: str) -> str:
        """
        Return the text of a file.

        Raises
        ------
        NotFound
            If the file does not exist.
        """
        try:
            return Path(self.path(relative)).read_text()
        except (FileNotFoundError, IsADirectoryError) as e:
            raise NotFound(f"no file '{relative}'") from e

    def piece(self, relative: str) -> dict[str, Any]:
        """
        Return a piece (or a setup) of a project: the options of its file.

        Raises
        ------
        NotFound
            If the file does not exist.
        Invalid
            If it is not YAML, or not a dictionary.
        """
        try:
            piece = ruamel.yaml.YAML(typ='safe').load(
                self.read_text(relative))
        except ruamel.yaml.YAMLError as e:
            raise Invalid(f"cannot read '{relative}': {e}") from e
        if not isinstance(piece, dict):
            raise Invalid(f"'{relative}' is not a dictionary of options")
        return piece

    def folder(self, relative: str) -> FolderInfo:
        """
        Return a folder.

        Raises
        ------
        NotFound
            If the folder does not exist.
        """
        path = Path(self.path(relative))
        if not path.is_dir():
            raise NotFound(f"no folder '{relative}'")
        entries = sorted(
            entry for entry in path.iterdir()
            if not entry.name.startswith('.'))
        return FolderInfo(
            self.relative(str(path)),
            [entry.name for entry in entries if entry.is_dir()],
            [dict(name=entry.name, size=entry.stat().st_size)
             for entry in entries if entry.is_file()])

    def describe(
            self, relative: str, velocity: str | None = None,
            rest: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Describe a file of data from its headers (see
        gbkfit.dataset.describe_fits), with the convention of velocities
        that its headers leave ambiguous and the rest of its spectral axis
        (which makes wavelengths or frequencies velocities), if given: each
        of its arrays (its index, name, shape and axes, in the order of the
        model, unit, the size of its pixels in that order (only the
        spatial ones while its spectral axis cannot be read: velocities of
        no convention, or wavelengths without a rest), beam, the noise its
        header states, whether its velocities are ambiguous, and problems).

        Raises
        ------
        NotFound
            If the file does not exist.
        Invalid
            If it is not a file of data that GBKFIT reads.
        """
        path = self.file(relative)
        if not path.lower().endswith(_DATA_FILES):
            raise Invalid(
                f"'{relative}' is not a file of data that GBKFIT reads: "
                f"FITS ({', '.join(_DATA_FILES)})")
        try:
            hdus = gbkfit.dataset.describe_fits(path, velocity, rest)
        except OSError as e:
            raise Invalid(f"cannot read '{relative}': {e}") from e
        return [
            dict(index=hdu.index, name=hdu.name, shape=list(hdu.shape),
                 axes=list(hdu.axes), unit=_readable_unit(hdu.unit),
                 step=_spatial_step(path, hdu.index) if hdu.coords is None
                 else [float(step) for step in hdu.coords.step],
                 beam=None if hdu.beam is None else dict(
                     bmaj=np.asarray(hdu.beam.bmaj).tolist(),
                     bmin=np.asarray(hdu.beam.bmin).tolist(),
                     bpa=np.asarray(hdu.beam.bpa).tolist()),
                 rms=hdu.rms, ambiguous_velocity=hdu.ambiguous_velocity,
                 # (the paths of the problems in the workspace)
                 problems=[p.replace(path, relative) for p in hdu.problems])
            for hdu in hdus]

    def record_preparation(
            self, task: str, options: dict[str, Any],
            files: dict[str, str | list[str | None]]
    ) -> None:
        """
        Record how files of data were prepared: the task of prep, its
        options (those of its files, paths relative to the workspace) and
        the files it prepared (by the options of their files, as those of
        a job; relative to the workspace), with the time, in the record of
        the folder of the prepared files, under the name of each prepared
        file of data (see preparation). Its paths are kept relative to that
        folder, so that the record follows the project as it moves.
        """
        data = files['file_d']
        first = data if isinstance(data, str) else next(
            path for path in data if path is not None)
        folder = Path(self.path(first)).parent

        def kept(path: str) -> str:
            return Path(os.path.relpath(self.path(path), folder)).as_posix()
        entry = dict(
            task=task,
            options={
                key: _file_paths(value, kept) if key in _PREP_FILES
                else value
                for key, value in options.items()
                if key not in ('output_dir', 'output_dir_mode')},
            files={
                key: _file_paths(value, kept) for key, value in files.items()},
            time=datetime.datetime.now(datetime.UTC).isoformat(
                timespec='seconds'))
        names = [data] if isinstance(data, str) else data
        with self._records:
            records = _read_records(folder)
            for name in names:
                if name is not None:
                    records[Path(name).name] = entry
            _write_records(folder, records)

    def preparation(self, relative: str) -> dict[str, Any] | None:
        """
        Return how a file of data was prepared, if a record of its folder
        has it (see record_preparation): the task, its options and the
        files it prepared (paths relative to the workspace), and when;
        None if the file was not prepared by the server (e.g. an imported
        file, or one prepared by the command line).

        Raises
        ------
        NotFound
            If the file does not exist.
        """
        path = Path(self.file(relative))
        entry = _read_records(path.parent).get(path.name)
        if entry is None:
            return None

        def workspace(name: str) -> str:
            return self.relative(os.path.normpath(path.parent / name))
        return dict(
            task=entry['task'],
            options={
                key: _file_paths(value, workspace) if key in _PREP_FILES
                else value
                for key, value in entry['options'].items()},
            files={
                key: _file_paths(value, workspace)
                for key, value in entry['files'].items()},
            time=entry['time'])

    def delete_data(self, relative: str) -> None:
        """
        Delete a file of data of a project (one of its folder data), if
        no piece, setup or run of the project names it (by its name, as
        a careful check), and forget how it was prepared.

        Raises
        ------
        NotFound
            If the file does not exist.
        Invalid
            If it is not a file of data of the folder data of a project,
            or the project names it.
        """
        path = Path(self.file(relative))
        if not path.name.lower().endswith(_DATA_FILES) \
                or path.parent.name != 'data':
            raise Invalid(
                f"'{relative}' is not a file of data of the folder data of "
                f"a project")
        project = path.parent.parent
        try:
            gbkfit.project.open(str(project))
        except ValueError as e:
            raise Invalid(
                f"'{relative}' is not a file of data of a project") from e
        for text in project.rglob('*.yaml'):
            if any(part.startswith('.') for part in
                   text.relative_to(project).parts):
                continue
            if path.name in text.read_text(errors='replace'):
                raise Invalid(
                    f"'{relative}' is in use: "
                    f"{text.relative_to(project).as_posix()} names it")
        path.unlink()
        with self._records:
            records = _read_records(path.parent)
            if records.pop(path.name, None) is not None:
                _write_records(path.parent, records)

    def import_data(self, project: str, relative: str) -> str:
        """
        Copy a file of data of the workspace into a project (its folder
        data), and return the path of the copy (see _keep); a file of that
        folder is not copied.

        Raises
        ------
        NotFound
            If the project or the file does not exist.
        Invalid
            If it is not a file of data that GBKFIT reads.
        """
        source = Path(self.file(relative))
        folder = self._data_folder(project)
        if source.is_relative_to(folder):
            return self.relative(str(source))
        temporary = self._temporary_data(folder, source.name)
        try:
            shutil.copyfile(source, temporary)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return self._keep(temporary, source.name)

    async def receive_data(
            self, project: str, name: str, chunks: AsyncIterator[bytes]
    ) -> str:
        """
        Write a file of data received in chunks (e.g. from the computer of
        a page) into a project (its folder data), by its name, and return
        the path of the file (see _keep).

        Raises
        ------
        NotFound
            If the project does not exist.
        Invalid
            If the name is not that of a file of data that GBKFIT reads.
        """
        temporary = self._temporary_data(self._data_folder(project), name)
        try:
            with temporary.open('wb') as f:
                async for chunk in chunks:
                    f.write(chunk)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return self._keep(temporary, name)

    def keep_data(self, filename: str, folder: str) -> str:
        """
        Move a new file of data (e.g. one that prep wrote) into a folder,
        by its name, as imported files are kept (see _keep), and return
        its path.

        Raises
        ------
        Invalid
            If it is not a file of data that GBKFIT reads.
        """
        name = Path(filename).name
        temporary = self._temporary_data(Path(folder), name)
        os.replace(filename, temporary)
        return self._keep(temporary, name)

    def _data_folder(self, project: str) -> Path:
        """
        Return the folder data of a project.

        Raises
        ------
        NotFound
            If the project does not exist.
        """
        try:
            return Path(gbkfit.project.open(self.path(project)).path, 'data')
        except ValueError as e:
            raise NotFound(str(e)) from e

    def _temporary_data(self, folder: Path, name: str) -> Path:
        """
        Return a temporary file of a file of data of a folder (made if
        needed) by its name.

        Raises
        ------
        Invalid
            If the name is not that of a file of data that GBKFIT reads.
        """
        if name != Path(name).name or not name.lower().endswith(_DATA_FILES):
            raise Invalid(
                f"'{name}' is not the name of a file of data that GBKFIT "
                f"reads: FITS ({', '.join(_DATA_FILES)})")
        folder.mkdir(exist_ok=True)
        return _temporary(folder / name)

    def _keep(self, temporary: Path, name: str) -> str:
        """
        Keep a new file of data (a temporary file of its folder) by its
        name, and return its path: that of the file of the name already
        there if it is the same, or else of the name, or of the first free
        one numbered after it (e.g. cube-2.fits).
        """
        try:
            for target in _numbered(temporary.parent, name):
                if not target.exists():
                    os.replace(temporary, target)
                    return self.relative(str(target))
                if filecmp.cmp(temporary, target, shallow=False):
                    return self.relative(str(target))
        finally:
            temporary.unlink(missing_ok=True)
        raise AssertionError("unreachable: the names are endless")

    def file(self, relative: str) -> str:
        """
        Return the absolute path of a file.

        Raises
        ------
        NotFound
            If the file does not exist.
        """
        path = self.path(relative)
        if not os.path.isfile(path):
            raise NotFound(f"no file '{relative}'")
        return path

    def write_text(self, relative: str, text: str) -> None:
        """
        Write the text of a file, and its folders: atomically, so that
        readers see the old text or the new one.
        """
        path = Path(self.path(relative))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = _temporary(path)
        temporary.write_text(text)
        os.replace(temporary, path)

    def run(self, relative: str) -> dict[str, Any]:
        """
        Return a run: its manifest.

        Raises
        ------
        NotFound
            If the folder is not a run.
        """
        try:
            run = gbkfit.outputs.open(self.path(relative))
        except ValueError as e:
            raise NotFound(str(e)) from e
        return dict(
            path=self.relative(run.path), task=run.task, options=run.options,
            status=run.status, gbkfit_version=run.gbkfit_version,
            created=run.created, updated=run.updated,
            products=[p.dump() for p in run.products()])

    def array(self, relative: str, plane: int | None = None) -> np.ndarray:
        """
        Return the data of a FITS file as float32, or one plane of them
        (along their first axis).

        Raises
        ------
        NotFound
            If the file does not exist.
        Invalid
            If it has no data, or not the plane.
        """
        try:
            data = fits.getdata(self.path(relative))
        except FileNotFoundError as e:
            raise NotFound(f"no file '{relative}'") from e
        except (OSError, IndexError) as e:
            raise Invalid(f"'{relative}' has no data: {e}") from e
        if plane is not None:
            if not 0 <= plane < data.shape[0] or data.ndim < 2:
                raise Invalid(
                    f"'{relative}' of shape {data.shape} has no plane {plane}")
            data = data[plane]
        return np.ascontiguousarray(data, dtype=np.float32)


def _temporary(path: Path) -> Path:
    """
    Return a temporary file beside a file, which is written first and then
    replaces it, so that readers see the old file or the whole new one.
    """
    return path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')


def _numbered(folder: Path, name: str) -> Iterator[Path]:
    """
    Yield the files of a name of a file of data in a folder: itself, then
    numbered (e.g. cube.fits, cube-2.fits, cube-3.fits).
    """
    ending = next(e for e in _DATA_FILES if name.lower().endswith(e))
    stem, ending = name[:-len(ending)], name[-len(ending):]
    yield folder / name
    for number in itertools.count(2):
        yield folder / f'{stem}-{number}{ending}'


def _spatial_step(path: str, index: int) -> list[float] | None:
    """
    Return the size of the pixels of an array of a file on the sky
    (arcsec; x, then y) from the celestial axes of its header, for an
    array whose coordinates GBKFIT cannot read yet (e.g. velocities of no
    convention, or wavelengths without a rest); None if its header has no
    celestial axes.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            celestial = astropy.wcs.WCS(fits.getheader(path, index)).celestial
            if celestial.naxis != 2:
                return None
            scales = astropy.wcs.utils.proj_plane_pixel_scales(celestial)
            return [
                float(astropy.units.Quantity(
                    scale, unit or astropy.units.deg).to_value('arcsec'))
                for scale, unit in zip(scales, celestial.wcs.cunit)]
    except Exception:
        # (any error of astropy: no size)
        return None


def _file_paths(value: Any, change: Callable[[str], str]) -> Any:
    """
    Return a file option of prep (a path, or a path and an HDU, {file,
    hdu}; a list of them, or None) with its paths changed.
    """
    if isinstance(value, str):
        return change(value)
    if isinstance(value, dict):
        return value | dict(file=change(value['file']))
    if isinstance(value, list):
        return [_file_paths(item, change) for item in value]
    return value


def _read_records(folder: Path) -> dict[str, Any]:
    """
    Return the records of the preparations of the files of a folder, by
    the names of their files (none if it has none, or if they cannot be
    read).
    """
    try:
        records = json.loads((folder / _PREPARED).read_text())
    except (OSError, ValueError):
        return {}
    return records.get('files', {}) if isinstance(records, dict) else {}


def _write_records(folder: Path, records: dict[str, Any]) -> None:
    """Write the records of the preparations of a folder, atomically."""
    path = folder / _PREPARED
    temporary = _temporary(path)
    temporary.write_text(json.dumps(dict(version=1, files=records), indent=1))
    os.replace(temporary, path)


def _readable_unit(text: str | None) -> str | None:
    """
    Return the unit of the values of an array (BUNIT, as its header writes
    it, e.g. JY/BEAM) as astropy writes it, to read (e.g. Jy / beam), or as
    written if it is not a unit.
    """
    if text is None:
        return None
    try:
        return unitutils.parse_unit(text).to_string()
    except ValueError:
        return text
