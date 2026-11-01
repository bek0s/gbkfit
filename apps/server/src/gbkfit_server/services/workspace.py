"""
The workspace: the folder that the server serves, with the projects under
it (see gbkfit.project), their files and their runs (see gbkfit.outputs).
The paths of the requests are relative to it, and stay in it.
"""

import dataclasses
import filecmp
import itertools
import os
import shutil
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import ruamel.yaml
from astropy.io import fits

import gbkfit.dataset
import gbkfit.outputs
import gbkfit.project
from gbkfit.utils import unitutils
from gbkfit.utils.parseutils import ConfigError

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

_yaml = ruamel.yaml.YAML(typ='safe')

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
        except (ConfigError, OSError) as e:
            raise Invalid(f"cannot create the piece '{name}': {e}") from e

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
            piece = _yaml.load(self.read_text(relative))
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
            self, relative: str, velocity: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Describe a file of data from its headers (see
        gbkfit.dataset.describe_fits), with the convention of velocities
        that its headers leave ambiguous, if given: each of its arrays (its
        index, name, shape and axes, in the order of the model, unit, the
        size of its pixels in that order, beam, the noise its header
        states, whether its velocities are ambiguous, and problems).

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
            hdus = gbkfit.dataset.describe_fits(path, velocity)
        except OSError as e:
            raise Invalid(f"cannot read '{relative}': {e}") from e
        return [
            dict(index=hdu.index, name=hdu.name, shape=list(hdu.shape),
                 axes=list(hdu.axes), unit=_readable_unit(hdu.unit),
                 step=None if hdu.coords is None
                 else [float(step) for step in hdu.coords.step],
                 beam=None if hdu.beam is None else dict(
                     bmaj=np.asarray(hdu.beam.bmaj).tolist(),
                     bmin=np.asarray(hdu.beam.bmin).tolist(),
                     bpa=np.asarray(hdu.beam.bpa).tolist()),
                 rms=hdu.rms, ambiguous_velocity=hdu.ambiguous_velocity,
                 # (the paths of the problems in the workspace)
                 problems=[p.replace(path, relative) for p in hdu.problems])
            for hdu in hdus]

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
    except ConfigError:
        return text
