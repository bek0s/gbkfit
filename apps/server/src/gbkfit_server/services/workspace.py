"""
The workspace: the folder that the server serves, with the projects under
it (see gbkfit.project), their files and their runs (see gbkfit.outputs).
The paths of the requests are relative to it, and stay in it.
"""

import dataclasses
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

import gbkfit.outputs
import gbkfit.project
from gbkfit.utils.parseutils import ConfigError

from .errors import Invalid, NotFound


__all__ = [
    'ProjectInfo',
    'Workspace'
]


# The folders not searched for projects: of their data and runs, and
# hidden ones
_SKIPPED = ('data', 'runs', 'node_modules', '__pycache__')

# The folders of the pieces and of the setups of a project
_PIECES = ('galaxies', 'observations', 'params', 'fitters', 'setups')


@dataclasses.dataclass
class ProjectInfo:
    """
    A project: its path, the names of its pieces and setups by folder,
    and the paths of its runs.
    """
    path: str
    pieces: dict[str, list[str]]
    runs: list[str]


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

    def create_piece(self, project: str, folder: str, name: str) -> str:
        """
        Create a piece (or a setup) of a project from its template (see
        gbkfit.project.Project.create_piece), and return its path.

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
            return self.relative(target.create_piece(folder, name))
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
        temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
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
