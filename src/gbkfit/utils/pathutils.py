"""
Helpers for paths of files and directories.
"""

import os
import pathlib


__all__ = [
    'make_unique_path'
]


def make_unique_path(path: str | os.PathLike) -> pathlib.Path:
    """
    Return a path that does not exist yet: the path itself, or else it
    with '_1', '_2', etc. added to its name (before the extension).

    The path is not created, so another process may take it first; a
    caller that creates it should try again then.

    Parameters
    ----------
    path : str or PathLike
        The path of a file or a directory.

    Returns
    -------
    pathlib.Path
        The first of the paths that does not exist.
    """
    path = pathlib.Path(path)
    if not path.exists():
        return path
    base, ext = path.stem, path.suffix
    i = 1
    while (new_path := path.with_name(f"{base}_{i}{ext}")).exists():
        i += 1
    return new_path
