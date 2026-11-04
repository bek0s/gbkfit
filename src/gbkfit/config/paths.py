"""
The files of configurations: relative paths, which are relative to the
directory of the configuration, and the options of FITS and ASDF
files.
"""

import contextlib
import contextvars
import os
from collections.abc import Generator, Mapping
from typing import Any

from .errors import ConfigError
from .options import parse_options
from .types import is_asdf_file


__all__ = [
    'parse_file',
    'paths_relative_to',
    'resolve_path'
]


# The directory that relative paths in configurations are relative to,
# while they are loaded (see paths_relative_to)
_base_dir: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    'base_dir', default=None)


@contextlib.contextmanager
def paths_relative_to(directory: str) -> Generator[None]:
    """
    Resolve the relative paths of configurations (see resolve_path)
    against a directory, e.g. that of the configuration file, while
    loading them.

    Parameters
    ----------
    directory : str
        The directory.
    """
    token = _base_dir.set(os.path.abspath(directory))
    try:
        yield
    finally:
        _base_dir.reset(token)


def resolve_path(path: str) -> str:
    """
    Return a path of a configuration: a relative path is relative to the
    directory of paths_relative_to, if any (and to the current directory
    otherwise).
    """
    base = _base_dir.get()
    if base is None or os.path.isabs(path):
        return path
    return os.path.join(base, path)


def parse_data_file(x: str | Mapping[str, Any]) -> str | tuple[str, int | str]:
    """
    Parse a file of data (see DataFile): a filename, or a dict
    with the filename ('file') and the HDU of a FITS file ('hdu') or the
    path of the array in the tree of an ASDF file ('path').

    Raises
    ------
    ConfigError
        If the dict has no filename, or the key of the array is not that
        of the kind of the file.
    """
    if isinstance(x, str):
        return resolve_path(x)
    options = parse_options(
        dict(x), required={'file'}, optional={'hdu', 'path'})
    file = resolve_path(options['file'])
    asdf = is_asdf_file(file)
    key, wrong = ('path', 'hdu') if asdf else ('hdu', 'path')
    if wrong in options:
        raise ConfigError(
            f"{file}: the array of {'an ASDF' if asdf else 'a FITS'} file is "
            f"given by its {key}, not by its {wrong}")
    return (file, options[key]) if key in options else file


def parse_file(x: str | dict[str, Any]) -> tuple[str, int | str]:
    """
    Parse a file option (a relative filename is resolved by resolve_path).

    Parameters
    ----------
    x : str or dict
        A filename, or a dict with the filename ('file') and, optionally,
        the HDU ('hdu', e.g. 'SCI').

    Returns
    -------
    tuple of str and (int or str)
        The filename and the HDU (by default the first, 0).

    Raises
    ------
    ConfigError
        If the dict has no filename.
    """
    if isinstance(x, str):
        x = dict(file=x)
    options = parse_options(x, required={'file'}, optional={'hdu'})
    return resolve_path(options['file']), options.get('hdu', 0)
