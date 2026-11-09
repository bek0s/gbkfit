from typing import TYPE_CHECKING, Any

from gbkfit._version import __version__

if TYPE_CHECKING:
    from gbkfit.config import ConfigError, ConfigProblem
    from gbkfit.setup import (
        Description, Setup, catalogue, describe, load, schema)


__all__ = [
    'ConfigError',
    'ConfigProblem',
    'Description',
    'Setup',
    '__version__',
    'catalogue',
    'describe',
    'load',
    'schema'
]


def __getattr__(name: str) -> Any:
    # The front door (see gbkfit.setup), imported on first use: it imports
    # the whole library, which a plain import of gbkfit (e.g. by its
    # command line) does not need
    if name in (
            'Description', 'Setup', 'catalogue', 'describe', 'load',
            'schema'):
        import gbkfit.setup
        return getattr(gbkfit.setup, name)
    if name in ('ConfigError', 'ConfigProblem'):
        import gbkfit.config
        return getattr(gbkfit.config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
