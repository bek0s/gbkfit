from typing import TYPE_CHECKING, Any

from gbkfit._version import __version__

if TYPE_CHECKING:
    from gbkfit._catalogue import catalogue
    from gbkfit._schema import schema
    from gbkfit.setup import Description, Setup, describe, load
    from gbkfit.utils.parseutils import ConfigError, Problem


__all__ = [
    'ConfigError',
    'Description',
    'Problem',
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
    if name in ('Description', 'Setup', 'describe', 'load'):
        import gbkfit.setup
        return getattr(gbkfit.setup, name)
    if name in ('ConfigError', 'Problem'):
        import gbkfit.utils.parseutils
        return getattr(gbkfit.utils.parseutils, name)
    if name == 'catalogue':
        import gbkfit._catalogue
        return gbkfit._catalogue.catalogue
    if name == 'schema':
        import gbkfit._schema
        return gbkfit._schema.schema
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
