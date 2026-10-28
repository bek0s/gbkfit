from typing import Any

from gbkfit._version import __version__


def __getattr__(name: str) -> Any:
    # The front door (see gbkfit.setup), imported on first use: it imports
    # the whole library, which a plain import of gbkfit (e.g. by its
    # command line) does not need
    if name in ('Description', 'Setup', 'describe', 'load'):
        import gbkfit.setup
        return getattr(gbkfit.setup, name)
    if name == 'catalogue':
        import gbkfit._catalogue
        return gbkfit._catalogue.catalogue
    if name == 'schema':
        import gbkfit._schema
        return gbkfit._schema.schema
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
