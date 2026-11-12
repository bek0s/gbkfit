"""
GBKFIT: fitting 3D kinematic models of galaxies to their data.

The front door is this package: gbkfit.load sets up a configuration (a
YAML or JSON file, or a dict) as a Setup, which evaluates, fits, mocks
and bootstraps it; gbkfit.describe checks a configuration without
devices, and gbkfit.catalogue and gbkfit.schema say what configurations
can hold. They are those of gbkfit.setup, imported on first use.

The packages, in layers from the bottom up: each imports only from its
own layer and those below. The files of a study stand apart: they need
only the foundations and the shared basics, and fitting and the entry
points use them.

Foundations
    utils        helpers of plain Python (functions, paths, timers)
    units        physical units, read by the rules of the FITS standard
    math         numerical algorithms (interpolation, overlaps, noise)
    control      stopping work as it runs (StopControl)
    config       configurations: their parsing, errors, schema, version
Shared basics
    grid         grids of pixels with world coordinates; spectral axes
    io           files of data (FITS, ASDF, measurement sets, tables)
    driver       the devices that evaluate models: the CPU or a GPU
    params       parameters: fixed, tied, free or solved; their priors
    relations    scaling relations that parameter expressions can call
    cosmology    the cosmology of a setup (distances from redshifts)
The data side (never imports the model)
    instrument   what the telescope does to the light: beam, PSF, LSF
    dataset      the data and where they were measured: pixels, regions
                 (bins and apertures), slits, visibilities, radii
The model
    model        galaxies: their components, mass, gravity, distance
Observation and fitting
    observation  how the model gives the data of a dataset; mock data
    objective    the comparison of models with data (likelihoods)
    parallel     copies of an objective evaluated at the same time
    fitting      the fitters, the result of a fit, its bootstrap
The files of a study
    project      the folder of a study: the pieces of its setups
    outputs      the files that each run writes, and their manifest
    migration    old configurations rewritten in the current format
Entry points and tools
    setup        the front door: a Setup, in Python or from a config
    tasks        the tasks programs run: eval, fit, plot, prep, migrate
    analysis     measurements of maps that need no model (e.g. kinemetry)
    prep         data made ready for the datasets; PSFs and LSFs

The applications (the command line gbkfit-cli, the server and Studio)
are packages of their own: they use the library, which never imports
them.
"""

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
