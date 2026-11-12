"""
The options of the observables: the types of the options of their grids,
the configuration of grids, the options their data give, and checks of
options.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias

from annotated_types import Ge

from gbkfit.config import ConfigError, Unit, make_typed_desc, warn
from gbkfit.dataset import Dataset
from gbkfit.grid import Grid, grid_options
from gbkfit.instrument import Instrument

if TYPE_CHECKING:
    from .base import Observable


# The options of the axes on the sky of the grid of an observable (see
# grid_options)
GRID_OPTIONS = ('size', 'step', 'rpix', 'rval', 'rota')

# The types of the options of the grid of an observable on the sky (see
# Coords; size and rpix are in pixels): one value for all its
# axes, or one for each (see per_axis); and of the size of a
# median filter (see check_median_size)
Size: TypeAlias = int | Sequence[int]
Step: TypeAlias = Annotated[float | Sequence[float], Unit('arcsec')]
Rpix: TypeAlias = float | Sequence[float]
Rval: TypeAlias = Annotated[float | Sequence[float], Unit('deg')]
Rota: TypeAlias = Annotated[float, Unit('deg')]
MedianSize: TypeAlias = Annotated[int, Ge(3)]


def dump_grid(grid: Grid) -> dict[str, Any]:
    """
    Return the configuration of the grid of an observable: its options
    (see grid_options), with its spectral axis dumped, if any.
    """
    options = grid_options(grid)
    if 'spectral' in options:
        options['spectral'] = options['spectral'].dump()
    return options


def check_no_options_from_data(
        cls: type['Observable'], info: dict[str, Any], dataset: Dataset
) -> None:
    """
    Raise ConfigError if the options of an observable of class cls (info)
    have one that its data give (see Observable.options_from_data).
    """
    given = [
        key for key in cls.options_from_data(dataset)
        if info.get(key) is not None]
    if given:
        desc = make_typed_desc(cls, 'observable')
        raise ConfigError(
            f"the data of {desc} give its options {given}; remove them (the "
            f"world coordinates of data can be set in the data options)")


def without_options_from_data(
        observable: 'Observable',
        info: dict[str, Any],
        dataset: Dataset | None
) -> dict[str, Any]:
    """
    Return the options of an observable (info, as dumped) without those
    its data give (see Observable.options_from_data), if it has data.
    """
    if dataset is None:
        return info
    options = observable.options_from_data(dataset)
    return {k: v for k, v in info.items() if k not in options}


def check_median_size(median_size: int | None) -> None:
    """
    Raise ValueError unless the size of a median filter (see
    ModelCube.median_size) is an odd number of pixels (a window centred on
    each pixel), or None.
    """
    if median_size is not None and median_size % 2 != 1:
        raise ValueError(
            f"median_size must be an odd number of pixels; it is "
            f"{median_size}")


def warn_unsmoothed_weights(
        smooth_weights: bool, instrument: Instrument
) -> None:
    """
    Warn if the weights are to be smoothed, but the instrument has no PSF
    or LSF to smooth them with.
    """
    if smooth_weights and instrument.psf() is None \
            and instrument.lsf() is None:
        warn(
            "smooth_weights is true, but the instrument has no PSF or LSF "
            "to smooth the weights with")
