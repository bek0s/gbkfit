"""
The options of the kinds of observations on the sky: the types of the
options of their grids, the configuration of grids and of the options
their data give, and checks of options.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias

from annotated_types import Ge, Gt, Len

from gbkfit.config import Unit, warn
from gbkfit.dataset import (
    Dataset, Regions, RegionsBins, UVCoverage, regions_parser)
from gbkfit.grid import Grid, SpectralAxis, grid_options
from gbkfit.instrument import Instrument

if TYPE_CHECKING:
    from .base import ObservationSky


# The options of the axes on the sky of the grid of an observation (see
# grid_options)
GRID_OPTIONS = ('size', 'step', 'rpix', 'rval', 'rota')

# A number of pixels, and a length on the sky (arcsec)
_Count: TypeAlias = Annotated[int, Ge(1)]
_Length: TypeAlias = Annotated[float, Gt(0)]

# The types of the options of the grid of an observation on the sky (see
# Coords; size and rpix are in pixels): one value for both its axes (x,
# y), or one for each (see per_axis); a length on the sky (e.g. the size
# of its pixels, its field of view) likewise
Size: TypeAlias = _Count | Annotated[Sequence[_Count], Len(2, 2)]
Lengths: TypeAlias = Annotated[
    _Length | Annotated[Sequence[_Length], Len(2, 2)], Unit('arcsec')]
Step: TypeAlias = Lengths
Rpix: TypeAlias = float | Annotated[Sequence[float], Len(2, 2)]
Rval: TypeAlias = Annotated[
    float | Annotated[Sequence[float], Len(2, 2)], Unit('deg')]
Rota: TypeAlias = Annotated[float, Unit('deg')]

# Those of a grid of one axis on the sky (along a slit): one value, or a
# list of one
SlitSize: TypeAlias = _Count | Annotated[Sequence[_Count], Len(1, 1)]
SlitStep: TypeAlias = Annotated[
    _Length | Annotated[Sequence[_Length], Len(1, 1)], Unit('arcsec')]
SlitRpix: TypeAlias = float | Annotated[Sequence[float], Len(1, 1)]
SlitRval: TypeAlias = Annotated[
    float | Annotated[Sequence[float], Len(1, 1)], Unit('deg')]

# A position on the sky: its RA and Dec (degrees)
SkyPosition: TypeAlias = Annotated[
    Annotated[Sequence[float], Len(2, 2)], Unit('deg')]

# How many times the model is oversampled along each axis of the data
# (see ObservationSky), of data of two axes (an image, moment maps,
# spectra along a slit, a dispersed image) or of three (a cube)
Oversampling2: TypeAlias = Annotated[Sequence[_Count], Len(2, 2)]
Oversampling3: TypeAlias = Annotated[Sequence[_Count], Len(3, 3)]

# The size of a median filter (see check_median_size)
MedianSize: TypeAlias = Annotated[int, Ge(3)]


def dump_grid(grid: Grid) -> dict[str, Any]:
    """
    Return the configuration of the grid of an observation: its options
    (see grid_options), with its spectral axis dumped, if any.
    """
    options = grid_options(grid)
    if 'spectral' in options:
        options['spectral'] = options['spectral'].dump()
    return options


def in_files(value: Any) -> bool:
    """
    Return whether the value of an option of the kind of an observation on
    the sky that data give is in a file in its configuration (regions in
    bins, and a UV coverage; see config_value).
    """
    return isinstance(value, (RegionsBins, UVCoverage))


def config_value(
        value: Any, prefix: str | None, dump_path: bool, overwrite: bool
) -> Any:
    """
    Return the value of an option of the kind of an observation on the
    sky that data give (see ObservationSky.data_options) as a
    configuration: a spectral axis, regions and a UV coverage dumped (the
    files of those in files named with the prefix, which they need; see
    in_files and Regions.dump), a tuple as a list, and others as they are
    (numbers).
    """
    if isinstance(value, SpectralAxis):
        return value.dump()
    if isinstance(value, Regions):
        return regions_parser.dump(
            value, prefix=prefix, dump_path=dump_path, overwrite=overwrite)
    if isinstance(value, UVCoverage):
        return value.dump(prefix, dump_path, overwrite)
    if isinstance(value, tuple):
        return list(value)
    return value


def without_options_from_data(
        observation: 'ObservationSky',
        info: dict[str, Any],
        dataset: Dataset | None
) -> dict[str, Any]:
    """
    Return the options of the kind of an observation (info, as dumped)
    without those its data give (see ObservationSky.options_from_data), if
    it is paired with data.
    """
    if dataset is None:
        return info
    options = observation.options_from_data(dataset)
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
