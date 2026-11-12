"""
Helpers of the datasets on grids (see DatasetGrid): the world
coordinates of their files and their grids (and of the spectral axis of
the spectra of regions), their dumps, the correlations of their noise,
and their spectra masked whole.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from gbkfit.config import warn
from gbkfit.grid import (
    Coords, Grid, SpectralAxis, grid_options, make_cube_grid, make_grid,
    per_axis, rest_of_convention)
from gbkfit.instrument import LSF, Channel
from gbkfit.io import write_data
from .data import dump_data, item_filenames

if TYPE_CHECKING:
    from .base import Dataset, DatasetGrid, NoiseCorrelation
    from .data import Data


def grid_coords(
        coords: Coords,
        step: float | Sequence[float] | None,
        rota: float | None
) -> dict[str, Any]:
    """
    Return the world coordinates of the grid of a dataset from those of
    its files, with step and rota, if given, instead (rpix and rval are
    given to the reading of the files).
    """
    return dict(
        step=coords.step if step is None else step,
        rpix=coords.rpix,
        rval=coords.rval,
        rota=coords.rota if rota is None else rota)


def file_references(
        nspatial: int,
        rpix: float | Sequence[float] | None,
        rval: float | Sequence[float] | None,
        spectral: SpectralAxis | None
) -> dict[str, Any]:
    """
    Return the options of the reading of the files of a dataset of
    nspatial axes on the sky and a spectral axis after them (see
    Data.from_files): the reference pixels and values given, for each
    axis (None where not given), and the rest, the medium and the
    convention of the spectral axis (velocity: how an ambiguous header is
    read, see model_header), and its type (spectral_type: how an axis whose
    header gives a type FITS does not define is read, see model_header).

    Raises
    ------
    ValueError
        If the spectral axis has a size: its data give it; or rpix or
        rval has neither one value nor one for each axis on the sky.
    """
    spectral = spectral or SpectralAxis()
    if spectral.size() is not None:
        raise ValueError(
            "the data give the number of channels of their spectral axis; "
            "remove its size")

    def with_spectral(
            option: str,
            spatial: float | Sequence[float] | None,
            value: float | None
    ) -> list[float | None] | None:
        values = [None] * nspatial if spatial is None \
            else per_axis(option, spatial, nspatial)
        values = [*values, value]
        return None if all(v is None for v in values) else values
    return dict(
        rpix=with_spectral('rpix', rpix, spectral.rpix()),
        rval=with_spectral('rval', rval, spectral.rval()),
        rest=spectral.rest(), medium=spectral.medium(),
        velocity=spectral.convention(), spectral_type=spectral.type())


def cube_coords(
        coords: Coords,
        step: float | Sequence[float] | None,
        rota: float | None,
        spectral: SpectralAxis | None
) -> dict[str, Any]:
    """
    Return the world coordinates of the grid of a dataset of axes on the
    sky and a spectral axis after them (see cube_grid), from those of its
    files, with the steps, rota, and the convention and the frame of its
    velocities, if given, instead (see file_references for the others).
    A convention or a frame given for files that state another (e.g.
    optical for VRAD, radio velocities) relabels their velocities,
    without converting them: for a header that states the wrong one; a
    warning says so (see warn), as a convention copied by
    mistake puts the lines off (by 3.4% at 5000 km/s), and a frame the
    velocities (by up to some 30 km/s). The rest of the files is then
    that of the same line for the convention given (see
    rest_of_convention; e.g. the wavelength of a rest frequency).
    """
    spectral = spectral or SpectralAxis()
    given, stated = spectral.convention(), coords.convention
    if given is not None and stated is not None and given != stated:
        warn(
            f"the spectral axis of the data states velocities of the "
            f"convention {stated!r}; they are taken as given, of the "
            f"convention {given!r}, without converting them")
    frame = spectral.frame()
    if frame is not None and coords.frame is not None \
            and frame != coords.frame:
        warn(
            f"the spectral axis of the data states velocities in the frame "
            f"{coords.frame!r}; they are taken as given, in the frame "
            f"{frame!r}, without converting them")
    rest, medium = coords.rest, coords.medium
    if given is not None:
        rest, medium = rest_of_convention(rest, medium, given)
    nspatial = len(coords.step) - 1
    return dict(
        step=coords.step[:nspatial] if step is None else step,
        rpix=coords.rpix[:nspatial],
        rval=coords.rval[:nspatial],
        rota=coords.rota if rota is None else rota,
        spectral=SpectralAxis(
            step=spectral.step() or coords.step[nspatial],
            rpix=coords.rpix[nspatial], rval=coords.rval[nspatial],
            convention=given or stated, rest=rest, medium=medium,
            frame=frame or coords.frame))


def dataset_grid(
        dataset: 'Dataset',
        step: float | Sequence[float] | None,
        rpix: float | Sequence[float] | None,
        rval: float | Sequence[float] | None,
        rota: float | None
) -> Grid:
    """
    Return the grid of the items of a dataset, with the given world
    coordinates or their defaults (see make_grid).
    """
    return make_grid(dataset.shape()[::-1], step, rpix, rval, rota)


def cube_grid(
        dataset: 'Dataset',
        step: float | Sequence[float] | None,
        rpix: float | Sequence[float] | None,
        rval: float | Sequence[float] | None,
        rota: float | None,
        spectral: SpectralAxis | None
) -> Grid:
    """
    Return the grid of the items of a dataset of axes on the sky and a
    spectral axis after them, its last, with the given world coordinates
    or their defaults (see make_cube_grid). The data give the
    number of channels.

    Raises
    ------
    ValueError
        If the spectral axis has another number of channels, or the grid is
        not valid.
    """
    shape = dataset.shape()[::-1]
    spectral = spectral or SpectralAxis()
    if spectral.size() not in (None, shape[-1]):
        raise ValueError(
            f"the data have {shape[-1]} channels; the spectral axis has "
            f"{spectral.size()}")
    return make_cube_grid(
        shape[:-1], step, rpix, rval, rota,
        spectral.replaced(SpectralAxis(size=shape[-1])))


def dump_grid_dataset(
        dataset: 'DatasetGrid',
        prefix: str = '',
        dump_path: bool = True,
        overwrite: bool = False
) -> dict[str, Any]:
    """
    Return the configuration of a dataset whose data items are on a grid:
    the world coordinates of the grid (and of its spectral axis, if any,
    but its convention, which the headers of the files give), and the
    items, written to FITS files named after the prefix and their names
    (with their world coordinates).
    """
    grid = dataset.grid()
    info = dict(type=dataset.type(), **grid_options(grid, sizes=False))
    if 'spectral' in info:
        info['spectral'] = {
            key: value for key, value in info['spectral'].dump().items()
            if key != 'convention'}

    def write(filename: str, array: np.ndarray) -> None:
        write_data(
            filename, array, grid.coords, grid.spectral_axis,
            overwrite=overwrite)
    for key, item in dataset.items():
        info[key] = dump_data(
            item, item_filenames(prefix, key), write, dump_path,
            f'{prefix}{key}_', overwrite)
    return info


def flatten_single_item(
        info: dict[str, Any], name: str
) -> dict[str, Any]:
    """
    Return the configuration of a dataset of one data item with the
    options of the item beside its own, as it is given.
    """
    info = dict(info)
    return info | info.pop(name)


def grid_noise_correlation(
        dataset: 'DatasetGrid', key: str
) -> 'NoiseCorrelation | None':
    """
    Return the correlation of the noise of a data item of a dataset on a
    grid of the sky (see Dataset.noise_correlation): the shapes of its
    correlations on the pixels and the channels of the grid, out to the
    largest separations of the data, divided by their values at their
    centres (the data have a spectral axis if they have a spectral
    correlation, see Dataset). Raise ValueError if a shape is not largest
    at its centre.
    """
    from .base import NoiseCorrelation
    item = dataset[key]
    spatial, spectral = item.spatial_correlation(), item.spectral_correlation()
    if spatial is None and spectral is None:
        return None
    grid = dataset.grid()
    if spatial is not None:
        nx, ny = grid.size[:2]
        spatial = _centred(spatial.asarray(
            grid.coords.step[:2], (2 * nx - 1, 2 * ny - 1),
            rota=grid.coords.rota), key)
    if spectral is not None:
        spectral = _spectral_shape(key, spectral, grid)
    return NoiseCorrelation(spatial, spectral)


def spectra_noise_correlation(
        dataset: 'Dataset', key: str, grid: Grid
) -> 'NoiseCorrelation | None':
    """
    Return the correlation of the noise of a data item of spectra at
    positions that are not pixels of the sky (along a slit, or regions;
    see Dataset.noise_correlation), on the channels of the spectral axis
    of a grid: along it alone. Raise ValueError if the item has a spatial
    correlation, which the positions cannot have (a slit has no pixels
    across it, and regions are of any shape), or the shape is not
    largest at its centre.
    """
    from .base import NoiseCorrelation
    item = dataset[key]
    if item.spatial_correlation() is not None:
        raise ValueError(
            f"the noise of {dataset.type()} data can be correlated along "
            f"their spectral axis alone (spectral_correlation), not on the "
            f"sky")
    if (spectral := item.spectral_correlation()) is None:
        return None
    return NoiseCorrelation(None, _spectral_shape(key, spectral, grid))


def _spectral_shape(key: str, spectral: LSF, grid: Grid) -> np.ndarray:
    """
    Return the shape of the spectral correlation of the noise of data
    item key on the channels of the spectral axis of a grid (see
    grid_noise_correlation).
    """
    # (drawn at the middle channel, of the widths of the light there)
    channel = Channel.middle(grid)
    try:
        shape = spectral.asarray(
            channel.width, channel, 2 * grid.size[grid.spectral_axis] - 1)
    except RuntimeError as e:
        raise RuntimeError(
            f"the spectral correlation of the noise of '{key}' is drawn "
            f"on the channels of the data, as they are: {e}") from None
    return _centred(shape, key)


def whole_spectra(item: 'Data') -> np.ndarray:
    """
    Return the values of a data item of spectra (its channels the first
    axis) with each spectrum masked whole (NaN) where it masks any of its
    channels, but for the channels it masks in every spectrum (see
    Dataset.mask_whole_spectra).
    """
    kept = item.mask() != 0
    channels = np.any(kept.reshape(kept.shape[0], -1), axis=1)
    whole = np.all(kept[channels], axis=0)
    return np.where(whole, item.data(), np.nan)


def _centred(shape: np.ndarray, key: str) -> np.ndarray:
    """
    Return the shape of a correlation of the noise of data item key (of
    odd sizes) divided by its value at its centre, or raise ValueError
    unless that is its largest (the variance of the noise).
    """
    centre = shape[tuple(n // 2 for n in shape.shape)]
    if not centre > 0 or np.any(shape > centre):
        raise ValueError(
            f"the correlation of the noise of '{key}' must be largest at "
            f"its centre (at a separation of 0)")
    return shape / centre
