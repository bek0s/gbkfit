"""
The prep task: prepare data files for the datasets of gbkfit (see
gbkfit.prep), and write the tables of the PSFs and LSFs of instruments.
The prepared files of data are named by their role, after the input of
the data: prep_<name>.fits, prep_<name>_error.fits and
prep_<name>_mask.fits (see output_files), in the output directory
(output_dir; if it exists: an error with output_dir_mode 'terminate',
written into with 'overwrite', the default, or a new one with a numeric
suffix with 'unique', e.g. output_2). The input of a FITS file and an
HDU is named with its HDU (e.g. prep_cube_SCI.fits). Two inputs of one
name (e.g. in two directories) are an error.
"""

import dataclasses
import logging
import os
from collections.abc import Sequence
from typing import Literal

import astropy.table
import astropy.units
import astropy.wcs
import ruamel.yaml
import numpy as np
import numpy.typing as npt
from astropy.io import fits

import gbkfit.units
from gbkfit.config import DataFile, split_data_file
from gbkfit.io import (
    ErrorKind, MaskKind, casa_beam_table, read_column, read_header,
    read_table, spectra_header, write_image)
import gbkfit.outputs
from gbkfit import prep
from gbkfit.prep import FitsArrays


__all__ = [
    'output_files',
    'prep_gauss_hermite_moments',
    'prep_lsf_kmos3d',
    'prep_lsf_muse',
    'prep_lsf_resolving_power',
    'prep_pixel_brightness',
    'prep_pixel_moments',
    'prep_pixel_spectra',
    'prep_psf_casa_beams',
    'prep_psf_kmos3d',
    'prep_psf_muse',
    'prep_region_moments',
    'prep_region_spectra',
    'prep_slit_spectra'
]


_log = logging.getLogger(__name__)

# A range of pixels, as given: its first and last + 1
_Range = Sequence[int]

# What to do if the output directory exists (see _output_dir)
_Exists = Literal['terminate', 'overwrite', 'unique']


def _output_dir(output_dir: str, mode: _Exists) -> str:
    """
    Create the output directory, and return its absolute path (see
    gbkfit.outputs.make_directory).
    """
    output_dir = gbkfit.outputs.make_directory(output_dir, mode)
    _log.info(f"output will be stored under directory: {output_dir}")
    return output_dir


def _output(output_dir: str, file: DataFile, role: str = '') -> str:
    """
    Return the name of a prepared file of an input file (with its HDU, or
    the path of its array in an ASDF file, if given): prep_<name>.fits, or
    prep_<name>_<role>.fits for a role.
    """
    filename, key = (file, None) if isinstance(file, str) else file
    name = os.path.splitext(os.path.basename(filename))[0]
    if key is not None:
        name += '_' + str(key).replace('/', '_')
    if role:
        name += f'_{role}'
    return os.path.join(output_dir, f'prep_{name}.fits')


def _check_distinct(outputs: Sequence[str]) -> None:
    """Raise ValueError if two inputs have one prepared file."""
    seen = set()
    for output in outputs:
        if output in seen:
            raise ValueError(
                f"two inputs would be prepared into {output}: inputs of one "
                f"name (e.g. in two directories) must be renamed")
        seen.add(output)


def output_files(
        output_dir: str,
        file_d: DataFile,
        file_e: DataFile | None,
        file_m: DataFile | None,
        noise_error: bool = False
) -> tuple[str, str | None, str | None]:
    """
    Return the prepared files of the data, errors and mask (None for
    those not given) that prep_pixel_brightness, prep_slit_spectra,
    prep_pixel_spectra and prep_pixel_moments (of each map) write in an
    output directory, named after the data (see the module); with
    noise_error (prep_pixel_spectra), the errors estimated from the noise
    of the data.

    Parameters
    ----------
    output_dir : str
        The output directory.
    file_d, file_e, file_m : DataFile
        The files of the data, errors and mask.
    noise_error : bool, optional
        Whether the errors are estimated from the noise of the data.

    Returns
    -------
    tuple of str and (str or None)
        The prepared files of the data, errors and mask.
    """
    error = _output(output_dir, file_d, 'error') \
        if file_e is not None or noise_error else None
    mask = _output(output_dir, file_d, 'mask') if file_m is not None else None
    return _output(output_dir, file_d), error, mask


def _clip(
        arrays: FitsArrays,
        clip_min: float | None,
        clip_max: float | None,
        sclip_sigma: float | None,
        sclip_iters: int | None,
        ccl_lcount: int | None,
        ccl_pcount: int | None,
        ccl_lratio: float | None
) -> FitsArrays:
    """
    Return the arrays without the values out of a range, those sigma
    clipping rejects (among the others), and those out of the largest
    connected regions (see gbkfit.prep).
    """
    arrays = arrays.masked(prep.range_mask(arrays.data, clip_min, clip_max))
    if sclip_sigma is not None:
        arrays = arrays.masked(prep.sigma_clip_mask(
            arrays.data, arrays.mask, sclip_sigma, sclip_iters))
    if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
        arrays = arrays.masked(prep.largest_regions_mask(
            arrays.mask, ccl_lcount, ccl_pcount, ccl_lratio))
    return arrays


def _write(
        arrays: FitsArrays,
        minify: bool,
        nanpad: int | None,
        outputs: tuple[str, str | None, str | None],
        dtype: npt.DTypeLike
) -> None:
    """
    Crop the arrays to their valid pixels (minify), pad them (nanpad),
    and write them in the prepared files of the data, errors and mask
    (see output_files).
    """
    if minify:
        arrays = arrays.minified()
    if nanpad:
        arrays = arrays.padded(nanpad)
    arrays.write(*outputs, dtype=dtype)


def prep_pixel_brightness(
        file_d: DataFile,
        file_e: DataFile | None,
        file_m: DataFile | None,
        roi_spat: _Range | None,
        clip_min: float | None,
        clip_max: float | None,
        ccl_lcount: int | None,
        ccl_pcount: int | None,
        ccl_lratio: float | None,
        sclip_sigma: float | None,
        sclip_iters: int | None,
        minify: bool,
        nanpad: int | None,
        dtype: npt.DTypeLike,
        output_dir: str = '.',
        output_dir_mode: _Exists = 'overwrite',
        error_kind: ErrorKind = 'sigma',
        mask_kind: MaskKind = 'valid'
) -> None:
    """
    Prepare an image (pixel_brightness): crop it to a region (roi_spat:
    x from, x to, y from, y to), clip it, crop it to its valid pixels
    (minify), and pad it with invalid pixels (nanpad). error_kind is what
    the file of the errors holds, and mask_kind what the mask marks (see
    read_arrays); the prepared errors are standard deviations.
    """
    output_dir = _output_dir(output_dir, output_dir_mode)
    arrays = FitsArrays.read(
        file_d, file_e, file_m, error_kind=error_kind, mask_kind=mask_kind)
    if roi_spat is not None:
        x0, x1, y0, y1 = roi_spat
        arrays = arrays.crop((slice(y0, y1), slice(x0, x1)))
    arrays = _clip(
        arrays, clip_min, clip_max, sclip_sigma, sclip_iters, ccl_lcount,
        ccl_pcount, ccl_lratio)
    _write(arrays, minify, nanpad,
           output_files(output_dir, file_d, file_e, file_m), dtype)


def prep_slit_spectra(
        file_d: DataFile,
        file_e: DataFile | None,
        file_m: DataFile | None,
        roi_spat: _Range | None,
        roi_spec: _Range | None,
        clip_min: float | None,
        clip_max: float | None,
        ccl_lcount: int | None,
        ccl_pcount: int | None,
        ccl_lratio: float | None,
        sclip_sigma: float | None,
        sclip_iters: int | None,
        minify: bool,
        nanpad: int | None,
        dtype: npt.DTypeLike,
        velocity_rest: str | None = None,
        velocity: str | None = None,
        spectral_type: str | None = None,
        output_dir: str = '.',
        output_dir_mode: _Exists = 'overwrite',
        error_kind: ErrorKind = 'sigma',
        mask_kind: MaskKind = 'valid'
) -> None:
    """
    Prepare the spectra along a slit (slit_spectra: positions along x,
    channels along y), with their spectral axis in velocities of a rest
    (velocity_rest), if given: crop them to positions (roi_spat) and
    channels (roi_spec, of the axis in velocities, increasing), clip them,
    crop them to their valid pixels (minify), and pad them (nanpad).
    velocity is the convention of the velocities of an axis that does not
    state it, spectral_type the type of an axis whose header gives one
    that FITS does not define, error_kind what the file of the errors
    holds, and mask_kind what the mask marks (see FitsArrays.read); the
    prepared errors are standard deviations.
    """
    output_dir = _output_dir(output_dir, output_dir_mode)
    arrays = FitsArrays.read(
        file_d, file_e, file_m, velocity, rest=velocity_rest,
        spectral_type=spectral_type,
        error_kind=error_kind, mask_kind=mask_kind)
    rows, columns = slice(None), slice(None)
    if roi_spec is not None:
        rows = slice(*roi_spec)
    if roi_spat is not None:
        columns = slice(*roi_spat)
    arrays = _clip(
        arrays.crop((rows, columns)), clip_min, clip_max, sclip_sigma,
        sclip_iters, ccl_lcount, ccl_pcount, ccl_lratio)
    _write(arrays, minify, nanpad,
           output_files(output_dir, file_d, file_e, file_m), dtype)


def prep_pixel_spectra(
        file_d: DataFile,
        file_e: DataFile | None,
        file_m: DataFile | None,
        roi_spat: _Range | None,
        roi_spec: _Range | None,
        clip_min: float | None,
        clip_max: float | None,
        ccl_lcount: int | None,
        ccl_pcount: int | None,
        ccl_lratio: float | None,
        sclip_sigma: float | None,
        sclip_iters: int | None,
        minify: bool,
        nanpad: int | None,
        dtype: npt.DTypeLike,
        velocity_rest: str | None = None,
        velocity: str | None = None,
        spectral_type: str | None = None,
        output_dir: str = '.',
        output_dir_mode: _Exists = 'overwrite',
        noise_error: bool = False,
        error_kind: ErrorKind = 'sigma',
        mask_kind: MaskKind = 'valid'
) -> None:
    """
    Prepare a cube (pixel_spectra), with its spectral axis in velocities
    of a rest (velocity_rest), if given: crop it to a region (roi_spat: x
    from, x to, y from, y to) and channels (roi_spec, of the axis in
    velocities, increasing), clip it, crop it to its valid pixels
    (minify), and pad it (nanpad). velocity is the convention of the
    velocities of an axis that does not state it, spectral_type the type
    of an axis whose header gives one that FITS does not define,
    error_kind what the file of the errors holds, and mask_kind what the
    mask marks (see FitsArrays.read); the prepared errors are standard
    deviations. With noise_error, the errors are the noise of each channel
    of the cropped cube, from its negative values (see prep.noise_errors;
    see output_files for the names of the prepared files).

    Raises
    ------
    ValueError
        If both noise_error and an errors file are given.
    """
    if noise_error and file_e is not None:
        raise ValueError(
            "give a file of the errors or estimate them from the noise "
            "(noise_error), not both")
    output_dir = _output_dir(output_dir, output_dir_mode)
    arrays = FitsArrays.read(
        file_d, file_e, file_m, velocity, rest=velocity_rest,
        spectral_type=spectral_type,
        error_kind=error_kind, mask_kind=mask_kind)
    channels, ys, xs = slice(None), slice(None), slice(None)
    if roi_spat is not None:
        xs, ys = slice(*roi_spat[0:2]), slice(*roi_spat[2:4])
    if roi_spec is not None:
        channels = slice(*roi_spec)
    arrays = arrays.crop((channels, ys, xs))
    if noise_error:
        # (of the noise before the clipping, which masks some of it)
        arrays = dataclasses.replace(
            arrays, error=prep.noise_errors(arrays.data, arrays.mask))
    arrays = _clip(
        arrays, clip_min, clip_max, sclip_sigma, sclip_iters, ccl_lcount,
        ccl_pcount, ccl_lratio)
    _write(arrays, minify, nanpad,
           output_files(output_dir, file_d, file_e, file_m, noise_error),
           dtype)


def prep_pixel_moments(
        file_d: Sequence[DataFile],
        file_e: Sequence[DataFile | None] | None,
        file_m: Sequence[DataFile | None] | None,
        roi_spat: _Range | None,
        clip_min: Sequence[float | None] | None,
        clip_max: Sequence[float | None] | None,
        ccl_lcount: int | None,
        ccl_pcount: int | None,
        ccl_lratio: float | None,
        sclip_sigma: Sequence[float | None] | None,
        sclip_iters: int | Sequence[int] | None,
        minify: bool,
        nanpad: int | None,
        dtype: npt.DTypeLike,
        output_dir: str = '.',
        output_dir_mode: _Exists = 'overwrite',
        error_kind: ErrorKind = 'sigma',
        mask_kind: MaskKind = 'valid'
) -> None:
    """
    Prepare moment maps (pixel_moments), as image (see
    prep_pixel_brightness), but with one mask for all: a pixel invalid in
    one map is invalid in all. The options of the clipping (clip_min,
    clip_max, sclip_sigma and sclip_iters) have one value for each map
    (None: none for that map); sclip_iters can also be one value for all
    maps. error_kind is what the files of the errors hold, and mask_kind
    what the masks mark (see read_arrays).
    """
    output_dir = _output_dir(output_dir, output_dir_mode)
    nmaps = len(file_d)
    file_e = file_e or [None] * nmaps
    file_m = file_m or [None] * nmaps
    if np.ndim(sclip_iters) == 0:
        sclip_iters = [sclip_iters] * nmaps
    outputs = [
        output_files(output_dir, file_d[i], file_e[i], file_m[i])
        for i in range(nmaps)]
    _check_distinct([output[0] for output in outputs])
    maps = []
    for i in range(nmaps):
        arrays = FitsArrays.read(
            file_d[i], file_e[i], file_m[i], error_kind=error_kind,
            mask_kind=mask_kind)
        if roi_spat is not None:
            x0, x1, y0, y1 = roi_spat
            arrays = arrays.crop((slice(y0, y1), slice(x0, x1)))
        maps.append(arrays)
    # One mask for all the maps (the sigma clipping of each sees the
    # pixels the maps before it left)
    mask = np.ones(maps[0].data.shape, bool)
    for i, arrays in enumerate(maps):
        mask &= (arrays.mask != 0) & prep.range_mask(
            arrays.data, clip_min[i] if clip_min is not None else None,
            clip_max[i] if clip_max is not None else None)
        if sclip_sigma is not None and sclip_sigma[i] is not None:
            mask &= prep.sigma_clip_mask(
                arrays.data, mask, sclip_sigma[i], sclip_iters[i])
    if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
        mask &= prep.largest_regions_mask(
            mask, ccl_lcount, ccl_pcount, ccl_lratio)
    box = prep.bounding_box(mask) if minify else None
    for i, arrays in enumerate(maps):
        arrays = arrays.masked(mask)
        if box is not None:
            arrays = arrays.crop(box)
        _write(arrays, False, nanpad, outputs[i], dtype)


def _read_bins(
        file: DataFile, plane: int | None, data: FitsArrays
) -> tuple[fits.Header, np.ndarray, int]:
    """
    Return the header of a map of bins (of a plane of a file of them, if
    given), read on the pixels of the sky of data as their errors and mask
    are (see FitsArrays.read_sky_map), the bin of each pixel numbered from
    0 (-1 for none; see gbkfit.prep.bin_index), and the number of bins.
    """
    bins, header = data.read_sky_map(file, plane)
    try:
        index, nbins = prep.bin_index(bins)
    except ValueError as e:
        raise ValueError(f"{split_data_file(file)[0]}: {e}") from e
    return header, index, nbins


def _write_bins(
        header: fits.Header,
        index: np.ndarray,
        filename: str,
        output_dir: str
) -> None:
    """Write the bins numbered from 0 (prep_<bins>.fits)."""
    FitsArrays(index, None, np.ones(index.shape), header).write(
        _output(output_dir, filename), dtype=np.int32)


def prep_region_moments(
        file_bins: DataFile,
        file_d: Sequence[DataFile],
        file_e: Sequence[DataFile | None] | None,
        file_m: Sequence[DataFile | None] | None,
        dtype: npt.DTypeLike,
        output_dir: str = '.',
        output_dir_mode: _Exists = 'overwrite',
        error_kind: ErrorKind = 'sigma',
        mask_kind: MaskKind = 'valid',
        bins_plane: int | None = None
) -> None:
    """
    Prepare binned moment maps (e.g. of MaNGA DAP or GIST) for
    region_moments: a map of the bin of each pixel (file_bins; negative or
    NaN for no bin, on the pixels of the maps as their errors and masks
    are, see FitsArrays.read_sky_map; of the plane bins_plane of a file of
    them, if given: e.g. 1, of the stellar continuum, of the BINID of the
    MAPS of MaNGA's DAP), and moment maps (file_d, with optional errors
    file_e, of the kind error_kind, and masks file_m, of the kind
    mask_kind, see read_arrays) whose pixels hold the value of their bin.
    Write the bins, numbered from 0 in the order of their numbers in the
    map
    (prep_<bins>.fits), and for each map a vector of the value of each bin
    (prep_<map>.fits), whose masked or empty bins are NaN, and of their
    errors, standard deviations (prep_<map>_error.fits), in the unit of
    the map (BUNIT), if it has one.
    """
    output_dir = _output_dir(output_dir, output_dir_mode)
    nmaps = len(file_d)
    file_e = file_e or [None] * nmaps
    file_m = file_m or [None] * nmaps
    _check_distinct(
        [_output(output_dir, file_bins)]
        + [_output(output_dir, file) for file in file_d])
    maps = [
        FitsArrays.read(
            file_d[i], file_e[i], file_m[i], error_kind=error_kind,
            mask_kind=mask_kind)
        for i in range(nmaps)]
    bins_header, index, nbins = _read_bins(file_bins, bins_plane, maps[0])
    _log.info(f"{nbins} bins")
    for file, arrays in zip(file_d[1:], maps[1:]):
        if not np.array_equal(
                _read_bins(file_bins, bins_plane, arrays)[1], index):
            raise ValueError(
                f"{file}: the bins on its pixels are not those on the "
                f"pixels of {file_d[0]}")
    _write_bins(bins_header, index, file_bins, output_dir)
    for i, arrays in enumerate(maps):
        # (the errors are in the unit of the map, see read_arrays)
        for filename, data, role in (
                (file_d[i], arrays.data, ''),
                (file_e[i], arrays.error, 'error')):
            if filename is None:
                continue
            try:
                values = prep.bin_values(index, nbins, data)
            except ValueError as e:
                raise ValueError(f"{filename}: {e}") from e
            write_image(
                _output(output_dir, file_d[i], role), values.astype(dtype),
                unit=arrays.unit, overwrite=True)


def prep_region_spectra(
        file_bins: DataFile,
        file_d: DataFile,
        file_e: DataFile | None,
        file_m: DataFile | None,
        roi_spec: _Range | None,
        dtype: npt.DTypeLike,
        velocity_rest: str | None = None,
        velocity: str | None = None,
        spectral_type: str | None = None,
        output_dir: str = '.',
        output_dir_mode: _Exists = 'overwrite',
        error_kind: ErrorKind = 'sigma',
        mask_kind: MaskKind = 'valid',
        bins_plane: int | None = None
) -> None:
    """
    Prepare the spectra of the bins of a cube for region_spectra: a map of
    the bin of each pixel (file_bins; negative or NaN for no bin, on the
    pixels of the cube as its errors and mask are, see
    FitsArrays.read_sky_map; of the plane bins_plane of a file of them, if
    given), and a cube (file_d, with optional errors file_e
    and mask file_m), with its spectral axis in velocities of a rest
    (velocity_rest), if given, and optionally cropped to channels
    (roi_spec, of the axis in velocities, increasing). velocity is the
    convention of the velocities of an axis that does not state it,
    spectral_type the type of an axis whose header gives one that FITS
    does not define, error_kind what the file of the errors holds, and
    mask_kind what the mask marks (see FitsArrays.read).
    Write the bins, numbered from 0 in the order of their numbers in the
    map (prep_<bins>.fits), the sum of the valid spectra of the pixels of
    each bin (prep_<cube>.fits: the bins along x, the channels along y),
    with its errors added in quadrature (prep_<cube>_error.fits), and
    which pixels are valid in each channel (prep_<cube>_valid.fits: 1 or
    0, on the pixels of the cube), with which the model sums the same
    pixels (the option valid of the bins, see region.RegionsBins). With
    a unit (BUNIT), the sums are fluxes, in their unit: the values of
    pixels per solid angle or per beam become the fluxes of the pixels
    (see prep.pixel_fluxes).
    """
    output_dir = _output_dir(output_dir, output_dir_mode)
    _check_distinct(
        [_output(output_dir, file_bins), _output(output_dir, file_d)])
    cube = FitsArrays.read(
        file_d, file_e, file_m, velocity, rest=velocity_rest,
        spectral_type=spectral_type,
        error_kind=error_kind, mask_kind=mask_kind)
    bins_header, index, nbins = _read_bins(file_bins, bins_plane, cube)
    _log.info(f"{nbins} bins")
    fluxes = prep.pixel_fluxes(cube.header, cube.unit, cube.beam, file_d)
    if roi_spec is not None:
        cube = cube.crop((slice(*roi_spec), slice(None), slice(None)))
        if fluxes is not None and np.ndim(fluxes[0]):
            fluxes = (fluxes[0][slice(*roi_spec)], fluxes[1])
    spectra, errors = prep.bin_spectra(index, nbins, cube)
    header = spectra_header(split_data_file(file_d)[0], cube.header, nbins)
    # (the sums of the fluxes of the pixels: the values of the pixels per
    # solid angle or per beam become theirs, see prep.pixel_fluxes)
    unit = None
    if fluxes is not None:
        factor = np.reshape(fluxes[0], (-1, 1))
        spectra = spectra * factor
        errors = None if errors is None else errors * factor
        unit = fluxes[1]
    _write_bins(bins_header, index, file_bins, output_dir)
    FitsArrays(
        spectra, errors, np.isfinite(spectra).astype(float), header,
        unit).write(
            _output(output_dir, file_d),
            None if file_e is None else _output(output_dir, file_d, 'error'),
            dtype=dtype)
    valid = cube.header.copy()
    valid.remove('EXTNAME', ignore_missing=True)
    write_image(
        _output(output_dir, file_d, 'valid'),
        (cube.mask != 0).astype(np.uint8), valid, overwrite=True)


def _write_table(
        table: astropy.table.QTable, output_dir: str, name: str
) -> str:
    """Write a table (ECSV) as prep_<name>.ecsv; return its path."""
    filename = os.path.join(output_dir, f'prep_{name}.ecsv')
    table.write(filename, overwrite=True)
    return filename


def prep_gauss_hermite_moments(
        file: str,
        columns: Sequence[str],
        errors: Sequence[str] | None = None,
        format_: str | None = None,
        output_dir: str = '.',
        output_dir_mode: _Exists = 'overwrite'
) -> None:
    """
    Write the mean and the dispersion of the Gauss-Hermite series of a
    table of stellar kinematics (e.g. the V, sigma, h3 and h4 of the bins
    of pPXF, one row per bin), for models that predict those and not the
    shape of the distribution of velocities (see
    gbkfit.prep.gauss_hermite_moments): a table (prep_<name>.ecsv) of the
    columns mean and dispersion, in the unit of the velocities, with
    mean_error and dispersion_error if the errors are given.

    Parameters
    ----------
    file : str
        The table (any that astropy reads, see Table.read).
    columns : Sequence of str
        The columns of V, sigma, h3 and h4.
    errors : Sequence of str, optional
        The columns of their errors.
    format_ : str, optional
        The format of the table, if astropy cannot tell it.
    output_dir, output_dir_mode : optional
        The output directory (see the module).
    """
    if len(columns) != 4 or (errors is not None and len(errors) != 4):
        raise ValueError(
            "give the columns of V, sigma, h3 and h4, and of their errors "
            "if any, four each")
    table = read_table(file, format_)
    velocity = read_column(table, columns[0], None)
    unit = velocity.unit

    def values(name: str, of_velocity: bool) -> np.ndarray:
        value = read_column(table, name, None)
        if of_velocity and value.unit != astropy.units.dimensionless_unscaled:
            value = value.to(unit)
        return value.value

    kinds = (True, True, False, False)
    moments = prep.gauss_hermite_moments(
        *(values(name, kind) for name, kind in zip(columns, kinds)),
        *(values(name, kind) for name, kind in zip(errors or (), kinds)))
    output = astropy.table.QTable(dict(
        mean=moments.mean * unit, dispersion=moments.dispersion * unit))
    if moments.mean_error is not None:
        output['mean_error'] = moments.mean_error * unit
        output['dispersion_error'] = moments.dispersion_error * unit
    output_dir = _output_dir(output_dir, output_dir_mode)
    name = os.path.splitext(os.path.basename(file))[0]
    filename = _write_table(output, output_dir, name)
    moment1 = 'mean, error: mean_error' if errors else 'mean'
    moment2 = 'dispersion, error: dispersion_error' if errors else 'dispersion'
    _log.info(
        f"the mean and the dispersion are in {filename}; use them as the "
        f"data of region_moments (file: {filename}, moment1: {{column: "
        f"{moment1}}}, moment2: {{column: {moment2}}}), of the observable "
        f"method moments")


def prep_psf_muse(
        file: str,
        output_dir: str = '.',
        output_dir_mode: _Exists = 'overwrite'
) -> None:
    """
    Write the PSF of a MUSE cube (from its header) as a table of the
    options fwhm and beta of a Moffat PSF against wavelength (see
    gbkfit.prep.muse_psf_table).
    """
    output_dir = _output_dir(output_dir, output_dir_mode)
    # (MUSE writes its PSF in the primary header of its cubes, of no
    # array: a header of keywords, not that of data)
    filename = _write_table(
        prep.muse_psf_table(fits.getheader(file, 0)), output_dir, 'psf')
    _log.info(
        f"the PSF is in {filename}; use it as psf: {{type: moffat, "
        f"fwhm: {{file: {filename}, points: wavelength, column: fwhm}}, "
        f"beta: {{file: {filename}, points: wavelength, column: beta}}}}")


def prep_lsf_muse(
        output_dir: str = '.',
        output_dir_mode: _Exists = 'overwrite'
) -> None:
    """
    Write the LSF of MUSE as a table of its FWHM against wavelength (see
    gbkfit.prep.muse_lsf_table).
    """
    output_dir = _output_dir(output_dir, output_dir_mode)
    filename = _write_table(prep.muse_lsf_table(), output_dir, 'lsf')
    _log.info(
        f"the LSF is in {filename}; use it as lsf: {{type: gauss, "
        f"fwhm: {{file: {filename}, points: wavelength, column: fwhm}}}}")


def prep_lsf_resolving_power(
        file: DataFile,
        points: str,
        column: str,
        points_unit: str | None = None,
        format_: str | None = None,
        output_dir: str = '.',
        output_dir_mode: _Exists = 'overwrite'
) -> None:
    """
    Write the LSF of an instrument of the resolving powers of a table
    file (e.g. of its curve of R against wavelength, as of JWST NIRSpec),
    a Gaussian, as a table of its FWHM against the points (see
    gbkfit.prep.resolving_power_table).

    The points are those of a column of wavelengths, frequencies or
    velocities; their unit is that of the column, or points_unit if it
    has none (e.g. of a CSV file).
    """
    table = read_table(file, format_)
    point_values = read_column(table, points, points_unit)
    if point_values.unit == astropy.units.dimensionless_unscaled:
        raise ValueError(
            f"the column of the points, {points!r}, has no unit; give it "
            f"(points_unit)")
    power = read_column(table, column, None)
    if power.unit != astropy.units.dimensionless_unscaled:
        raise ValueError(
            f"the column of the resolving powers, {column!r}, has a unit "
            f"({gbkfit.units.format_unit(power.unit)}): a resolving power "
            f"is a number")
    output_dir = _output_dir(output_dir, output_dir_mode)
    filename = _write_table(prep.resolving_power_table(
        point_values, power.value, points), output_dir, 'lsf')
    _log.info(
        f"the LSF is in {filename}; use it as lsf: {{type: gauss, fwhm: "
        f"{{file: {filename}, points: {points}, column: fwhm}}}}")


def prep_lsf_kmos3d(
        file: str,
        output_dir: str = '.',
        output_dir_mode: _Exists = 'overwrite'
) -> None:
    """
    Write the LSF of a cube of KMOS3D (from the resolving power of its
    primary header) as a table of its FWHM against wavelength (see
    gbkfit.prep.kmos3d_lsf_table).
    """
    output_dir = _output_dir(output_dir, output_dir_mode)
    # (KMOS3D writes its resolving power in the primary header, of no
    # array, and its cube of the flux in the HDU after it)
    table = prep.kmos3d_lsf_table(
        fits.getheader(file, 0), read_header((file, 1)))
    filename = _write_table(table, output_dir, 'lsf')
    _log.info(
        f"the LSF is in {filename}; use it as lsf: {{type: gauss, fwhm: "
        f"{{file: {filename}, points: wavelength, column: fwhm}}}}")


def prep_psf_kmos3d(
        file: str,
        output_dir: str = '.',
        output_dir_mode: _Exists = 'overwrite'
) -> None:
    """
    Write the PSF of a cube of KMOS3D, the Moffat fitted to the image of
    its PSF (see gbkfit.prep.kmos3d_psf), as the configuration of the PSF
    (prep_psf.yaml, e.g. a piece of the PSFs of a project).
    """
    # (KMOS3D writes the image of the PSF in an HDU of its own, named for
    # the object, e.g. 'U3_10584 PSF')
    with fits.open(file) as hdus:
        headers = [x.header for x in hdus if x.name.endswith(' PSF')]
    if not headers:
        raise ValueError(
            f"{file}: there is no image of the PSF (an HDU named *PSF)")
    psf = prep.kmos3d_psf(headers[0])
    output_dir = _output_dir(output_dir, output_dir_mode)
    filename = os.path.join(output_dir, 'prep_psf.yaml')
    with open(filename, 'w') as f:
        ruamel.yaml.YAML().dump(psf, f)
    options = ', '.join(f'{key}: {value:.6g}' if isinstance(value, float)
                        else f'{key}: {value}' for key, value in psf.items())
    _log.info(f"the PSF is in {filename}; use it as psf: {{{options}}}")


def prep_psf_casa_beams(
        file: str,
        output_dir: str = '.',
        output_dir_mode: _Exists = 'overwrite'
) -> None:
    """
    Write the beams of the channels of a cube of CASA as a table of the
    options of a Gaussian beam PSF (see casa_beam_table).
    """
    output_dir = _output_dir(output_dir, output_dir_mode)
    table = casa_beam_table(file)
    filename = _write_table(table, output_dir, 'psf')
    # The first column of the table is that of the points
    points = table.colnames[0]
    _log.info(
        f"the beams are in {filename}; use them as psf: {{type: gauss, "
        f"bmaj: {{file: {filename}, points: {points}, column: bmaj}}, ...}} "
        f"(and bmin and bpa)")
