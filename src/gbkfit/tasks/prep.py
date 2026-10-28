"""
The prep task: prepare data files for the datasets of gbkfit (see
gbkfit.prep), and write the tables of the PSFs and LSFs of instruments.
Each prepared file is written as prep_<name of the input>, in the output
directory (output_dir; if it exists: an error with output_dir_mode
'terminate', written into with 'overwrite', the default, or a new one
with a numeric suffix with 'unique', e.g. output_2).
"""

import logging
import os
from collections.abc import Sequence
from typing import Literal

import astropy.table
import astropy.wcs
import numpy as np
import numpy.typing as npt
from astropy.io import fits

from gbkfit import prep
from gbkfit.prep import FitsArrays
from gbkfit.utils import fitsutils, tableutils, unitutils
from gbkfit.utils.parseutils import ConfigError
from . import _detail


__all__ = [
    'prep_gauss_hermite_moments',
    'prep_lsf_muse',
    'prep_pixel_brightness',
    'prep_pixel_moments',
    'prep_pixel_spectra',
    'prep_psf_casa_beams',
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
    _detail.make_output_dir).
    """
    output_dir = _detail.make_output_dir(output_dir, mode)
    _log.info(f"output will be stored under directory: {output_dir}")
    return output_dir


def _output(output_dir: str, filename: str | None) -> str | None:
    """Return the name of the prepared file of an input file, if any."""
    if filename is None:
        return None
    name = os.path.splitext(os.path.basename(filename))[0]
    return os.path.join(output_dir, f'prep_{name}.fits')


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
        velocity_rest: str | None,
        files: tuple[str, str | None, str | None],
        dtype: npt.DTypeLike,
        output_dir: str
) -> None:
    """
    Crop the arrays to their valid pixels (minify), pad them (nanpad),
    convert their spectral axis (velocity_rest), and write the prepared
    files of the data, errors and mask files.
    """
    if minify:
        arrays = arrays.minified()
    if nanpad:
        arrays = arrays.padded(nanpad)
    if velocity_rest is not None:
        arrays = arrays.with_velocity_axis(velocity_rest)
    arrays.write(*(_output(output_dir, f) for f in files), dtype=dtype)


def prep_pixel_brightness(
        file_d: str,
        file_e: str | None,
        file_m: str | None,
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
        output_dir_mode: _Exists = 'overwrite'
) -> None:
    """
    Prepare an image (pixel_brightness): crop it to a region (roi_spat:
    x from, x to, y from, y to), clip it, crop it to its valid pixels
    (minify), and pad it with invalid pixels (nanpad).
    """
    output_dir = _output_dir(output_dir, output_dir_mode)
    arrays = FitsArrays.read(file_d, file_e, file_m)
    if roi_spat is not None:
        x0, x1, y0, y1 = roi_spat
        arrays = arrays.crop((slice(y0, y1), slice(x0, x1)))
    arrays = _clip(
        arrays, clip_min, clip_max, sclip_sigma, sclip_iters, ccl_lcount,
        ccl_pcount, ccl_lratio)
    _write(arrays, minify, nanpad, None, (file_d, file_e, file_m), dtype,
           output_dir)


def prep_slit_spectra(
        file_d: str,
        file_e: str | None,
        file_m: str | None,
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
        output_dir: str = '.',
        output_dir_mode: _Exists = 'overwrite'
) -> None:
    """
    Prepare the spectra along a slit (slit_spectra: positions along x,
    channels along y): crop them to positions (roi_spat) and channels
    (roi_spec), clip them, crop them to their valid pixels (minify), pad
    them (nanpad), and convert their spectral axis to velocities
    (velocity_rest). velocity is the convention of the velocities of an
    axis that does not state it (see FitsArrays.read).
    """
    output_dir = _output_dir(output_dir, output_dir_mode)
    arrays = FitsArrays.read(file_d, file_e, file_m, velocity)
    rows, columns = slice(None), slice(None)
    if roi_spec is not None:
        rows = slice(*roi_spec)
    if roi_spat is not None:
        columns = slice(*roi_spat)
    arrays = _clip(
        arrays.crop((rows, columns)), clip_min, clip_max, sclip_sigma,
        sclip_iters, ccl_lcount, ccl_pcount, ccl_lratio)
    _write(arrays, minify, nanpad, velocity_rest, (file_d, file_e, file_m),
           dtype, output_dir)


def prep_pixel_spectra(
        file_d: str,
        file_e: str | None,
        file_m: str | None,
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
        output_dir: str = '.',
        output_dir_mode: _Exists = 'overwrite',
        noise_error: bool = False
) -> None:
    """
    Prepare a cube (pixel_spectra): crop it to a region (roi_spat: x from,
    x to, y from, y to) and channels (roi_spec), clip it, crop it to its
    valid pixels (minify), pad it (nanpad), and convert its spectral axis
    to velocities (velocity_rest). velocity is the convention of the
    velocities of an axis that does not state it (see FitsArrays.read).
    With noise_error, the errors are the noise of each channel of the
    cropped cube, from its negative values (see prep.noise_errors),
    written as prep_<name of the cube>_noise.fits.

    Raises
    ------
    ConfigError
        If both noise_error and an errors file are given.
    """
    if noise_error and file_e is not None:
        raise ConfigError(
            "give a file of the errors or estimate them from the noise "
            "(noise_error), not both")
    output_dir = _output_dir(output_dir, output_dir_mode)
    arrays = FitsArrays.read(file_d, file_e, file_m, velocity)
    channels, ys, xs = slice(None), slice(None), slice(None)
    if roi_spat is not None:
        xs, ys = slice(*roi_spat[0:2]), slice(*roi_spat[2:4])
    if roi_spec is not None:
        channels = slice(*roi_spec)
    arrays = arrays.crop((channels, ys, xs))
    if noise_error:
        # (of the noise before the clipping, which masks some of it)
        arrays = FitsArrays(
            arrays.data, prep.noise_errors(arrays.data, arrays.mask),
            arrays.mask, arrays.header)
        file_e = os.path.splitext(file_d)[0] + '_noise.fits'
    arrays = _clip(
        arrays, clip_min, clip_max, sclip_sigma, sclip_iters, ccl_lcount,
        ccl_pcount, ccl_lratio)
    _write(arrays, minify, nanpad, velocity_rest, (file_d, file_e, file_m),
           dtype, output_dir)


def prep_pixel_moments(
        file_d: Sequence[str],
        file_e: Sequence[str | None] | None,
        file_m: Sequence[str | None] | None,
        roi_spat: _Range | None,
        clip_min: Sequence[float] | None,
        clip_max: Sequence[float] | None,
        ccl_lcount: int | None,
        ccl_pcount: int | None,
        ccl_lratio: float | None,
        sclip_sigma: Sequence[float] | None,
        sclip_iters: int | Sequence[int] | None,
        minify: bool,
        nanpad: int | None,
        dtype: npt.DTypeLike,
        output_dir: str = '.',
        output_dir_mode: _Exists = 'overwrite'
) -> None:
    """
    Prepare moment maps (pixel_moments), as image (see
    prep_pixel_brightness), but with one mask for all: a pixel invalid in
    one map is invalid in all. The options of the clipping (clip_min,
    clip_max, sclip_sigma and sclip_iters) have one value for each map;
    sclip_iters can also be one value for all maps.
    """
    output_dir = _output_dir(output_dir, output_dir_mode)
    nmaps = len(file_d)
    file_e = file_e or [None] * nmaps
    file_m = file_m or [None] * nmaps
    if np.ndim(sclip_iters) == 0:
        sclip_iters = [sclip_iters] * nmaps
    maps = []
    for i in range(nmaps):
        arrays = FitsArrays.read(file_d[i], file_e[i], file_m[i])
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
        if sclip_sigma is not None:
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
        _write(arrays, False, nanpad, None, (file_d[i], file_e[i], file_m[i]),
               dtype, output_dir)


def _read_bins(filename: str) -> tuple[FitsArrays, np.ndarray, int]:
    """
    Return a map of bins, the bin of each pixel numbered from 0 (-1 for
    none; see gbkfit.prep.bin_index), and the number of bins.
    """
    bins = FitsArrays.read(filename)
    if bins.data.ndim != 2:
        raise ConfigError(
            f"{filename}: the bins must be an image; they have "
            f"{bins.data.ndim} axes")
    try:
        index, nbins = prep.bin_index(bins.data)
    except ConfigError as e:
        raise ConfigError(f"{filename}: {e}") from e
    _log.info(f"{nbins} bins")
    return bins, index, nbins


def _write_bins(
        bins: FitsArrays,
        index: np.ndarray,
        filename: str,
        output_dir: str
) -> None:
    """Write the bins numbered from 0 (prep_<bins>.fits)."""
    FitsArrays(index, None, np.ones(index.shape), bins.header).write(
        _output(output_dir, filename), dtype=np.int32)


def prep_region_moments(
        file_bins: str,
        file_d: Sequence[str],
        file_e: Sequence[str | None] | None,
        file_m: Sequence[str | None] | None,
        dtype: npt.DTypeLike,
        output_dir: str = '.',
        output_dir_mode: _Exists = 'overwrite'
) -> None:
    """
    Prepare binned moment maps (e.g. of MaNGA DAP or GIST) for
    region_moments: a map of the bin of each pixel (file_bins; negative or
    NaN for no bin), and moment maps (file_d, with optional errors file_e
    and masks file_m) whose pixels hold the value of their bin. Write the
    bins, numbered from 0 in the order of their numbers in the map
    (prep_<bins>.fits), and for each map a vector of the value of each
    bin (prep_<map>.fits), whose masked or empty bins are NaN.
    """
    output_dir = _output_dir(output_dir, output_dir_mode)
    nmaps = len(file_d)
    file_e = file_e or [None] * nmaps
    file_m = file_m or [None] * nmaps
    bins, index, nbins = _read_bins(file_bins)
    _write_bins(bins, index, file_bins, output_dir)
    for i in range(nmaps):
        arrays = FitsArrays.read(file_d[i], file_e[i], file_m[i])
        if arrays.data.shape != index.shape:
            raise ConfigError(
                f"{file_d[i]}: the maps must have the shape of the bins "
                f"{index.shape}; they have {arrays.data.shape}")
        for filename, data in ((file_d[i], arrays.data),
                               (file_e[i], arrays.error)):
            if filename is None:
                continue
            try:
                values = prep.bin_values(index, nbins, data)
            except ConfigError as e:
                raise ConfigError(f"{filename}: {e}") from e
            fits.writeto(
                _output(output_dir, filename), values.astype(dtype),
                overwrite=True)


def _spectral_header(cube: FitsArrays, nbins: int) -> fits.Header:
    """
    Return the header of the spectra of the bins of a cube: the bins along
    x (in pixels), and the spectral axis of the cube along y.
    """
    wcs = astropy.wcs.WCS(cube.header)
    if wcs.wcs.spec < 0:
        raise ConfigError("the cube has no spectral axis")
    spectral = wcs.sub([wcs.wcs.spec + 1])
    points = spectral.pixel_to_world_values(np.arange(2))
    header = fits.Header()
    header['CTYPE1'], header['CRPIX1'], header['CRVAL1'] = 'LINEAR', 1, 0
    header['CDELT1'] = 1
    header['CTYPE2'] = spectral.wcs.ctype[0]
    header['CUNIT2'] = str(spectral.wcs.cunit[0])
    header['CRPIX2'], header['CRVAL2'] = 1, float(points[0])
    header['CDELT2'] = float(points[1] - points[0])
    for key in ('RESTFRQ', 'RESTWAV'):
        if key in cube.header:
            header[key] = cube.header[key]
    return fits.PrimaryHDU(
        np.zeros((cube.data.shape[0], nbins)), header).header


def prep_region_spectra(
        file_bins: str,
        file_d: str,
        file_e: str | None,
        file_m: str | None,
        roi_spec: _Range | None,
        dtype: npt.DTypeLike,
        velocity_rest: str | None = None,
        velocity: str | None = None,
        output_dir: str = '.',
        output_dir_mode: _Exists = 'overwrite'
) -> None:
    """
    Prepare the spectra of the bins of a cube for region_spectra: a map of
    the bin of each pixel (file_bins; negative or NaN for no bin, on the
    pixels of the cube), and a cube (file_d, with optional errors file_e
    and mask file_m), optionally cropped to channels (roi_spec) and with
    its spectral axis converted to velocities (velocity_rest; velocity is
    the convention of the velocities of an axis that does not state it,
    see FitsArrays.read). Write the
    bins, numbered from 0 in the order of their numbers in the map
    (prep_<bins>.fits), and the sum of the valid spectra of the pixels of
    each bin (prep_<cube>.fits: the bins along x, the channels along y),
    with its errors added in quadrature (prep_<errors>.fits). With a unit
    (BUNIT), the sums are fluxes, in their unit: the values of pixels per
    solid angle or per beam become the fluxes of the pixels (see
    prep.pixel_fluxes).
    """
    output_dir = _output_dir(output_dir, output_dir_mode)
    bins, index, nbins = _read_bins(file_bins)
    cube = FitsArrays.read(file_d, file_e, file_m, velocity)
    fluxes = prep.pixel_fluxes(cube.header, file_d, cube.data.shape[0])
    if roi_spec is not None:
        cube = cube.crop((slice(*roi_spec), slice(None), slice(None)))
        if fluxes is not None and np.ndim(fluxes[0]):
            fluxes = (fluxes[0][slice(*roi_spec)], fluxes[1])
    if cube.data.shape[1:] != index.shape:
        raise ConfigError(
            f"{file_d}: the cube must have the spatial shape of the bins "
            f"{index.shape}; it has {cube.data.shape[1:]}")
    if velocity_rest is not None:
        cube = cube.with_velocity_axis(velocity_rest)
    spectra, errors = prep.bin_spectra(index, nbins, cube)
    header = _spectral_header(cube, nbins)
    # (the sums of the fluxes of the pixels: the values of the pixels per
    # solid angle or per beam become theirs, see prep.pixel_fluxes)
    if fluxes is not None:
        factor = np.reshape(fluxes[0], (-1, 1))
        spectra = spectra * factor
        errors = None if errors is None else errors * factor
        header['BUNIT'] = unitutils.format_unit(fluxes[1])
    _write_bins(bins, index, file_bins, output_dir)
    FitsArrays(
        spectra, errors, np.isfinite(spectra).astype(float),
        header).write(
            _output(output_dir, file_d), _output(output_dir, file_e),
            dtype=dtype)


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
        raise ConfigError(
            "give the columns of V, sigma, h3 and h4, and of their errors "
            "if any, four each")
    table = tableutils.read_table(file, format_)
    velocity = tableutils.read_column(table, columns[0], None)
    unit = velocity.unit

    def values(name: str, of_velocity: bool) -> np.ndarray:
        value = tableutils.read_column(table, name, None)
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
    options alpha and beta of a Moffat PSF against wavelength (see
    gbkfit.prep.muse_psf_table).
    """
    output_dir = _output_dir(output_dir, output_dir_mode)
    filename = _write_table(
        prep.muse_psf_table(fits.getheader(file, 0)), output_dir, 'psf')
    _log.info(
        f"the PSF is in {filename}; use it as psf: {{type: moffat, "
        f"alpha: {{file: {filename}, column: alpha}}, "
        f"beta: {{file: {filename}, column: beta}}}}")


def prep_lsf_muse(
        output_dir: str = '.',
        output_dir_mode: _Exists = 'overwrite'
) -> None:
    """
    Write the LSF of MUSE as a table of its sigma against wavelength (see
    gbkfit.prep.muse_lsf_table).
    """
    output_dir = _output_dir(output_dir, output_dir_mode)
    filename = _write_table(prep.muse_lsf_table(), output_dir, 'lsf')
    _log.info(
        f"the LSF is in {filename}; use it as lsf: {{type: gauss, "
        f"sigma: {{file: {filename}, column: sigma}}}}")


def prep_psf_casa_beams(
        file: str,
        output_dir: str = '.',
        output_dir_mode: _Exists = 'overwrite'
) -> None:
    """
    Write the beams of the channels of a cube of CASA as a table of the
    options of a Gaussian beam PSF (see fitsutils.casa_beam_table).
    """
    output_dir = _output_dir(output_dir, output_dir_mode)
    filename = _write_table(fitsutils.casa_beam_table(file), output_dir, 'psf')
    _log.info(
        f"the beams are in {filename}; use them as psf: {{type: "
        f"gauss_beam, bmaj: {{file: {filename}, column: bmaj}}, ...}} "
        f"(and bmin and bpa)")
