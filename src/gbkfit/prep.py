"""
Data preparation: data read from FITS files made ready for the datasets
of gbkfit (cropped, masked, clipped, in the axis order and direction of
the model, and with velocity spectral axes), the binning of maps and
cubes for region datasets, and the tables of the PSFs and LSFs of
instruments. The prep task of gbkfit-cli is a client of this module.
"""

import dataclasses
import logging
import re
from collections.abc import Sequence

import astropy.constants
import astropy.io.fits as fits
import astropy.stats
import astropy.table
import astropy.units
import astropy.wcs
import numpy as np
import numpy.typing as npt
import skimage.measure

from gbkfit.utils import fitsutils, gridutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'FitsArrays',
    'bin_index',
    'bin_spectra',
    'bin_values',
    'bounding_box',
    'largest_regions_mask',
    'muse_lsf_table',
    'muse_psf_table',
    'range_mask',
    'sigma_clip_mask'
]


_log = logging.getLogger(__name__)

# The keywords of the world coordinates of each axis, or pair of axes
_WCS_KEYWORDS = re.compile(
    r'(WCSAXES'
    r'|(CRPIX|CRVAL|CDELT|CUNIT|CTYPE|CROTA)\d+'
    r'|(PC|CD|PV|PS)\d+_\d+)$')

# The keywords of the rest of the spectral axis, also in their old forms
_REST_KEYWORDS = ('RESTWAV', 'RESTFRQ', 'RESTWAVE', 'RESTFREQ')

# The wavelengths (Angstrom) of the tables of MUSE (its range, and more)
_MUSE_WAVELENGTHS = np.arange(4600.0, 9401.0, 50.0)


def _has_wcs(header: fits.Header) -> bool:
    """Return whether a header has world coordinates."""
    return any(_WCS_KEYWORDS.match(key) for key in header)


def _with_wcs(header: fits.Header, wcs: astropy.wcs.WCS) -> fits.Header:
    """
    Return a copy of a header with the world coordinates of wcs, and its
    rest (if any).
    """
    header = header.copy()
    for key in [
            key for key in header
            if _WCS_KEYWORDS.match(key) or key in _REST_KEYWORDS]:
        del header[key]
    header.update(wcs.to_header())
    return header


def _drop_axes(header: fits.Header, axes: Sequence[int]) -> fits.Header:
    """
    Return the header of the data without the given axes (numpy axes, of
    length 1), without the world coordinates of these axes.
    """
    if not _has_wcs(header):
        return header
    wcs = astropy.wcs.WCS(header)
    for axis in sorted(wcs.naxis - 1 - axis for axis in axes)[::-1]:
        wcs = wcs.dropaxis(axis)
    return _with_wcs(header, wcs)


def _shift_axes(header: fits.Header, offset: npt.ArrayLike) -> fits.Header:
    """
    Return the header of the data whose first pixel is the pixel at the
    given offset (on each numpy axis) of the original data: positive where
    the data were cropped, negative where they were padded.
    """
    if not _has_wcs(header):
        return header
    wcs = astropy.wcs.WCS(header)
    wcs.wcs.crpix -= np.asarray(offset)[::-1]
    return _with_wcs(header, wcs)


def _check_same_rest(wcs: astropy.wcs.WCS, rest: astropy.units.Quantity
                     ) -> None:
    """
    Raise ConfigError unless the velocity axis of wcs has no rest, or the
    given one (they are compared as frequencies).
    """
    hz, spectral = astropy.units.Hz, astropy.units.spectral()
    current = (
        wcs.wcs.restfrq * hz if wcs.wcs.restfrq > 0
        else wcs.wcs.restwav * astropy.units.m if wcs.wcs.restwav > 0
        else None)
    if current is None:
        return
    current = current.to_value(hz, spectral)
    if not np.isclose(rest.to_value(hz, spectral), current, rtol=1e-9):
        raise ConfigError(
            f"the velocities of the spectral axis refer to the rest "
            f"{current * hz}, not {rest}; give that rest, or convert the "
            f"velocities to the new rest first")


def _spectral_to_velocity(
        header: fits.Header,
        rest: str | astropy.units.Quantity
) -> fits.Header:
    """
    Return the header with its spectral axis converted to the velocities
    of a rest (see gridutils.make_rest): a linear wavelength axis (WAVE, or
    AWAV with a rest in air too) to the optical velocities c (w / rest -
    1), a linear frequency axis (FREQ) to the radio velocities c (1 - f /
    rest). Both are linear in the pixels, so the conversion is exact; the
    header gets the rest (RESTWAV or RESTFRQ), and for wavelengths their
    medium, in which the velocities and the rest are (AIRORVAC: 'air' for
    AWAV, 'vac' for WAVE; see fitsutils.read_data). A velocity axis (VOPT,
    VRAD) gets the rest only, and if it has one, it must be the same: its
    velocities refer to it.
    """
    rest = gridutils.make_rest(rest)
    wcs = astropy.wcs.WCS(header)
    s = wcs.wcs.spec
    if s < 0:
        raise ConfigError("the data have no spectral axis to convert")
    ctype = wcs.wcs.ctype[s]
    kind = ctype[:4]
    if ctype[4:].strip('-').strip():
        raise ConfigError(
            f"the spectral axis ({ctype}) is not linear in its coordinate")
    c = astropy.constants.c.to_value('km/s')
    m, hz = astropy.units.m, astropy.units.Hz
    rest_wavelength = rest.to_value(m, astropy.units.spectral())
    rest_frequency = rest.to_value(hz, astropy.units.spectral())
    unit = astropy.units.Unit(str(wcs.wcs.cunit[s]) or '1')
    if kind in ('WAVE', 'AWAV'):
        # The wavelength in units of the rest, and the optical velocity
        ratio = unit.to(m) / rest_wavelength
        crval, factor = c * (wcs.wcs.crval[s] * ratio - 1), c * ratio
        velocity = 'VOPT'
    elif kind == 'FREQ':
        ratio = unit.to(hz) / rest_frequency
        crval, factor = c * (1 - wcs.wcs.crval[s] * ratio), -c * ratio
        velocity = 'VRAD'
    elif kind in ('VOPT', 'VRAD'):
        crval, factor, velocity = None, None, kind
        _check_same_rest(wcs, rest)
    else:
        raise ConfigError(
            f"the spectral axis ({ctype}) is not a wavelength, frequency, "
            f"or optical or radio velocity")
    if crval is not None:
        wcs.wcs.ctype[s] = velocity
        wcs.wcs.cunit[s] = 'km/s'
        wcs.wcs.crval[s] = crval
        if wcs.wcs.has_cd():
            cd = wcs.wcs.cd.copy()
            cd[s, :] *= factor
            wcs.wcs.cd = cd
        else:
            wcs.wcs.cdelt[s] *= factor
    if velocity == 'VOPT':
        wcs.wcs.restwav, wcs.wcs.restfrq = rest_wavelength, 0
    else:
        wcs.wcs.restfrq, wcs.wcs.restwav = rest_frequency, 0
    _log.info(
        f"converting the spectral axis ({ctype}) to {velocity} velocities "
        f"of the rest {rest}")
    header = _with_wcs(header, wcs)
    if kind in ('WAVE', 'AWAV'):
        header['AIRORVAC'] = 'air' if kind == 'AWAV' else 'vac'
    return header


def _read_fits(filename: str) -> tuple[np.ndarray, fits.Header]:
    """
    Return the data and the header of a file, without the axes of length
    1: those of its first HDU with data (e.g. the extension after an empty
    primary HDU, as in MUSE and JWST cubes). Integer data become floats,
    which can hold the NaN of invalid pixels.
    """
    with fits.open(filename) as hdus:
        hdu = next((hdu for hdu in hdus if hdu.data is not None), None)
        if hdu is None:
            raise ConfigError(f"{filename}: the file has no data")
        data = np.array(hdu.data)
        header = hdu.header.copy()
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(float)
    # Astropy writes the FITS default for a missing reference pixel, so
    # write the one of the model (see fitsutils.read_data)
    if _has_wcs(header):
        header = fitsutils.centre_missing_crpix(header, data.shape)
    axes = [axis for axis, length in enumerate(data.shape) if length == 1]
    data = np.squeeze(data)
    # The header of the squeezed data: its NAXISn, and the world
    # coordinates of its axes
    header = fits.PrimaryHDU(data, _drop_axes(header, axes)).header
    return data, header


@dataclasses.dataclass(frozen=True)
class FitsArrays:
    """
    Data read from FITS files, being prepared for a dataset: the data,
    their errors (if any) and mask (1 where valid), with the header of the
    data (their world coordinates, if any). The data and errors are NaN
    where the mask is 0. The axes are in the order of the model (e.g. RA
    before Dec; see fitsutils.model_axis_order).

    The methods return new arrays; those that move the first pixel
    (crop, minified, padded) keep the world coordinates of the pixels.

    Attributes
    ----------
    data : np.ndarray
        The data.
    error : np.ndarray or None
        The errors.
    mask : np.ndarray
        The mask: 1 where valid, 0 elsewhere.
    header : fits.Header
        The header of the data.
    """
    data: np.ndarray
    error: np.ndarray | None
    mask: np.ndarray
    header: fits.Header

    @classmethod
    def read(
            cls,
            data: str,
            error: str | None = None,
            mask: str | None = None
    ) -> 'FitsArrays':
        """
        Read the data, errors and mask from FITS files (their first HDU
        with data), without their axes of length 1, in the axis order of
        the model. Values that are not finite, errors that are not finite
        or are 0 (e.g. blank pixels of an error map), and pixels the mask
        file marks with 0 are invalid; negative errors are an error.

        Parameters
        ----------
        data : str
            The file of the data (with the world coordinates).
        error, mask : str, optional
            The files of the errors and the mask, of the shape of the data.

        Returns
        -------
        FitsArrays
            The arrays.

        Raises
        ------
        ConfigError
            If a file has no data, the shapes differ, or an error is
            negative.
        """
        data_d, header = _read_fits(data)
        data_e = _read_fits(error)[0] if error else None
        data_m = _read_fits(mask)[0] if mask else None
        for name, array in (('error', data_e), ('mask', data_m)):
            if array is not None and array.shape != data_d.shape:
                raise ConfigError(
                    f"the {name} must have the shape of the data "
                    f"{data_d.shape}; it has {array.shape}")
        # (the error and the mask follow the axes of the data, whose world
        # coordinates they may not have)
        if _has_wcs(header):
            order = fitsutils.model_axis_order(astropy.wcs.WCS(header))
            if order != tuple(range(data_d.ndim)):
                _log.info("reordering the axes in the order of the model")
                wcs = fitsutils.reorder_wcs(astropy.wcs.WCS(header), order)
                header = _with_wcs(header, wcs)
                data_d = fitsutils.reorder_data(data_d, order)
                if data_e is not None:
                    data_e = fitsutils.reorder_data(data_e, order)
                if data_m is not None:
                    data_m = fitsutils.reorder_data(data_m, order)
        valid = np.isfinite(data_d)
        if data_e is not None:
            if np.any(data_e < 0):
                raise ConfigError(
                    f"errors cannot be negative; "
                    f"{np.count_nonzero(data_e < 0)} are, down to "
                    f"{np.nanmin(data_e):g}")
            valid &= np.isfinite(data_e) & (data_e > 0)
        if data_m is not None:
            valid &= np.isfinite(data_m) & (data_m != 0)
        return cls(
            np.ascontiguousarray(data_d),
            None if data_e is None else np.ascontiguousarray(data_e),
            np.ones(data_d.shape), header).masked(valid)

    def masked(self, mask: npt.ArrayLike) -> 'FitsArrays':
        """
        Return the arrays with the pixels a mask marks invalid (0 or
        False) also invalid.
        """
        valid = (self.mask != 0) & (np.asarray(mask) != 0)
        data = np.where(valid, self.data, np.nan)
        error = None if self.error is None \
            else np.where(valid, self.error, np.nan)
        return FitsArrays(data, error, valid.astype(float), self.header)

    def crop(self, slices: Sequence[slice]) -> 'FitsArrays':
        """
        Return the arrays cropped to ranges of pixels (one slice per numpy
        axis, of step 1).
        """
        slices = tuple(slices)
        offset = [s.indices(n)[0] for s, n in zip(slices, self.data.shape)]
        return FitsArrays(
            self.data[slices],
            None if self.error is None else self.error[slices],
            self.mask[slices], _shift_axes(self.header, offset))

    def minified(self) -> 'FitsArrays':
        """
        Return the arrays cropped to their valid pixels (see
        bounding_box).

        Raises
        ------
        ConfigError
            If no pixel is valid.
        """
        return self.crop(bounding_box(self.mask))

    def padded(self, size: int) -> 'FitsArrays':
        """Return the arrays padded with invalid pixels on every side."""
        def pad(array: np.ndarray, value: float) -> np.ndarray:
            return np.pad(array, size, 'constant', constant_values=value)
        return FitsArrays(
            pad(self.data, np.nan),
            None if self.error is None else pad(self.error, np.nan),
            pad(self.mask, 0),
            _shift_axes(self.header, [-size] * self.data.ndim))

    def with_velocity_axis(
            self, rest: str | astropy.units.Quantity
    ) -> 'FitsArrays':
        """
        Return the arrays with their spectral axis in velocities of a rest
        (a wavelength or frequency; see gridutils.make_rest): optical
        velocities from wavelengths, radio velocities from frequencies.

        Raises
        ------
        ConfigError
            If there is no linear spectral axis, or its velocities refer to
            another rest.
        """
        return dataclasses.replace(
            self, header=_spectral_to_velocity(self.header, rest))

    def write(
            self,
            data: str,
            error: str | None = None,
            mask: str | None = None,
            dtype: npt.DTypeLike = np.float32
    ) -> None:
        """
        Write the data, errors and mask (if given filenames) as FITS
        files, with the header of the data. The axes that the model needs
        reversed (see fitsutils.model_axis_flips: that of a mirrored
        image, and a spectral axis of decreasing velocities) are reversed.

        Parameters
        ----------
        data : str
            The file of the data.
        error, mask : str, optional
            The files of the errors and the mask.
        dtype : DTypeLike, optional
            The type of the values.
        """
        arrays = [self.data, self.error, self.mask]
        header = self.header
        flips = fitsutils.model_axis_flips(astropy.wcs.WCS(header)) \
            if _has_wcs(header) else ()
        for axis in flips:
            _log.info(
                f"reversing the axis {axis + 1}: the image is mirrored, or "
                f"the velocity decreases along it")
            size = self.data.shape[self.data.ndim - 1 - axis]
            header = _with_wcs(header, fitsutils.flip_wcs(
                astropy.wcs.WCS(header), axis, size))
            arrays = [
                None if a is None else fitsutils.flip_data(a, axis)
                for a in arrays]
        for filename, array in zip((data, error, mask), arrays):
            if filename is not None and array is not None:
                fits.writeto(
                    filename, array.astype(dtype), header, overwrite=True)


def range_mask(
        data: np.ndarray,
        minimum: float | None = None,
        maximum: float | None = None
) -> np.ndarray:
    """
    Return the mask of the values in a range (True where kept); NaN values
    are left to the other masks (kept here).
    """
    with np.errstate(invalid='ignore'):
        kept = np.ones(data.shape, bool)
        if minimum is not None:
            kept &= ~(data < minimum)
        if maximum is not None:
            kept &= ~(data > maximum)
    return kept


def sigma_clip_mask(
        data: np.ndarray,
        mask: np.ndarray,
        sigma: float,
        maxiters: int | None = 5
) -> np.ndarray:
    """
    Return the mask of the values that sigma clipping keeps (True where
    kept), of those a mask keeps (the others do not count in its
    statistics, and are not kept).
    """
    kept = np.ma.masked_array(data, mask=(mask == 0))
    return ~astropy.stats.sigma_clip(
        kept, sigma=sigma, maxiters=maxiters).mask


def largest_regions_mask(
        mask: np.ndarray,
        count: int | None = None,
        min_pixels: int | None = None,
        min_ratio: float | None = None
) -> np.ndarray:
    """
    Return the mask (True where kept) of the largest connected regions of
    the pixels a mask keeps: the count largest, those of at least
    min_pixels pixels, and those at least min_ratio times the largest
    (each criterion if given).
    """
    regions = skimage.measure.regionprops(skimage.measure.label(mask != 0))
    regions.sort(key=lambda r: r.area, reverse=True)
    if count is not None:
        regions = regions[:count]
    if min_pixels is not None:
        regions = [r for r in regions if r.area >= min_pixels]
    if min_ratio is not None and regions:
        regions = [r for r in regions if r.area / regions[0].area >= min_ratio]
    kept = np.zeros(mask.shape, bool)
    for region in regions:
        kept[tuple(region.coords.T)] = True
    return kept


def bounding_box(mask: np.ndarray) -> tuple[slice, ...]:
    """
    Return the smallest ranges of pixels (one slice per axis) that hold
    the pixels a mask keeps.

    Raises
    ------
    ConfigError
        If the mask keeps no pixel.
    """
    nonzero = np.nonzero(mask)
    if not nonzero[0].size:
        raise ConfigError(
            "no valid pixels are left to crop the data to; relax the "
            "clipping, or do not minify")
    return tuple(slice(i.min(), i.max() + 1) for i in nonzero)


def bin_index(bins: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Return the bins of a map numbered from 0 (in the order of their
    numbers; -1 for no bin), and their number.

    Parameters
    ----------
    bins : np.ndarray
        The number of the bin of each pixel (negative or NaN for no bin).

    Raises
    ------
    ConfigError
        If the bins are not integers.
    """
    valid = np.isfinite(bins) & (bins >= 0)
    if np.any(bins[valid] != np.round(bins[valid])):
        raise ConfigError("the bins must be integers")
    numbers = np.unique(bins[valid])
    index = np.full(bins.shape, -1, dtype=np.int32)
    index[valid] = np.searchsorted(numbers, bins[valid])
    return index, len(numbers)


def bin_values(index: np.ndarray, nbins: int, data: np.ndarray) -> np.ndarray:
    """
    Return the value of each bin of a map whose pixels hold the value of
    their bin (NaN for a bin without finite pixels).

    Raises
    ------
    ConfigError
        If the finite pixels of a bin hold different values.
    """
    values = np.full(nbins, np.nan)
    for i in range(nbins):
        pixels = data[(index == i) & np.isfinite(data)]
        if pixels.size == 0:
            continue
        if not np.allclose(pixels, pixels[0], rtol=1e-6, atol=0):
            raise ConfigError(
                f"the pixels of bin {i} hold different values; the map must "
                f"hold the value of each bin on its pixels")
        values[i] = pixels[0]
    return values


def bin_spectra(
        index: np.ndarray,
        nbins: int,
        cube: FitsArrays
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Return the spectrum of each bin of a cube: the sum of the valid
    spectra of its pixels (a flux, as the region datasets compare; in
    float64), and its errors (added in quadrature). A channel of a bin
    without valid pixels is NaN.

    Parameters
    ----------
    index : np.ndarray
        The bin of each pixel, from 0 (-1 for none; see bin_index).
    nbins : int
        The number of bins.
    cube : FitsArrays
        The cube (channels first).

    Returns
    -------
    np.ndarray
        The spectra, of shape (nchannels, nbins).
    np.ndarray or None
        Their errors, if the cube has some.
    """
    nchannels = cube.data.shape[0]
    spectra = np.full((nchannels, nbins), np.nan)
    errors = None if cube.error is None else spectra.copy()
    valid = cube.mask != 0
    for i in range(nbins):
        pixels = index == i
        counts = valid[:, pixels].sum(axis=1)
        sums = np.nansum(cube.data[:, pixels], axis=1, dtype=np.float64)
        spectra[:, i] = np.where(counts > 0, sums, np.nan)
        if errors is not None:
            variances = np.nansum(
                cube.error[:, pixels].astype(np.float64) ** 2, axis=1)
            errors[:, i] = np.where(counts > 0, np.sqrt(variances), np.nan)
    return spectra, errors


def muse_psf_table(header: fits.Header) -> astropy.table.QTable:
    """
    Return the PSF of a MUSE cube (from the keywords of its header: the
    MOFFAT1 and MOFFAT2 modes of MPDAF, its first field) as a table of the
    options alpha and beta of a Moffat PSF against wavelength.

    Raises
    ------
    ConfigError
        If the header has no PSF of MUSE.
    """
    wavelengths = _MUSE_WAVELENGTHS
    mode = header.get('FSFMODE')
    fields = [f for f in (*range(10), 99) if f'FSF{f:02d}BET' in header
              or f'FSF{f:02d}FNC' in header]
    if mode not in ('MOFFAT1', 'MOFFAT2') or not fields:
        raise ConfigError(
            f"the header has no PSF of MUSE (FSFMODE MOFFAT1 or MOFFAT2); "
            f"FSFMODE is {mode!r}")
    key = f'FSF{fields[0]:02d}'
    if mode == 'MOFFAT1':
        fwhm = header[f'{key}FWA'] + header[f'{key}FWB'] * wavelengths
        beta = np.full_like(wavelengths, header[f'{key}BET'])
    else:
        # (polynomials of the wavelength normalised to [-0.5, 0.5] in the
        # range FSFLB1 to FSFLB2, the coefficient of the highest power
        # first)
        lb1, lb2 = header['FSFLB1'], header['FSFLB2']
        x = (wavelengths - lb1) / (lb2 - lb1) - 0.5
        fwhm = np.polyval([
            header[f'{key}F{k:02d}']
            for k in range(header[f'{key}FNC'])], x)
        beta = np.polyval([
            header[f'{key}B{k:02d}']
            for k in range(header[f'{key}BNC'])], x)
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    u = astropy.units
    return astropy.table.QTable(dict(
        wavelength=wavelengths * u.Angstrom, alpha=alpha * u.arcsec,
        beta=beta * u.dimensionless_unscaled))


def muse_lsf_table() -> astropy.table.QTable:
    """
    Return the LSF of MUSE (a Gaussian whose FWHM is that of Bacon et al.
    2017, eq. 8) as a table of its sigma against wavelength.
    """
    w = _MUSE_WAVELENGTHS
    fwhm = 5.866e-8 * w ** 2 - 9.187e-4 * w + 6.040
    u = astropy.units
    return astropy.table.QTable(dict(
        wavelength=w * u.Angstrom,
        sigma=fwhm / np.sqrt(8 * np.log(2)) * u.Angstrom))


