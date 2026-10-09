
import logging
import os
import re

import astropy.constants
import astropy.io.fits as fits
import astropy.stats as stats
import astropy.units
import astropy.wcs
import numpy as np
import skimage.measure

from gbkfit.utils import fitsutils
from gbkfit.utils.parseutils import ConfigError


_log = logging.getLogger(__name__)

# The keywords of the world coordinates of each axis, or pair of axes
_WCS_KEYWORDS = re.compile(
    r'(WCSAXES'
    r'|(CRPIX|CRVAL|CDELT|CUNIT|CTYPE|CROTA)\d+'
    r'|(PC|CD|PV|PS)\d+_\d+)$')


def _has_wcs(header):
    return any(_WCS_KEYWORDS.match(key) for key in header)


def _with_wcs(header, wcs):
    """A copy of the header with the world coordinates of wcs."""
    header = header.copy()
    for key in [key for key in header if _WCS_KEYWORDS.match(key)]:
        del header[key]
    header.update(wcs.to_header())
    return header


def _drop_axes(header, axes):
    """
    The header of the data without the given axes (numpy axes, of length
    1), without the world coordinates of these axes.
    """
    if not _has_wcs(header):
        return header
    wcs = astropy.wcs.WCS(header)
    for axis in sorted(wcs.naxis - 1 - axis for axis in axes)[::-1]:
        wcs = wcs.dropaxis(axis)
    return _with_wcs(header, wcs)


def _shift_axes(header, offset):
    """
    The header of the data whose first pixel is the pixel at the given
    offset (on each numpy axis) of the original data: positive where the
    data was cropped, negative where it was padded.
    """
    if not _has_wcs(header):
        return header
    wcs = astropy.wcs.WCS(header)
    wcs.wcs.crpix -= offset[::-1]
    return _with_wcs(header, wcs)


def _spectral_to_velocity(header, rest):
    """
    The header with its spectral axis converted to the velocities of the
    given rest (see fitsutils.make_rest; None leaves it unchanged): a
    linear wavelength axis (WAVE, or AWAV with a rest in air too) to the
    optical velocities c (w / rest - 1), a linear frequency axis (FREQ) to
    the radio velocities c (1 - f / rest). Both are linear in the pixels,
    so the conversion is exact; the header gets the rest (RESTWAV or
    RESTFRQ). A velocity axis (VOPT, VRAD) gets the rest only.
    """
    if rest is None:
        return header
    rest = fitsutils.make_rest(rest)
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
    return _with_wcs(header, wcs)


def _reverse_decreasing_velocity(data, header):
    """
    The data and its header with the spectral axis reversed if it is a
    velocity that decreases along the axis (the model needs increasing
    velocities). Each channel keeps its velocity.
    """
    if not _has_wcs(header):
        return data, header
    wcs = astropy.wcs.WCS(header)
    s = wcs.wcs.spec
    if s < 0 or wcs.wcs.ctype[s][:4] not in fitsutils.VELOCITY_TYPES:
        return data, header
    if wcs.wcs.get_cdelt()[s] * wcs.wcs.get_pc()[s, s] >= 0:
        return data, header
    _log.info("reversing the spectral axis: the velocity decreases along it")
    # Reversing pixel axis s negates column s of the linear transformation
    # (CD, or CDELT times the rows of PC). For PC, negating CDELT[s] and
    # row s of PC cancel out, leaving a positive CDELT[s].
    if wcs.wcs.has_cd():
        cd = wcs.wcs.cd.copy()
        cd[:, s] *= -1
        wcs.wcs.cd = cd
    else:
        pc = wcs.wcs.get_pc().copy()
        pc[s, :] *= -1
        pc[:, s] *= -1
        wcs.wcs.pc = pc
        wcs.wcs.cdelt[s] *= -1
    wcs.wcs.crpix[s] = data.shape[data.ndim - 1 - s] + 1 - wcs.wcs.crpix[s]
    data = np.flip(data, data.ndim - 1 - s)
    return data, _with_wcs(header, wcs)


def _read_fits(filename):
    """The data and the header of a file, without the axes of length 1."""
    data = fits.getdata(filename)
    header = fits.getheader(filename)
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


def _read_data(file_d, file_e, file_m):
    # Load data and headers, without their axes of length 1
    data_d, header_d = _read_fits(file_d)
    data_e, header_e = _read_fits(file_e) if file_e else (None, None)
    data_m, header_m = _read_fits(file_m) if file_m else (None, None)
    # Deal with invalid or non-sensible pixel values
    data_d[~np.isfinite(data_d)] = np.nan
    if data_e is not None:
        data_e[~np.isfinite(data_e)] = np.nan
        data_e[data_e <= 0] = np.nan
    if data_m is not None:
        data_m[~np.isfinite(data_m)] = 0
        data_m[np.nonzero(data_m)] = 1
    return data_d, header_d, data_e, header_e, data_m, header_m


def _save_data(
        file_d, data_d, header_d,
        file_e, data_e, header_e,
        file_m, data_m, header_m,
        offset, dtype, velocity_rest=None):
    """
    Save the data, whose first pixel is the pixel at the given offset (on
    each numpy axis) of the data read, with its spectral axis converted to
    the velocities of velocity_rest (see _spectral_to_velocity; the
    error and mask files are converted if they have world coordinates),
    and a decreasing velocity axis reversed.
    """
    basename = os.path.basename
    splitext = os.path.splitext

    def prepare(data, header, has_wcs):
        header = _shift_axes(header, offset)
        if has_wcs:
            header = _spectral_to_velocity(header, velocity_rest)
        return _reverse_decreasing_velocity(data, header)
    data_d, header_d = prepare(data_d, header_d, True)
    if header_e is not None:
        data_e, header_e = prepare(data_e, header_e, _has_wcs(header_e))
    if header_m is not None:
        data_m, header_m = prepare(data_m, header_m, _has_wcs(header_m))
    data_d = data_d.astype(dtype)
    file_d = splitext(basename(file_d))[0]
    fits.writeto(f'prep_{file_d}.fits', data_d, header_d, overwrite=True)
    if data_e is not None:
        data_e = data_e.astype(dtype)
        file_e = splitext(basename(file_e))[0]
        fits.writeto(f'prep_{file_e}.fits', data_e, header_e, overwrite=True)
    if data_m is not None:
        data_m = data_m.astype(dtype)
        file_m = splitext(basename(file_m))[0] if file_m else file_d + '_mask'
        fits.writeto(f'prep_{file_m}.fits', data_m, header_m, overwrite=True)


def _crop_data(data_d, data_e, data_m, axis, range_):
    s = [slice(None), ] * data_d.ndim
    s[axis] = slice(range_[0], range_[1])
    data_d = data_d[tuple(s)]
    if data_e is not None:
        data_e = data_e[tuple(s)]
    if data_m is not None:
        data_m = data_m[tuple(s)]
    return data_d, data_e, data_m


def _make_mask(data_d, data_e, data_m):
    """
    The pixels with finite data and error, and not masked (data_m is 0
    or 1, see _read_data).
    """
    mask = np.ones_like(data_d)
    mask *= np.isfinite(data_d)
    if data_e is not None:
        mask *= np.isfinite(data_e)
    if data_m is not None:
        mask *= data_m != 0
    return mask


def _compare_nan_array(func, ary, threshold):
    out = ~np.isnan(ary)
    out[out] = func(ary[out], threshold)
    return out


def _make_mask_clip_min(data, min_value):
    return ~_compare_nan_array(np.less, data, min_value)


def _make_mask_clip_max(data, max_value):
    return ~_compare_nan_array(np.greater, data, max_value)


def _make_mask_clip_sig(data, sigma, maxiters):
    """The pixels that sigma clipping keeps (its mask marks the others)."""
    return ~stats.sigma_clip(data, sigma=sigma, maxiters=maxiters).mask


def _make_mask_clip_ccl(data, lcount, pcount, lratio):
    """
    The largest connected regions of the pixels of data that are not 0:
    the lcount largest, those of at least pcount pixels, and those at
    least lratio times the largest.
    """
    labels = skimage.measure.label(data)
    props = skimage.measure.regionprops(labels)
    props.sort(key=lambda x: x.area, reverse=True)
    if lcount is not None:
        props = props[0:lcount]
    if pcount is not None:
        props = [p for p in props if p.area >= pcount]
    if lratio is not None and props:
        props = [p for p in props if p.area / props[0].area >= lratio]
    mask = np.zeros_like(data)
    for p in props:
        mask[tuple(p.coords.T)] = 1
    return mask


def _apply_mask(data_d, data_e, data_m, mask):
    data_d[mask == 0] = np.nan
    if data_e is not None:
        data_e[mask == 0] = np.nan
    if data_m is not None:
        data_m[mask == 0] = 0


def _minify_data(data_d, data_e, data_m, mask):
    """
    Crop the data to the masked pixels. Return it, and the index of its
    first pixel on each axis.
    """
    nonzero = mask.nonzero()
    start = np.array([indices.min() for indices in nonzero])
    slices = tuple(
        slice(indices.min(), indices.max() + 1) for indices in nonzero)
    data_d = data_d[slices]
    if data_e is not None:
        data_e = data_e[slices]
    if data_m is not None:
        data_m = data_m[slices]
    return data_d, data_e, data_m, start


def _pad_data(data_d, data_e, data_m, size, value_d, value_e, value_m):
    data_d = np.pad(data_d, size, 'constant', constant_values=value_d)
    if data_e is not None:
        data_e = np.pad(data_e, size, 'constant', constant_values=value_e)
    if data_m is not None:
        data_m = np.pad(data_m, size, 'constant', constant_values=value_m)
    return data_d, data_e, data_m


def prep_image(
        file_d, file_e, file_m,
        roi_spat, clip_min, clip_max, ccl_lcount, ccl_pcount, ccl_lratio,
        sclip_sigma, sclip_iters, minify, nanpad, dtype):

    (data_d, header_d,
     data_e, header_e,
     data_m, header_m) = _read_data(file_d, file_e, file_m)
    # The index of the first pixel of the prepared data in the data read
    offset = np.zeros(data_d.ndim, int)

    if roi_spat is not None:
        xrange = roi_spat[0:2]
        yrange = roi_spat[2:4]
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 1, xrange)
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 0, yrange)
        offset += [yrange[0], xrange[0]]

    mask = _make_mask(data_d, data_e, data_m)

    if clip_min is not None:
        mask *= _make_mask_clip_min(data_d, clip_min)
    if clip_max is not None:
        mask *= _make_mask_clip_max(data_d, clip_max)
    if sclip_sigma is not None:
        mask *= _make_mask_clip_sig(data_d, sclip_sigma, sclip_iters)
    if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
        mask *= _make_mask_clip_ccl(
            mask != 0, ccl_lcount, ccl_pcount, ccl_lratio)

    _apply_mask(data_d, data_e, data_m, mask)

    if minify:
        data_d, data_e, data_m, start = _minify_data(
            data_d, data_e, data_m, mask)
        offset += start

    if nanpad:
        data_d, data_e, data_m = _pad_data(
            data_d, data_e, data_m, nanpad, np.nan, np.nan, 0)
        offset -= nanpad

    _save_data(
        file_d, data_d, header_d,
        file_e, data_e, header_e,
        file_m, data_m, header_m,
        offset, dtype)


def prep_lslit(
        file_d, file_e, file_m,
        roi_spat, roi_spec, clip_min, clip_max,
        ccl_lcount, ccl_pcount, ccl_lratio,
        sclip_sigma, sclip_iters, minify, nanpad, dtype,
        velocity_rest=None):

    (data_d, header_d,
     data_e, header_e,
     data_m, header_m) = _read_data(file_d, file_e, file_m)
    # The index of the first pixel of the prepared data in the data read
    offset = np.zeros(data_d.ndim, int)

    if roi_spat is not None:
        xrange = roi_spat
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 0, xrange)
        offset[0] += xrange[0]
    if roi_spec is not None:
        srange = roi_spec
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 1, srange)
        offset[1] += srange[0]

    mask = _make_mask(data_d, data_e, data_m)

    if clip_min is not None:
        mask *= _make_mask_clip_min(data_d, clip_min)
    if clip_max is not None:
        mask *= _make_mask_clip_max(data_d, clip_max)
    if sclip_sigma is not None:
        mask *= _make_mask_clip_sig(data_d, sclip_sigma, sclip_iters)
    if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
        mask *= _make_mask_clip_ccl(
            mask != 0, ccl_lcount, ccl_pcount, ccl_lratio)

    _apply_mask(data_d, data_e, data_m, mask)

    if minify:
        data_d, data_e, data_m, start = _minify_data(
            data_d, data_e, data_m, mask)
        offset += start

    if nanpad:
        data_d, data_e, data_m = _pad_data(
            data_d, data_e, data_m, nanpad, np.nan, np.nan, 0)
        offset -= nanpad

    _save_data(
        file_d, data_d, header_d,
        file_e, data_e, header_e,
        file_m, data_m, header_m,
        offset, dtype, velocity_rest)


def prep_mmaps(
        file_d, file_e, file_m,
        roi_spat, clip_min, clip_max, ccl_lcount, ccl_pcount, ccl_lratio,
        sclip_sigma, sclip_iters, minify, nanpad, dtype):
    """
    The options of the clipping (clip_min, clip_max, sclip_sigma and
    sclip_iters) have one value for each map; sclip_iters can also be
    one value for all maps.
    """

    nmmaps = len(file_d)
    if file_e is None:
        file_e = [None] * nmmaps
    if file_m is None:
        file_m = [None] * nmmaps
    if np.ndim(sclip_iters) == 0:
        sclip_iters = [sclip_iters] * nmmaps

    data_d = []
    data_e = []
    data_m = []
    header_d = []
    header_e = []
    header_m = []
    # The index of the first pixel of the prepared data in the data read
    offset = np.zeros(2, int)
    if roi_spat is not None:
        offset += [roi_spat[2], roi_spat[0]]
    for i in range(nmmaps):
        (data_d_, header_d_,
         data_e_, header_e_,
         data_m_, header_m_) = _read_data(file_d[i], file_e[i], file_m[i])
        if roi_spat is not None:
            xrange = roi_spat[0:2]
            yrange = roi_spat[2:4]
            data_d_, data_e_, data_m_ = _crop_data(
                data_d_, data_e_, data_m_, 1, xrange)
            data_d_, data_e_, data_m_ = _crop_data(
                data_d_, data_e_, data_m_, 0, yrange)
        data_d.append(data_d_)
        data_e.append(data_e_)
        data_m.append(data_m_)
        header_d.append(header_d_)
        header_e.append(header_e_)
        header_m.append(header_m_)

    mask = np.ones_like(data_d[0])

    for i in range(nmmaps):
        mask *= _make_mask(data_d[i], data_e[i], data_m[i])
        if clip_min is not None:
            mask *= _make_mask_clip_min(data_d[i], clip_min[i])
        if clip_max is not None:
            mask *= _make_mask_clip_max(data_d[i], clip_max[i])
        if sclip_sigma is not None:
            mask *= _make_mask_clip_sig(
                data_d[i], sclip_sigma[i], sclip_iters[i])
    if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
        mask *= _make_mask_clip_ccl(
            mask != 0, ccl_lcount, ccl_pcount, ccl_lratio)

    if minify:
        offset += [indices.min() for indices in mask.nonzero()]
    if nanpad:
        offset -= nanpad

    for i in range(nmmaps):

        _apply_mask(data_d[i], data_e[i], data_m[i], mask)

        if minify:
            data_d[i], data_e[i], data_m[i], _ = _minify_data(
                data_d[i], data_e[i], data_m[i], mask)

        if nanpad:
            data_d[i], data_e[i], data_m[i] = _pad_data(
                data_d[i], data_e[i], data_m[i], nanpad, np.nan, np.nan, 0)

        _save_data(
            file_d[i], data_d[i], header_d[i],
            file_e[i], data_e[i], header_e[i],
            file_m[i], data_m[i], header_m[i],
            offset, dtype)


def prep_scube(
        file_d, file_e, file_m,
        roi_spat, roi_spec, clip_min, clip_max,
        ccl_lcount, ccl_pcount, ccl_lratio,
        sclip_sigma, sclip_iters, minify, nanpad, dtype,
        velocity_rest=None):

    (data_d, header_d,
     data_e, header_e,
     data_m, header_m) = _read_data(file_d, file_e, file_m)
    # The index of the first pixel of the prepared data in the data read
    offset = np.zeros(data_d.ndim, int)

    if roi_spat is not None:
        xrange = roi_spat[0:2]
        yrange = roi_spat[2:4]
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 2, xrange)
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 1, yrange)
        offset[1:] += [yrange[0], xrange[0]]
    if roi_spec is not None:
        srange = roi_spec
        data_d, data_e, data_m = _crop_data(
            data_d, data_e, data_m, 0, srange)
        offset[0] += srange[0]

    mask = _make_mask(data_d, data_e, data_m)

    if clip_min is not None:
        mask *= _make_mask_clip_min(data_d, clip_min)
    if clip_max is not None:
        mask *= _make_mask_clip_max(data_d, clip_max)
    if sclip_sigma is not None:
        mask *= _make_mask_clip_sig(data_d, sclip_sigma, sclip_iters)
    if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
        mask *= _make_mask_clip_ccl(
            mask != 0, ccl_lcount, ccl_pcount, ccl_lratio)

    _apply_mask(data_d, data_e, data_m, mask)

    if minify:
        data_d, data_e, data_m, start = _minify_data(
            data_d, data_e, data_m, mask)
        offset += start

    if nanpad:
        data_d, data_e, data_m = _pad_data(
            data_d, data_e, data_m, nanpad, np.nan, np.nan, 0)
        offset -= nanpad

    _save_data(
        file_d, data_d, header_d,
        file_e, data_e, header_e,
        file_m, data_m, header_m,
        offset, dtype, velocity_rest)
