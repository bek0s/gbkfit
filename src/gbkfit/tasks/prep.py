
import os
import re

import astropy.io.fits as fits
import astropy.stats as stats
import astropy.wcs
import numpy as np
import skimage.measure


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


def _read_fits(filename):
    """The data and the header of a file, without the axes of length 1."""
    data = fits.getdata(filename)
    header = fits.getheader(filename)
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
        offset, dtype):
    """
    Save the data, whose first pixel is the pixel at the given offset (on
    each numpy axis) of the data read.
    """
    basename = os.path.basename
    splitext = os.path.splitext
    header_d = _shift_axes(header_d, offset)
    if header_e is not None:
        header_e = _shift_axes(header_e, offset)
    if header_m is not None:
        header_m = _shift_axes(header_m, offset)
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
    mask = np.ones_like(data_d)
    mask *= np.isfinite(data_d)
    if data_e is not None:
        mask *= np.isfinite(data_e)
    if data_m is not None:
        mask *= np.isfinite(data_m)
    return mask


def _compare_nan_array(func, ary, threshold):
    out = ~np.isnan(ary)
    out[out] = func(ary[out], threshold)
    return out


def _make_mask_clip_min(data, min_value):
    return ~_compare_nan_array(np.less, data, min_value)


def _make_mask_clip_max(data, max_value):
    return ~_compare_nan_array(np.greater, data, max_value)


def _make_mask_clip_sig(data, sigma, maxiters, invert):
    mask = stats.sigma_clip(data, sigma=sigma, maxiters=maxiters).mask
    return mask if not invert else ~mask


def _make_mask_clip_ccl(data, lcount, pcount, lratio):
    labels = skimage.measure.label(data)
    props = skimage.measure.regionprops(labels)
    props.sort(key=lambda x: x.area, reverse=True)
    if lcount is not None:
        props = props[0:lcount]
    if pcount is not None:
        props = [p for p in props if p.area >= pcount]
    if lratio is not None:
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
        mask *= _make_mask_clip_sig(
            data_d, sclip_sigma, sclip_iters, False)
    if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
        mask *= _make_mask_clip_ccl(
            np.isfinite(data_d), ccl_lcount, ccl_pcount, ccl_lratio)

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
        sclip_sigma, sclip_iters, minify, nanpad, dtype):

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
        mask *= _make_mask_clip_sig(
            data_d, sclip_sigma, sclip_iters, False)
    if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
        mask *= _make_mask_clip_ccl(
            np.isfinite(data_d), ccl_lcount, ccl_pcount, ccl_lratio)

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


def prep_mmaps(
        orders, file_d, file_e, file_m,
        roi_spat, clip_min, clip_max, ccl_lcount, ccl_pcount, ccl_lratio,
        sclip_sigma, sclip_iters, minify, nanpad, dtype):

    nmmaps = len(file_d)
    if file_e is None:
        file_e = [None] * nmmaps
    if file_m is None:
        file_m = [None] * nmmaps

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
                data_d[i], sclip_sigma, sclip_iters, orders[i] == 0)
        if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
            mask *= _make_mask_clip_ccl(
                np.isfinite(data_d[i]), ccl_lcount, ccl_pcount, ccl_lratio)

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
        mask *= _make_mask_clip_sig(
            data_d, sclip_sigma, sclip_iters, False)
    if ccl_lcount is not None or ccl_pcount is not None or ccl_lratio:
        mask *= _make_mask_clip_ccl(
            np.isfinite(data_d), ccl_lcount, ccl_pcount, ccl_lratio)

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
