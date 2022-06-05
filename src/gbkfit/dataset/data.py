
import os.path

import astropy.io.fits as fits
import astropy.wcs
import numpy as np

from gbkfit.utils import miscutils, parseutils


def _make_filename(filename, dump_full_path):
    return filename if dump_full_path else os.path.basename(filename)


def load_fits(filename):
    data = fits.getdata(filename)
    header = fits.getheader(filename)
    wcs = astropy.wcs.WCS(header)
    return data, header, wcs


def dump_fits(
        filename, data, cdelt=None, crpix=None, crval=None, crota=None,
        overwrite=False):
    wcs = astropy.wcs.WCS(naxis=data.ndim, relax=False)
    if cdelt is not None:
        wcs.wcs.cdelt = cdelt
    if crpix is not None:
        wcs.wcs.crpix = crpix
    if crval is not None:
        wcs.wcs.crval = crval
    if crota is not None:
        crota = np.radians(crota)
        wcs.wcs.pc = np.identity(data.ndim)
        wcs.wcs.pc[0][0] = +np.cos(crota)
        wcs.wcs.pc[0][1] = -np.sin(crota)
        wcs.wcs.pc[1][0] = +np.sin(crota)
        wcs.wcs.pc[1][1] = +np.cos(crota)
    fits.writeto(
        filename, data, header=wcs.to_header(),
        output_verify='exception', overwrite=overwrite, checksum=True)


class Data:

    @classmethod
    def load(cls, info, step=None, rpix=None, rval=None, rota=None):
        desc = parseutils.make_basic_desc(cls, 'data')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        data_d, header_d, wcs_d = None, None, None
        data_m, header_m, wcs_m = None, None, None
        data_e, header_e, wcs_e = None, None, None
        if 'data' in opts:
            data_d, header_d, wcs_d = load_fits(opts['data'])
        if 'mask' in opts:
            data_m, header_m, wcs_m = load_fits(opts['mask'])
        if 'error' in opts:
            data_e, header_e, wcs_e = load_fits(opts['error'])
        # Local information has higher priority than global
        step = opts.get('step', step)
        rpix = opts.get('rpix', rpix)
        rval = opts.get('rval', rval)
        # If no information is provided, use fits header
        if step is None:
            step = wcs_d.wcs.cdelt
        if rpix is None:
            rpix = wcs_d.wcs.crpix
        if rval is None:
            rval = wcs_d.wcs.crval
        # todo: deal with rotation (PC Matrix and CROTA (deprecated))
        # Build class arguments dict
        opts.update(dict(
            data=data_d,
            mask=data_m,
            error=data_e,
            step=step,
            rpix=rpix,
            rval=rval,
            rota=0))
        return cls(**opts)

    def dump(
            self, filename_d=None, filename_m=None, filename_e=None,
            dump_full_path=True, overwrite=False):
        dat = self.data()
        msk = self.mask()
        err = self.error()
        step = self.step()
        rpix = self.rpix()
        rval = self.rval()
        rota = self.rota()
        info = dict(
            step=step,
            rpix=rpix,
            rval=rval,
            rota=rota)
        if filename_d and dat is not None:
            info['data'] = _make_filename(filename_d, dump_full_path)
            dump_fits(filename_d, dat, step, rpix, rval, rota, overwrite)
        if filename_m and msk is not None:
            info['mask'] = _make_filename(filename_m, dump_full_path)
            dump_fits(filename_m, msk, step, rpix, rval, rota, overwrite)
        if filename_e and err is not None:
            info['error'] = _make_filename(filename_e, dump_full_path)
            dump_fits(filename_e, err, step, rpix, rval, rota, overwrite)
        return info

    def __init__(
            self, data, mask=None, error=None,
            step=None, rpix=None, rval=None, rota=0):
        if mask is None:
            mask = np.ones_like(data)
        if error is None:
            error = np.ones_like(data)
        if step is None:
            step = (1,) * data.ndim
        if rpix is None:
            rpix = tuple((np.array(data.shape[::-1]) / 2 - 0.5).tolist())
        if rval is None:
            rval = (0,) * data.ndim
        data = miscutils.to_native_byteorder(data)
        mask = miscutils.to_native_byteorder(mask)
        error = miscutils.to_native_byteorder(error)
        if data.shape != mask.shape:
            raise RuntimeError(
                f"data and mask have incompatible shapes "
                f"({data.shape} != {mask.shape})")
        if data.shape != error.shape:
            raise RuntimeError(
                f"data and error have incompatible shapes "
                f"({data.shape} != {error.shape})")
        if data.ndim != len(step):
            raise RuntimeError(
                f"data dimensionality and step length are incompatible "
                f"({data.ndim} != {len(step)})")
        if data.ndim != len(rpix):
            raise RuntimeError(
                f"data dimensionality and rpix length are incompatible "
                f"({data.ndim} != {len(rpix)})")
        if data.ndim != len(rval):
            raise RuntimeError(
                f"data dimensionality and rval length are incompatible "
                f"({data.dim} != {len(rval)})")
        total_mask = np.ones_like(data)
        total_mask *= np.isfinite(data)
        total_mask *= np.isfinite(mask)
        total_mask *= np.isfinite(error)
        total_mask *= mask != 0
        data[total_mask == 0] = np.nan
        mask[total_mask == 0] = 0
        mask[total_mask != 0] = 1
        error[total_mask == 0] = np.nan
        rpix = (np.array(data.shape[::-1]) / 2 - 0.5).tolist()
        zero = (np.array(rval) - np.array(rpix) * np.array(step)).tolist()
        self._data = data.copy()
        self._mask = mask.copy().astype(data.dtype)
        self._error = error.copy().astype(data.dtype)
        self._step = tuple(step)
        self._zero = tuple(zero)
        self._rpix = tuple(rpix)
        self._rval = tuple(rval)
        self._rota = rota

    def ndim(self):
        return self._data.ndim

    def npix(self):
        return self._data.size

    def size(self):
        return self._data.shape[::-1]

    def step(self):
        return self._step

    def zero(self):
        return self._zero

    def rpix(self):
        return self._rpix

    def rval(self):
        return self._rval

    def rota(self):
        return self._rota

    def data(self):
        return self._data

    def mask(self):
        return self._mask

    def error(self):
        return self._error

    def dtype(self):
        return self._data.dtype


data_parser = parseutils.BasicParser(Data)
