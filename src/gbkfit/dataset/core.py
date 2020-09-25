
import abc
import copy
import logging

import astropy.io.fits as fits
import astropy.wcs
import numpy as np

from gbkfit.utils import miscutils, parseutils
from . import _detail


log = logging.getLogger(__name__)


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
            overwrite=False):
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
            info['data'] = filename_d
            dump_fits(filename_d, dat, step, rpix, rval, rota, overwrite)
        if filename_m and msk is not None:
            info['mask'] = filename_m
            dump_fits(filename_m, msk, step, rpix, rval, rota, overwrite)
        if filename_e and err is not None:
            info['error'] = filename_e
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
        finite_mask = np.ones_like(data)
        finite_mask *= np.isfinite(data)
        finite_mask *= np.isfinite(mask)
        finite_mask *= np.isfinite(error)
        data[finite_mask == 0] = np.nan
        mask[finite_mask == 0] = 0
        mask[finite_mask != 0] = 1
        error[finite_mask == 0] = np.nan
        zero = (
            rval[0] - rpix[0] * step[0],
            rval[1] - rpix[1] * step[1],
            rval[2] - rpix[2] * step[2])
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


class Dataset(parseutils.TypedParserSupport, abc.ABC):

    def __init__(self, data):
        desc = parseutils.make_typed_desc(self.__class__, 'dataset')
        # At least one data item must be defined
        if not data:
            raise RuntimeError(f"{desc} contains no data items")
        # All data items must have the same dtype
        _detail.ensure_same_attrib_value(data, 'dtype', desc)
        # We need to copy the data to ensure they are kept intact
        self._data = copy.deepcopy(data)

    def __contains__(self, item):
        return item in self._data

    def __getitem__(self, item):
        return self._data[item]

    def __iter__(self):
        return iter(self._data)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def get(self, item):
        return self._data.get(item)

    @property
    def npixs(self):
        return tuple(data.npix for data in self.values())

    @property
    def sizes(self):
        return tuple(data.size for data in self.values())

    @property
    def steps(self):
        return tuple(data.step for data in self.values())

    @property
    def zeros(self):
        return tuple(data.zero for data in self.values())

    @property
    def rpixs(self):
        return tuple(data.rpix for data in self.values())

    @property
    def rvals(self):
        return tuple(data.rval for data in self.values())

    @property
    def rotas(self):
        return tuple(data.rota for data in self.values())

    @property
    def dtype(self):
        return next(iter(self.values())).dtype()


data_parser = parseutils.BasicParser(Data)
dataset_parser = parseutils.TypedParser(Dataset)
