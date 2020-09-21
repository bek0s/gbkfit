
import abc
import copy
import logging

import astropy.io.fits as fits
import numpy as np

from gbkfit.utils import miscutils, parseutils
from . import _detail


log = logging.getLogger(__name__)


class Data:

    @classmethod
    def load(cls, info, step=None, rpix=None, rval=None, rota=None):
        desc = make_data_desc(cls)
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        data_d = fits.getdata(opts['data'])
        data_m = fits.getdata(opts['mask']) if 'mask' in info else None
        data_e = fits.getdata(opts['error']) if 'error' in info else None
        # Local information has higher priority than global
        step = opts.get('step', step)
        rpix = opts.get('rpix', rpix)
        rval = opts.get('rval', rval)
        rota = opts.get('rota', rota)
        # TODO: If no information is provided yet, use fits header
        # Build class arguments dict
        opts.update(dict(
            data=data_d,
            mask=data_m,
            error=data_e,
            step=step,
            rpix=rpix,
            rval=rval,
            rota=rota))
        return cls(**opts)

    def dump(self, prefix=''):
        file_d = f'{prefix}d.fits'
        file_m = f'{prefix}m.fits'
        file_e = f'{prefix}e.fits'
        info = dict(
            data=file_d,
            mask=file_m,
            error=file_e,
            step=self.step(),
            rpix=self.rpix(),
            rval=self.rval(),
            rota=self.rota())
        fits.writeto(file_d, self.data(), overwrite=True)
        fits.writeto(file_m, self.mask(), overwrite=True)
        fits.writeto(file_e, self.error(), overwrite=True)
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
                f"({data.dim} != {len(step)})")
        if data.ndim != len(rpix):
            raise RuntimeError(
                f"data dimensionality and rpix length are incompatible "
                f"({data.dim} != {len(rpix)})")
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
        desc = make_dataset_desc(self.__class__)
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


def make_data_desc(cls):
    return f'data (class={cls.__qualname__})'


def make_dataset_desc(cls):
    return f'{cls.type()} dataset (class={cls.__qualname__})'
