
import logging

import gbkfit.math

from gbkfit.psflsf.lsfs import LSFPoint
from gbkfit.psflsf.psfs import PSFPoint


_log = logging.getLogger(__name__)


class DCube:

    def __init__(
            self, size, step, rpix, rval, rota, scale, psf, lsf,
            weight, mask_cutoff, mask_create, mask_apply, dtype):

        if mask_cutoff is None and mask_apply:
            raise RuntimeError(
                "masking is disabled (i.e., mask_cutoff is None), "
                "mask_apply can't be True")

        if mask_cutoff is not None and not (mask_create or mask_apply):
            raise RuntimeError(
                "masking is enabled (i.e., mask_cutoff >= 0), "
                "mask_create or mask_apply must be True")

        if mask_cutoff is None and mask_create:
            _log.warning(
                "masking is disabled (i.e., mask_cutoff is None) but "
                "mask creation is enabled (i.e., mask_create is True); "
                "this will result in an unused mask; "
                "are you sure you have compute cycles and memory to waste?")

        # Low-res cube zero pixel center position
        zero = (
            rval[0] - rpix[0] * step[0],
            rval[1] - rpix[1] * step[1],
            rval[2] - rpix[2] * step[2])

        self._size_lo = size
        self._step_lo = step
        self._zero_lo = zero
        self._size_hi = None
        self._step_hi = None
        self._zero_hi = None
        self._edge_hi = None
        self._rota = rota
        self._scale = scale
        self._dcube_lo = None
        self._dcube_hi = None
        self._wcube_lo = None
        self._wcube_hi = None
        self._mcube_lo = None
        self._pcube_hi = None
        self._psf = psf
        self._lsf = lsf
        self._weight = weight
        self._weighting = weight != 1
        self._mask_cutoff = mask_cutoff
        self._mask_create = mask_create
        self._mask_apply = mask_apply
        self._dtype = dtype
        self._driver = None
        self._backend_fft = None
        self._backend_dmodel = None

    def size(self):
        return self._size_lo

    def step(self):
        return self._step_lo

    def zero(self):
        return self._zero_lo

    def rota(self):
        return self._rota

    def dcube(self):
        return self._dcube_lo

    def wcube(self):
        return self._wcube_lo

    def mcube(self):
        return self._mcube_lo

    def scratch_size(self):
        return self._size_hi

    def scratch_step(self):
        return self._step_hi

    def scratch_zero(self):
        return self._zero_hi

    def scratch_dcube(self):
        return self._dcube_hi

    def scratch_wcube(self):
        return self._wcube_hi

    def scale(self):
        return self._scale

    def psf(self):
        return self._psf

    def lsf(self):
        return self._lsf

    def dtype(self):
        return self._dtype

    def prepare(self, driver, enable_weighting):

        # Shortcuts
        size_lo = self.size()
        step_lo = self.step()
        zero_lo = self.zero()
        scale = self.scale()
        psf = self.psf()
        lsf = self.lsf()
        dtype = self.dtype()

        # Use the native fft library
        backend_fft = driver.backends().fft(dtype)

        # High-res cube size (before taking padding into account)
        size_hi = (
            size_lo[0] * scale[0],
            size_lo[1] * scale[1],
            size_lo[2] * scale[2])

        # High-res cube step
        step_hi = (
            step_lo[0] / scale[0],
            step_lo[1] / scale[1],
            step_lo[2] / scale[2])

        # High-res psf/lsf size
        psf_size_hi = psf.size(step_hi[:2]) if psf else (1, 1)
        lsf_size_hi = lsf.size(step_hi[2]) if lsf else 1

        # If psf/lsf was provided, we need to convolve the model cube with it.
        # We always perform an fft-based convolution because it is faster.
        # Fft-based convolution requires some padding on the model cube.
        edge_hi = [0, 0, 0]
        if psf or lsf:
            # Get convolution shape and left offset due to padding
            size_hi, edge_hi = backend_fft.fft_convolution_shape(
                size_hi, psf_size_hi + (lsf_size_hi,))

        # High-res cube zero pixel center position
        zero_hi = (
            zero_lo[0] - step_lo[0] / 2 - (edge_hi[0] - 0.5) * step_hi[0],
            zero_lo[1] - step_lo[1] / 2 - (edge_hi[1] - 0.5) * step_hi[1],
            zero_lo[2] - step_lo[2] / 2 - (edge_hi[2] - 0.5) * step_hi[2])

        # Create high-res psf/lsf images if psf/lsf was provided
        # If they have an even size, they must be offset by -1 pixel
        # This is because in most cases the psf/lsf have a central peak
        offset = gbkfit.math.is_odd(size_hi) - 1
        psfargs = (step_hi[:2], size_hi[:2], offset[:2])
        lsfargs = (step_hi[2], size_hi[2], offset[2])
        psf_hi = psf.asarray(*psfargs) if psf else PSFPoint().asarray(*psfargs)
        lsf_hi = lsf.asarray(*lsfargs) if lsf else LSFPoint().asarray(*lsfargs)

        # Create high-res psf/lsf cube, if psf/lsf was provided
        # The psf cube will be used for the fft-based convolution
        if psf or lsf:
            # Build high-res psf cube (psf + lsf)
            self._pcube_hi = (psf_hi * lsf_hi[:, None, None]).astype(dtype)
            # Roll the centre of the psf cube to (0, 0, 0)
            self._pcube_hi = backend_fft.fft_convolution_shift(self._pcube_hi)
            # Transfer the psf cube to device memory
            self._pcube_hi = driver.mem_copy_h2d(self._pcube_hi)

        # Create low- and high-res data and weight cubes.
        # If the low- and high-res versions have the same size,
        # just create one and have the latter point to the former.
        # This can happen when there is no supersampling or padding.
        self._dcube_lo = driver.mem_alloc_d(size_lo[::-1], dtype)
        self._dcube_hi = driver.mem_alloc_d(size_hi[::-1], dtype) \
            if size_lo != size_hi else self._dcube_lo
        # Enable weighing if requested and not already enabled
        self._weighting = self._weighting or enable_weighting
        if self._weighting:
            self._wcube_lo = driver.mem_alloc_d(size_lo[::-1], dtype)
            self._wcube_hi = driver.mem_alloc_d(size_hi[::-1], dtype) \
                if size_lo != size_hi else self._wcube_lo
            driver.mem_fill(self._wcube_lo, 1)
            driver.mem_fill(self._wcube_hi, 1)

        # Create low-res mask cube if requested
        if self._mask_create:
            self._mcube_lo = driver.mem_alloc_d(size_lo[::-1], dtype)
            driver.mem_fill(self._mcube_lo, 1)

        self._size_hi = size_hi
        self._step_hi = step_hi
        self._zero_hi = zero_hi
        self._edge_hi = edge_hi
        self._driver = driver
        self._backend_fft = backend_fft
        self._backend_dmodel = driver.backends().dmodel(dtype)

    def evaluate(self, out_extra):

        # Convenience shortcuts
        step_lo = self._step_lo
        step_hi = self._step_hi
        edge_hi = self._edge_hi
        scale = self._scale
        psf = self._psf
        lsf = self._lsf
        dcube_lo = self._dcube_lo
        dcube_hi = self._dcube_hi
        wcube_lo = self._wcube_lo
        wcube_hi = self._wcube_hi
        mcube_lo = self._mcube_lo
        pcube_hi = self._pcube_hi
        weight = self._weight
        weighting = self._weighting
        mask_cutoff = self._mask_cutoff
        mask_create = self._mask_create
        mask_apply = self._mask_apply
        driver = self._driver
        backend_fft = self._backend_fft
        backend_dmodel = self._backend_dmodel

        # Perform fft-based convolution
        if psf or lsf:
            backend_fft.fft_convolve_cached(dcube_hi, pcube_hi)
            if weighting:
                backend_fft.fft_convolve_cached(wcube_hi, pcube_hi)

        # Downscale data and weight cubes
        if dcube_lo is not dcube_hi:
            backend_dmodel.dcube_downscale(scale, edge_hi, dcube_hi, dcube_lo)
            if weighting:
                backend_dmodel.dcube_downscale(
                    scale, edge_hi, wcube_hi, wcube_lo)

        # Apply intrinsic weighting
        if weighting and weight != 1:
            driver.math_mul(wcube_hi, weight, wcube_hi)

        # Apply masking.
        # Checked if mask_create or mask_apply are True in __init__()
        if mask_cutoff is not None:
            backend_dmodel.dcube_mask(
                mask_cutoff, mask_apply, dcube_lo, mcube_lo, wcube_lo)

        # Output extra information
        if out_extra is not None:
            out_extra.update(
                dcube_lo=driver.mem_copy_d2h(dcube_lo),
                dcube_hi=driver.mem_copy_d2h(dcube_hi))
            if mask_create:
                out_extra.update(
                    mcube_lo=driver.mem_copy_d2h(mcube_lo))
            if weighting:
                out_extra.update(
                    wcube_lo=driver.mem_copy_d2h(wcube_lo),
                    wcube_hi=driver.mem_copy_d2h(wcube_hi))
            if psf:
                out_extra.update(
                    psf_lo=psf.asarray(step_lo[:2]),
                    psf_hi=psf.asarray(step_hi[:2]))
            if lsf:
                out_extra.update(
                    lsf_lo=lsf.asarray(step_lo[2]),
                    lsf_hi=lsf.asarray(step_hi[2]))
            if psf or lsf:
                out_extra.update(
                    psf3d_hi_fft=driver.mem_copy_d2h(pcube_hi))
