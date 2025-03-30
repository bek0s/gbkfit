
import logging
from typing import Any

import numpy as np

import gbkfit.math
from gbkfit.driver import Driver
from gbkfit.psflsf import LSF, PSF
from gbkfit.psflsf.lsfs import LSFPoint
from gbkfit.psflsf.psfs import PSFPoint


_log = logging.getLogger(__name__)


class DCube:

    def __init__(
            self,
            size: tuple[int, int, int],
            step: tuple[float, float, float],
            rpix: tuple[float, float, float],
            rval: tuple[float, float, float],
            rota: float,
            scale: tuple[int, int, int],
            psf: PSF | None,
            lsf: LSF | None,
            smooth_weights: bool,
            mask_cutoff: float | None,
            mask_apply: bool,
            dtype: type[np.float16] | type[np.float32] | type[np.float64]
    ):
        if mask_apply and mask_cutoff is None:
            _log.warning(
                "mask_apply is set to True, but mask_cutoff is not provided; "
                "no mask will be generated or applied to the model data")
            mask_apply = False
        if smooth_weights and not (psf or lsf):
            _log.warning(
                "smooth_weights is set to True, but neither PSF nor LSF is "
                "provided; if weights exist, they will not be smoothed")
            smooth_weights = False

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
        self._has_weights = None
        self._smooth_weights = smooth_weights
        self._mask_cutoff = mask_cutoff
        self._mask_apply = mask_apply
        self._dtype = dtype
        self._driver = None
        self._backend_fft = None
        self._backend_dmodel = None

    def size(self) -> tuple[int, int, int]:
        return self._size_lo

    def step(self) -> tuple[float, float, float]:
        return self._step_lo

    def zero(self) -> tuple[float, float, float]:
        return self._zero_lo

    def rota(self) -> float:
        return self._rota

    def scale(self) -> tuple[int, int, int]:
        return self._scale

    def dcube(self) -> Any:
        return self._dcube_lo

    def wcube(self) -> Any:
        return self._wcube_lo

    def mcube(self) -> Any:
        return self._mcube_lo

    def scratch_size(self) -> tuple[int, int, int]:
        return self._size_hi

    def scratch_step(self) -> tuple[float, float, float]:
        return self._step_hi

    def scratch_zero(self) -> tuple[float, float, float]:
        return self._zero_hi

    def scratch_edge(self) -> tuple[int, int, int]:
        return self._edge_hi

    def scratch_dcube(self) -> Any:
        return self._dcube_hi

    def scratch_wcube(self) -> Any:
        return self._wcube_hi

    def psf(self) -> PSF | None:
        return self._psf

    def lsf(self) -> LSF | None:
        return self._lsf

    def dtype(self) -> type[np.float16] | type[np.float32] | type[np.float64]:
        return self._dtype

    def prepare(self, driver: Driver, has_weights: bool) -> None:

        # Convenience variables
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

        # Convenience variables
        spat_step_hi = (step_hi[0], step_hi[1])
        spec_step_hi = step_hi[2]

        # Calculate the minimum size required to store the psf/lsf
        # In the absense of a psf/lsf we use a size of 1. This is done
        # to facilitate some calculations when only onn of psf or lsf
        # is present. When both are absent, no psf/lsf will be created.
        minimum_psf_size_hi = psf.size(spat_step_hi) if psf else (1, 1)
        minimum_lsf_size_hi = lsf.size(spec_step_hi) if lsf else 1

        # If psf/lsf is provided, we convolve the model cube with it.
        # We always perform fft-based convolution because it is faster.
        # Fft-based convolution requires padding on the model cube.
        edge_hi = [0, 0, 0]
        if psf or lsf:
            # Get convolution shape and left offset due to padding
            size_hi, edge_hi = backend_fft.fft_convolution_shape(
                size_hi, minimum_psf_size_hi + (minimum_lsf_size_hi,))
            print(size_hi)

        # High-res cube zero pixel center position
        zero_hi = (
            zero_lo[0] - step_lo[0] / 2 - (edge_hi[0] - 0.5) * step_hi[0],
            zero_lo[1] - step_lo[1] / 2 - (edge_hi[1] - 0.5) * step_hi[1],
            zero_lo[2] - step_lo[2] / 2 - (edge_hi[2] - 0.5) * step_hi[2])

        # The shape of the arrays created below are the reversed size
        shape_lo = size_lo[::-1]
        shape_hi = size_hi[::-1]

        # Convenience variables
        spat_size_hi = (size_hi[0], size_hi[1])
        spec_size_hi = size_hi[2]

        # Create high-res psf/lsf cube, if psf/lsf was provided
        # The psf cube will be used for the fft-based convolution
        if psf or lsf:
            # Create separate high-res psf/lsf images
            # If they have an even size, they must be offset by -1 pixel
            # This is because in most cases the psf/lsf have a central peak
            offset_hi = gbkfit.math.is_odd(size_hi) - 1
            psf_offset_hi = offset_hi[:2]
            lsf_offset_hi = offset_hi[2]
            psf_args = (spat_step_hi, spat_size_hi, psf_offset_hi)
            lsf_args = (spec_step_hi, spec_size_hi, lsf_offset_hi)
            psf_hi = psf.asarray(*psf_args) if psf \
                else PSFPoint().asarray(*psf_args)
            lsf_hi = lsf.asarray(*lsf_args) if lsf \
                else LSFPoint().asarray(*lsf_args)
            # Build high-res psf/lsf cube
            self._pcube_hi = (psf_hi * lsf_hi[:, None, None]).astype(dtype)
            # Roll the centre of the psf cube to (0, 0, 0)
            self._pcube_hi = backend_fft.fft_convolution_shift(self._pcube_hi)
            # Transfer the psf cube to device memory
            self._pcube_hi = driver.mem_copy_h2d(self._pcube_hi)

        # Create low- and high-res data and weight cubes.
        # If the low- and high-res versions have the same size,
        # just create one and have the latter point to the former.
        # This can happen when there is no supersampling or padding.
        self._dcube_lo = driver.mem_alloc_d(shape_lo, dtype)
        self._dcube_hi = self._dcube_lo
        driver.mem_fill(self._dcube_lo, 0)
        if size_lo != size_hi:
            self._dcube_hi = driver.mem_alloc_d(shape_hi, dtype)
            driver.mem_fill(self._dcube_hi, 0)
        if has_weights:
            self._wcube_lo = driver.mem_alloc_d(shape_lo, dtype)
            self._wcube_hi = self._wcube_lo
            driver.mem_fill(self._wcube_lo, 1)
            if size_lo != size_hi:
                self._wcube_hi = driver.mem_alloc_d(shape_hi, dtype)
                driver.mem_fill(self._wcube_hi, 1)

        # Create low-res mask cube if requested.
        # There is no high-res mask cube because masking is always done
        # on the low-res cubes.
        if self._mask_cutoff:
            self._mcube_lo = driver.mem_alloc_d(shape_lo, dtype)
            driver.mem_fill(self._mcube_lo, 1)

        self._size_hi = size_hi
        self._step_hi = step_hi
        self._zero_hi = zero_hi
        self._edge_hi = edge_hi
        self._has_weights = has_weights
        self._driver = driver
        self._backend_fft = backend_fft
        self._backend_dmodel = driver.backends().dmodel(dtype)

    def evaluate(self, out_extra: dict[str, Any] | None) -> None:

        # Convenience variables
        step_lo = self._step_lo
        step_hi = self._step_hi
        spat_step_lo = (step_lo[0], step_lo[1])
        spec_step_lo = step_lo[2]
        spat_step_hi = (step_hi[0], step_hi[1])
        spec_step_hi = step_hi[2]
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
        has_weights = self._wcube_lo is not None
        mask_cutoff = self._mask_cutoff
        mask_apply = self._mask_apply
        driver = self._driver
        backend_fft = self._backend_fft
        backend_dmodel = self._backend_dmodel

        # Perform fft-based convolution
        if psf or lsf:
            backend_fft.fft_convolve_cached(dcube_hi, pcube_hi)
            if has_weights:
                backend_fft.fft_convolve_cached(wcube_hi, pcube_hi)

        # Perform downscaling
        if dcube_lo is not dcube_hi:
            backend_dmodel.dcube_downscale(
                scale, edge_hi, dcube_hi, dcube_lo)
            if has_weights and self._smooth_weights:
                backend_dmodel.dcube_downscale(
                    scale, edge_hi, wcube_hi, wcube_lo)

        # Apply masking.
        # Checked if mask_create or mask_apply are True in __init__()
        if mask_cutoff is not None:
            backend_dmodel.dcube_mask(
                mask_cutoff, mask_apply, mcube_lo, dcube_lo, wcube_lo)

        # Output extra information
        if out_extra is not None:
            out_extra.update(
                dcube_lo=driver.mem_copy_d2h(dcube_lo),
                dcube_hi=driver.mem_copy_d2h(dcube_hi))
            if mask_cutoff:
                out_extra.update(
                    mcube_lo=driver.mem_copy_d2h(mcube_lo))
            if has_weights:
                out_extra.update(
                    wcube_lo=driver.mem_copy_d2h(wcube_lo),
                    wcube_hi=driver.mem_copy_d2h(wcube_hi))
            if psf:
                out_extra.update(
                    psf_lo=psf.asarray(spat_step_lo),
                    psf_hi=psf.asarray(spat_step_hi))
            if lsf:
                out_extra.update(
                    lsf_lo=lsf.asarray(spec_step_lo),
                    lsf_hi=lsf.asarray(spec_step_hi))
            if psf or lsf:
                out_extra.update(
                    pcube_hi_fft=driver.mem_copy_d2h(pcube_hi))
