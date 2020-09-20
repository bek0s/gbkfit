
import numpy as np

import gbkfit.math


def fft_size(size):
    mul_size = 16
    new_size = gbkfit.math.roundu_po2(size)
    if new_size > mul_size:
        new_size = gbkfit.math.roundu_multiple(size, mul_size)
    return int(new_size)


class DCube:

    def __init__(self, size, step, cval, rota, scale, psf, lsf, dtype):

        # Central position in pixel space
        cpix = (
            size[0] / 2 - 0.5,
            size[1] / 2 - 0.5,
            size[2] / 2 - 0.5)

        # Low-res cube zero pixel center position
        zero = (
            cval[0] - cpix[0] * step[0],
            cval[1] - cpix[1] * step[1],
            cval[2] - cpix[2] * step[2])

        # High-res cube step
        step_hi = (
            step[0] / scale[0],
            step[1] / scale[1],
            step[2] / scale[2])

        # High-res psf/lsf size
        psf_size_hi = psf.size(step_hi[:2]) if psf else (1, 1)
        lsf_size_hi = lsf.size(step_hi[2]) if lsf else 1

        # High-res cube left edge size (left padding)
        edge_hi = (
            psf_size_hi[0] // 2,
            psf_size_hi[1] // 2,
            lsf_size_hi // 2)

        # High-res cube size
        size_hi = (
            size[0] * scale[0] + psf_size_hi[0] - 1,
            size[1] * scale[1] + psf_size_hi[1] - 1,
            size[2] * scale[2] + lsf_size_hi - 1)

        # If we need to perform fft-based convolution,
        # adjust sizes for optimal performance
        if psf or lsf:
            size_hi = (
                fft_size(size_hi[0]),
                fft_size(size_hi[1]),
                fft_size(size_hi[2]))

        # High-res cube zero pixel center position
        zero_hi = (
            zero[0] - step[0] / 2 - (edge_hi[0] - 0.5) * step_hi[0],
            zero[1] - step[1] / 2 - (edge_hi[1] - 0.5) * step_hi[1],
            zero[2] - step[2] / 2 - (edge_hi[2] - 0.5) * step_hi[2])

        # Create the psf/lsf images.
        # Always assume an fft-based convolution.
        # If they have an even size, they must be offset by -1 pixel.
        # Their centers must be rolled at their first pixel.
        psf_hi = np.zeros(size_hi[:2][::-1], dtype)
        psf_hi[0, 0] = 1
        if psf:
            psf_hi_offset = size_hi[0] % 2 - 1, size_hi[1] % 2 - 1
            psf_hi[:] = psf.asarray(step_hi[:2], size_hi[:2], psf_hi_offset)
            psf_hi[:] = np.roll(psf_hi, -size_hi[0] // 2 + 1, axis=1)
            psf_hi[:] = np.roll(psf_hi, -size_hi[1] // 2 + 1, axis=0)
        lsf_hi = np.zeros(size_hi[2], dtype)
        lsf_hi[0] = 1
        if lsf:
            lsf_hi_offset = size_hi[2] % 2 - 1
            lsf_hi[:] = lsf.asarray(step_hi[2], size_hi[2], lsf_hi_offset)
            lsf_hi[:] = np.roll(lsf_hi, -size_hi[2] // 2 + 1)

        self._size_lo = size
        self._step_lo = step
        self._zero_lo = zero
        self._edge_hi = edge_hi
        self._size_hi = size_hi
        self._step_hi = step_hi
        self._zero_hi = zero_hi
        self._cpix = cpix
        self._cval = cval
        self._rota = rota
        self._scale = scale
        self._dcube_lo = None
        self._dcube_hi = None
        self._dmask_hi = None
        self._dcube_hi_fft = None
        self._psf3d_hi_fft = None
        self._psf = psf
        self._lsf = lsf
        self._psf_hi = psf_hi
        self._lsf_hi = lsf_hi
        self._psf3d_hi = None
        self._dtype = dtype
        self._dcube = None
        self._driver = None

    def size(self):
        return self._size_lo

    def step(self):
        return self._step_lo

    def zero(self):
        return self._zero_lo

    def data(self):
        return self._dcube_lo

    def scratch_size(self):
        return self._size_hi

    def scratch_step(self):
        return self._step_hi

    def scratch_zero(self):
        return self._zero_hi

    def scratch_data(self):
        return self._dcube_hi

    def cpix(self):
        return self._cpix

    def cval(self):
        return self._cval

    def rota(self):
        return self._rota

    def scale(self):
        return self._scale

    def psf(self):
        return self._psf

    def lsf(self):
        return self._lsf

    def dtype(self):
        return self._dtype

    def prepare(self, driver):
        self._driver = driver
        size_lo = self._size_lo
        size_hi = self._size_hi
        edge_hi = self._edge_hi
        scale = self._scale
        dtype = self._dtype
        # Allocate the low- and high-resolution data cubes.
        # If they have the same size, just create one and have the
        # latter point to the former. This can happen when there is
        # no super sampling, no psf, and no lsf.
        self._dcube_lo = driver.mem_alloc_d(size_lo[::-1], dtype)
        self._dcube_hi = driver.mem_alloc_d(size_hi[::-1], dtype) \
            if size_lo != size_hi else self._dcube_lo
        # Allocate buffers for the fft-transformed 3d psf and data cube
        if self._psf or self._lsf:
            self._psf3d_hi = self._psf_hi * self._lsf_hi[:, None, None]
            self._psf3d_hi = driver.mem_copy_h2d(self._psf3d_hi)
            size_hi_fft = 2 * size_hi[2] * size_hi[1] * (size_hi[0] // 2 + 1)
            self._dcube_hi_fft = driver.mem_alloc_d(size_hi_fft, dtype)
            self._psf3d_hi_fft = driver.mem_alloc_d(size_hi_fft, dtype)
        # The psf convolution also affects pixels outside the galaxy model
        # Allocate a spatial mask for all the pixels of the galaxy model
        if self._psf:
            self._dmask_hi = driver.mem_alloc_d(size_hi[:2][::-1], dtype)
            driver.mem_fill(self._dmask_hi, 0)
        # Create and prepare dcube backend
        self._dcube = driver.make_dmodel_dcube(dtype)

    def evaluate(self, out_extra):

        if self._psf or self._lsf:
            self._dcube.convolve(
                self._size_hi,
                self._dcube_hi, self._dcube_hi_fft,
                self._psf3d_hi, self._psf3d_hi_fft)

        if self._dcube_lo is not self._dcube_hi:
            self._dcube.downscale(
                self._scale, self._edge_hi, self._size_hi, self._size_lo,
                self._dcube_hi, self._dcube_lo)

        if out_extra is not None:
            out_extra.update(
                dcube_lo=self._driver.mem_copy_d2h(self._dcube_lo),
                dcube_hi=self._driver.mem_copy_d2h(self._dcube_hi))
            if self._psf:
                out_extra.update(
                    psf_lo=self._psf.asarray(self._step_lo[:2]),
                    psf_hi=self._psf.asarray(self._step_hi[:2]),
                    psf_hi_fft=self._psf_hi.copy())
            if self._lsf:
                out_extra.update(
                    lsf_lo=self._lsf.asarray(self._step_lo[2]),
                    lsf_hi=self._lsf.asarray(self._step_hi[2]),
                    lsf_hi_fft=self._lsf_hi.copy())
            if self._psf or self._lsf:
                out_extra.update(
                    psf3d_hi_fft=self._driver.mem_copy_d2h(self._psf3d_hi))
