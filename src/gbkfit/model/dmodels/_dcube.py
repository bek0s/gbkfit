
import numpy as np

import gbkfit.math


def _pix2world(pos, step, rpix, rval, rota):
    pos = [
        pos[0] - rpix[0],
        pos[1] - rpix[1],
        pos[2] - rpix[2]]
    pos[0], pos[1] = gbkfit.math.transform_lh_rotate_z(pos[0], pos[1], rota)
    pos = [
        (pos[0] - rpix[0]) * step[0] + rval[0],
        pos[1] * step[1] + rval[1],
        pos[2] * step[2] + rval[2]]
    return pos


def _fft_size(size):
    mul_size = 16
    new_size = gbkfit.math.roundu_po2(size)
    if new_size > mul_size:
        new_size = gbkfit.math.roundu_multiple(size, mul_size)
    return int(new_size)


class DCube:

    def __init__(
            self, size, step, rpix, rval, rota, scale, psf, lsf,
            weights, weights_conv, dtype):

        # Low-res cube zero pixel center position
        zero = (
            rval[0] - rpix[0] * step[0],
            rval[1] - rpix[1] * step[1],
            rval[2] - rpix[2] * step[2])

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
                _fft_size(size_hi[0]),
                _fft_size(size_hi[1]),
                _fft_size(size_hi[2]))

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
        self._rota = rota
        self._scale = scale
        self._dcube_lo = None
        self._dcube_hi = None
        self._wcube_lo = None
        self._wcube_hi = None
        self._mcube_lo = None
        self._dcube_hi_fft = None
        self._wcube_hi_fft = None
        self._psf3d_hi_fft = None
        self._psf = psf
        self._lsf = lsf
        self._psf_hi = psf_hi
        self._lsf_hi = lsf_hi
        self._psf3d_hi = None
        self._weights = weights
        self._weights_conv = weights_conv
        self._dtype = dtype
        self._dcube = None
        self._driver = None

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

    def weights(self):
        return self._weights

    def weights_conv(self):
        return self._weights_conv

    def dtype(self):
        return self._dtype

    def prepare(self, driver):
        self._driver = driver
        size_lo = self._size_lo
        size_hi = self._size_hi
        dtype = self._dtype
        # Allocate the low- and high-resolution data cubes.
        # If they have the same size, just create one and have the
        # latter point to the former. This can happen when there is
        # no super sampling, no psf, and no lsf.
        self._dcube_lo = driver.mem_alloc_d(size_lo[::-1], dtype)
        self._dcube_hi = driver.mem_alloc_d(size_hi[::-1], dtype) \
            if size_lo != size_hi else self._dcube_lo
        if self._weights:
            self._wcube_lo = driver.mem_alloc_d(size_lo[::-1], dtype)
            self._wcube_hi = driver.mem_alloc_d(size_hi[::-1], dtype) \
                if size_lo != size_hi else self._wcube_lo
        # Allocate buffers for the fft-transformed 3d psf and data cube
        if self._psf or self._lsf:
            self._psf3d_hi = self._psf_hi * self._lsf_hi[:, None, None]
            self._psf3d_hi = driver.mem_copy_h2d(self._psf3d_hi)
            size_hi_fft = 2 * size_hi[2] * size_hi[1] * (size_hi[0] // 2 + 1)
            self._dcube_hi_fft = driver.mem_alloc_d(size_hi_fft, dtype)
            self._psf3d_hi_fft = driver.mem_alloc_d(size_hi_fft, dtype)
            if self._weights and self._weights_conv:
                self._wcube_hi_fft = driver.mem_alloc_d(size_hi_fft, dtype)
        # The psf convolution affects pixels outside the galaxy model
        # This can create unwanted noise in the model
        # Hence, we use a spatial mask to mark all the good pixels
        self._mcube_lo = driver.mem_alloc_d(size_lo[::-1], dtype)
        # Create and prepare dcube backend
        self._dcube = driver.make_dmodel_dcube(dtype)

    def evaluate(self, out_extra):

        if self._psf or self._lsf:
            self._dcube.convolve(
                self._size_hi,
                self._dcube_hi, self._dcube_hi_fft,
                self._wcube_hi, self._wcube_hi_fft,
                self._psf3d_hi, self._psf3d_hi_fft)

        if self._dcube_lo is not self._dcube_hi:
            self._dcube.downscale(
                self._scale, self._edge_hi, self._size_hi, self._size_lo,
                self._dcube_hi, self._dcube_lo)
            if self._weights:
                self._dcube.downscale(
                    self._scale, self._edge_hi, self._size_hi, self._size_lo,
                    self._wcube_hi, self._wcube_lo)

        self._dcube.make_mask(
            True, True, 1e-6, self._size_lo, self._dcube_lo, self._mcube_lo)

        if out_extra is not None:
            out_extra.update(
                dcube_lo=self._driver.mem_copy_d2h(self._dcube_lo),
                dcube_hi=self._driver.mem_copy_d2h(self._dcube_hi),
                mcube_lo=self._driver.mem_copy_d2h(self._mcube_lo))
            if self._weights:
                out_extra.update(
                    wcube_lo=self._driver.mem_copy_d2h(self._wcube_lo),
                    wcube_hi=self._driver.mem_copy_d2h(self._wcube_hi))
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
