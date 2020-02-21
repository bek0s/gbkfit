
import numpy as np


class DCube:

    def __init__(self, size, step, cval, scale, psf, lsf, dtype):

        # Low-res cube zero pixel center position
        zero = (
            cval[0] - (size[0] / 2 - 0.5) * step[0],
            cval[1] - (size[1] / 2 - 0.5) * step[1],
            cval[2] - (size[2] / 2 - 0.5) * step[2])

        # High-res cube step
        step_hi = (
            step[0] / scale[0],
            step[1] / scale[1],
            step[2] / scale[2])

        # High-res psf/lsf size
        psf_size_hi = psf.size(step_hi[:2]) if psf else (1, 1)
        lsf_size_hi = lsf.size(step_hi[2]) if lsf else 1

        # High-res cube edge size (padding)
        edge_hi = (
            psf_size_hi[0] // 2,
            psf_size_hi[1] // 2,
            lsf_size_hi // 2)

        # High-res cube size
        size_hi = (
            (size[0] * scale[0] + psf_size_hi[0] - 1),
            (size[1] * scale[1] + psf_size_hi[1] - 1),
            (size[2] * scale[2] + lsf_size_hi - 1))

        # High-res cube zero pixel center position
        zero_hi = (
            zero[0] - step[0] / 2 - (edge_hi[0] - 0.5) * step_hi[0],
            zero[1] - step[1] / 2 - (edge_hi[1] - 0.5) * step_hi[1],
            zero[2] - step[2] / 2 - (edge_hi[2] - 0.5) * step_hi[2])

        # Create the psf/lsf images.
        # If they have an even size, they must be offset by -1 pixel.
        # Always assume an fft-based convolution.
        # Hence, their centers must be rolled at their first pixel.

        if psf:
            psf_hi_offset = size_hi[0] % 2 - 1, size_hi[1] % 2 - 1
            psf_hi = psf.asarray(step_hi[:2], size_hi[:2], psf_hi_offset)
            psf_hi = np.roll(psf_hi, -size_hi[0] // 2 + 1, axis=1)
            psf_hi = np.roll(psf_hi, -size_hi[1] // 2 + 1, axis=0)
        else:
            psf_hi = np.ones((1, 1), dtype)

        if lsf:
            lsf_hi_offset = size_hi[2] % 2 - 1
            lsf_hi = lsf.asarray(step_hi[2], size_hi[2], lsf_hi_offset)
            lsf_hi = np.roll(lsf_hi, -size_hi[2] // 2 + 1)
        else:
            lsf_hi = np.ones(1, dtype)

        self._cval = cval
        self._size_lo = size
        self._step_lo = step
        self._zero_lo = zero
        self._data_lo = None
        self._size_hi = size_hi
        self._step_hi = step_hi
        self._zero_hi = zero_hi
        self._data_hi = None
        self._edge_hi = edge_hi
        self._scale = scale
        self._psf = psf
        self._lsf = lsf
        self._psf_hi = psf_hi
        self._lsf_hi = lsf_hi
        self._dtype = dtype
        self._dcube = None
        self._driver = None

    def cval(self):
        return self._cval

    def size(self):
        return self._size_lo

    def step(self):
        return self._step_lo

    def zero(self):
        return self._zero_lo

    def data(self):
        return self._data_lo

    def scratch_size(self):
        return self._size_hi

    def scratch_step(self):
        return self._step_hi

    def scratch_zero(self):
        return self._zero_hi

    def scratch_data(self):
        return self._data_hi

    def scale(self):
        return self._scale

    def psf(self):
        return self._psf

    def lsf(self):
        return self._lsf

    def dtype(self):
        return self._dtype

    def prepare(self, driver):

        size_hi = self._size_hi
        size_lo = self._size_lo
        step_hi = self._step_hi
        step_lo = self._step_lo
        edge_hi = self._edge_hi
        scale = self._scale
        size_hi_fft = 2 * size_hi[2] * size_hi[1] * (size_hi[0] // 2 + 1)

        self._driver = driver
        self._data_lo = driver.mem_alloc(self._size_lo[::-1], self._dtype)
        self._data_hi = driver.mem_alloc_d(self._size_hi[::-1], self._dtype) \
            if self._size_lo != self._size_hi else self._data_lo[1]
        self._dcube = driver.make_dmodel_dcube(
            size_lo, size_hi, edge_hi, scale,
            self._data_lo[1],
            self._data_hi[1], None,
            None, None,
            self._dtype)
        """
        size_lo, size_hi, edge_hi, scale,
            scube_lo,
            scube_hi, scube_hi_fft,
            psf3d_hi, psf3d_hi_fft,
            dtype
        """



        psf3d = self._psf_hi * self._lsf_hi[:, None, None]

        """
        print(psf3d.shape)
        import astropy.io.fits as fits
        fits.writeto('foo.fits', psf3d, overwrite=True)
        exit()
        """

    def evaluate(self, out_extra):

        if self._data_lo[1] is not self._data_hi:
            self._dcube.downscale()

        if True:
            #self._dcube.convolve(self._size_hi, self._)
            pass

        self._driver.mem_copy_d2h(self._data_lo[1], self._data_lo[0])

        if out_extra is not None:
            out_extra['grid_lo'] = self._driver.mem_copy_d2h(self._data_lo[0])
            out_extra['grid_hi'] = self._driver.mem_copy_d2h(self._data_hi)
            if self._psf is not None:
                out_extra['psf_lo'] = self._psf.asarray(self._step_lo[:2])
                out_extra['psf_hi'] = self._psf.asarray(self._step_hi[:2])
                out_extra['psf_hi_fft'] = self._psf_hi.copy()
            if self._lsf is not None:
                out_extra['lsf_lo'] = self._lsf.asarray(self._step_lo[2])
                out_extra['lsf_hi'] = self._lsf.asarray(self._step_hi[2])
                out_extra['lsf_hi_fft'] = self._lsf_hi.copy()
