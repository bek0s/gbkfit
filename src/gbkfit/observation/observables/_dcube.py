from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import astropy.units as u
import numpy as np

import gbkfit.math
from gbkfit.driver import DeviceArray, Driver, DriverFFT
from gbkfit.instrument import LSF, LSFPoint, PSF, PSFPoint, PrimaryBeam
from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import ConfigError


# The unit of the velocities of the spectral axis
_KMS = u.km / u.s

# How an extra output is made from an array of a cube of DCube and the
# grid of the cube (e.g. cube_extra)
ExtraFunction: TypeAlias = Callable[
    [np.ndarray, gridutils.Grid], gridutils.GridData | np.ndarray]


def cube_extra(data: np.ndarray, grid: gridutils.Grid) -> gridutils.GridData:
    """Make an extra output on a grid of DCube a sky and velocity cube."""
    return gridutils.GridData(data, grid.coords, grid.spectral_axis)


def plain_extra(data: np.ndarray, grid: gridutils.Grid) -> np.ndarray:  # noqa
    """Make an extra output on a grid of DCube a plain array."""
    return data


def _distinct(items: Sequence[Any]) -> list[Any]:
    """Return the distinct objects of a list (by identity), in order."""
    return list({id(item): item for item in items}.values())


def _varying_padding(
        dcube: 'DCube',
        size_hi: tuple[int, int, int],
        step_hi: tuple[float, float, float],
        backend_fft: DriverFFT
) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
    """
    Return the size and the padding (edge) of the high-res cube of a DCube
    whose PSF or LSF varies along the spectral axis, for the largest PSF
    and LSF of the channels of the data, and the size of the LSF arrays.
    Along z there is no FFT, so the padding is that of the LSF only. Raise
    ConfigError if a channel of the data is beyond the velocities where
    the PSF or the LSF is known.
    """
    psf, lsf = dcube.psf(), dcube.lsf()
    rest = dcube.grid().coords.rest
    step_lo = dcube.step()
    velocities_lo = dcube.zero()[2] + step_lo[2] * np.arange(dcube.size()[2])
    for kernel, desc in ((psf, 'PSF'), (lsf, 'LSF')):
        if kernel is None:
            continue
        low, high = kernel.velocity_range(rest)
        margin = 1e-6 * abs(step_lo[2])
        if (velocities_lo.min() < low - margin
                or velocities_lo.max() > high + margin):
            raise ConfigError(
                f"the {desc} is given from {low:g} to {high:g} km/s, but "
                f"the channels of the data go from {velocities_lo.min():g} "
                f"to {velocities_lo.max():g} km/s")
    # The high-res channels of the data (before the padding)
    first = velocities_lo[0] - step_lo[2] / 2 + step_hi[2] / 2
    velocities = first + step_hi[2] * np.arange(size_hi[2])
    psf_size = (1, 1)
    if psf is not None:
        psf_size = tuple(np.max([
            p.size(step_hi[:2])
            for p in _distinct(psf.at(velocities * _KMS, rest))],
            axis=0).tolist())
    lsf_size = 1
    if lsf is not None:
        lsf_size = max(
            x.size(step_hi[2])
            for x in _distinct(lsf.at(velocities * _KMS, rest)))
    size, edge = backend_fft.fft_convolution_shape(
        size_hi, psf_size + (lsf_size,))
    return ((int(size[0]), int(size[1]), size_hi[2] + lsf_size - 1),
            (int(edge[0]), int(edge[1]), lsf_size // 2),
            lsf_size)


def _equivalent_dispersion(lsf: LSF, step: float) -> float:
    """
    Return the dispersion of the Gaussian of the FWHM of an LSF (in the
    units of step), from its kernel sampled 16 times finer than step: 0 for
    a kernel narrower than that.
    """
    fine = step / 16
    kernel = lsf.asarray(fine, lsf.size(fine))
    half = kernel.max() / 2
    above = np.flatnonzero(kernel >= half)
    lo, hi = above[0], above[-1]
    if lo == hi:
        return 0.0

    def crossing(inside: int, outside: int) -> float:
        # Where the kernel crosses half its peak between two samples (at
        # the end of the kernel if it does not)
        if not 0 <= outside < len(kernel):
            return float(inside)
        k_in, k_out = kernel[inside], kernel[outside]
        return inside + (outside - inside) * (k_in - half) / (k_in - k_out)
    fwhm = (crossing(hi, hi + 1) - crossing(lo, lo - 1)) * fine
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def _varying_kernels(
        dcube: 'DCube', grid_hi: gridutils.Grid, lsf_size: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Return the images of the PSF of a DCube on its high-res grid: one for
    each channel (nz, ny, nx), or one for all if the PSF does not vary;
    and its LSF of each channel, of size lsf_size (nz, lsf_size). None for
    no PSF or no LSF. The channels of the padding take the PSF and the
    LSF of the nearest velocity where they are known, with the arrays of
    the channels of the data.
    """
    psf, lsf = dcube.psf(), dcube.lsf()
    rest = grid_hi.coords.rest
    size, step = grid_hi.size, grid_hi.coords.step
    velocities = grid_hi.zero()[2] + step[2] * np.arange(size[2])
    psf_images = None
    if psf is not None:
        # (like the PSF/LSF cube of DCubePlan: an even size puts the centre
        # one pixel before the middle)
        offset = gbkfit.math.is_odd(size[:2]) - 1
        psfs = psf.at(velocities * _KMS, rest) if psf.varies() else [psf]
        psf_images = np.stack([
            p.asarray(step[:2], size[:2], offset, dcube.rota())
            for p in psfs])
    lsf_kernels = None
    if lsf is not None:
        if lsf.varies():
            lsf_kernels = np.stack([
                x.asarray(step[2], lsf_size)
                for x in lsf.at(velocities * _KMS, rest)])
        else:
            lsf_kernels = np.tile(lsf.asarray(step[2], lsf_size), (size[2], 1))
    return psf_images, lsf_kernels


def _kernel_cube(dcube: 'DCube', grid_hi: gridutils.Grid) -> np.ndarray:
    """
    Return the PSF/LSF cube of a DCube whose PSF and LSF do not vary, on
    its (padded) high-res grid: the product of the PSF and the LSF (a point
    for the one not given), with its centre rolled to the first pixel, for
    the FFT convolution. Kernels of an even size are offset by -1 pixel,
    since they mostly have a central peak.
    """
    psf, lsf = dcube.psf(), dcube.lsf()
    size, step = grid_hi.size, grid_hi.coords.step
    offset = gbkfit.math.is_odd(size) - 1
    psf_args = (step[:2], size[:2], offset[:2], dcube.rota())
    lsf_args = (step[2], size[2], offset[2])
    # (rolled before their product, which is then the only cube made)
    psf_image = DriverFFT.fft_convolution_shift(
        (psf or PSFPoint()).asarray(*psf_args))
    lsf_profile = DriverFFT.fft_convolution_shift(
        (lsf or LSFPoint()).asarray(*lsf_args))
    return psf_image * lsf_profile[:, None, None]


class DCube:
    """
    A cube of the data of an observable (x, y and the spectral axis), of
    the given grid, made from a model evaluated on a finer grid (scale
    times finer along each axis): the primary beam attenuates it, the PSF
    and the LSF convolve it, it is downscaled to the grid, and masked
    where its values are not above mask_cutoff (the mask is applied, NaN,
    if mask_apply). The weights are smoothed by the PSF and the LSF if
    smooth_weights. Its values are of dtype, and the FFTs of its
    convolutions of fft_dtype (at least as precise).
    """

    def __init__(
            self,
            *,
            size: tuple[int, int, int],
            step: tuple[float, float, float],
            rpix: tuple[float, float, float],
            rval: tuple[float, float, float],
            rota: float,
            rest: u.Quantity | None,
            medium: str | None,
            scale: tuple[int, int, int],
            primary_beam: PrimaryBeam | None,
            psf: PSF | None,
            lsf: LSF | None,
            smooth_weights: bool,
            mask_cutoff: float | None,
            mask_apply: bool,
            dtype: np.dtype,
            fft_dtype: np.dtype
    ):
        # The low-res grid (the grid of the data); the high-res one is made
        # by the plan
        rest = gridutils.make_rest(rest)
        coords = gridutils.Coords(
            step, rpix, rval, rota, rest, gridutils.make_medium(medium, rest))
        self._grid_lo = gridutils.Grid(size, coords, 2)
        self._scale = scale
        self._primary_beam = primary_beam
        self._psf = psf
        self._lsf = lsf
        self._smooth_weights = smooth_weights
        self._mask_cutoff = mask_cutoff
        self._mask_apply = mask_apply
        self._dtype = dtype
        self._fft_dtype = fft_dtype

    def grid(self) -> gridutils.Grid:
        return self._grid_lo

    def size(self) -> tuple[int, int, int]:
        return self._grid_lo.size

    def step(self) -> tuple[float, float, float]:
        return self._grid_lo.coords.step

    def zero(self) -> tuple[float, float, float]:
        return self._grid_lo.zero()

    def rpix(self) -> tuple[float, float, float]:
        return self._grid_lo.coords.rpix

    def rval(self) -> tuple[float, float, float]:
        return self._grid_lo.coords.rval

    def rota(self) -> float:
        return self._grid_lo.coords.rota

    def scale(self) -> tuple[int, int, int]:
        return self._scale

    def primary_beam(self) -> PrimaryBeam | None:
        return self._primary_beam

    def psf(self) -> PSF | None:
        return self._psf

    def lsf(self) -> LSF | None:
        return self._lsf

    def smooth_weights(self) -> bool:
        return self._smooth_weights

    def mask_cutoff(self) -> float | None:
        return self._mask_cutoff

    def mask_apply(self) -> bool:
        return self._mask_apply

    def dtype(self) -> np.dtype:
        return self._dtype

    def fft_dtype(self) -> np.dtype:
        return self._fft_dtype

    def spectral_smoothing(self) -> float:
        """
        Return the dispersion (km/s) by which the instrument smooths the
        spectra of the model: its LSF (the Gaussian of its FWHM; the
        narrowest of the channels if it varies) and its channels (a box of
        their width), added in quadrature.
        """
        channel = self.step()[2]
        lsf = self._lsf
        dispersion = 0.0
        if lsf is not None:
            lsfs = lsf.at(
                (self.zero()[2] + channel * np.arange(self.size()[2])) * _KMS,
                self._grid_lo.coords.rest) if lsf.varies() else [lsf]
            dispersion = min(
                _equivalent_dispersion(x, channel) for x in _distinct(lsfs))
        return float(np.hypot(dispersion, channel / np.sqrt(12)))

    def plan(self, driver: Driver, has_weights: bool) -> 'DCubePlan':
        """
        Plan the evaluation of the cube on a driver, with weights if
        has_weights: its high-res grid, which depends on the FFT of the
        driver, and its memory.
        """
        return DCubePlan(self, driver, has_weights)


class DCubePlan:
    """
    The evaluation of a DCube on a driver: the high-res (scratch) cube the
    model adds to, the low-res cube of the data, and their weights, mask,
    primary beam and convolutions by the PSF and the LSF.
    """

    def __init__(self, dcube: DCube, driver: Driver, has_weights: bool):
        size_lo = dcube.size()
        step_lo = dcube.step()
        rpix_lo = dcube.rpix()
        scale = dcube.scale()
        psf = dcube.psf()
        lsf = dcube.lsf()
        dtype = dcube.dtype()
        fft_dtype = dcube.fft_dtype()
        backend_fft = driver.fft(fft_dtype)

        # The size and step of the high-res cube, before the padding
        size_hi = tuple(size_lo[i] * scale[i] for i in range(3))
        step_hi = tuple(step_lo[i] / scale[i] for i in range(3))
        spat_step_hi = step_hi[:2]
        spec_step_hi = step_hi[2]

        # A PSF or an LSF that varies along the spectral axis has one for
        # each channel: then the PSF is convolved along y and x, with an
        # image for each channel, and the LSF along z (see _varying_padding
        # and _varying_kernels); so are constant ones on drivers whose FFT
        # does it faster (see DriverFFT.separable)
        varies = bool(psf and psf.varies()) or bool(lsf and lsf.varies())
        separable = varies or (backend_fft.separable and bool(psf or lsf))
        if separable:
            size_hi, edge_hi, lsf_size_hi = _varying_padding(
                dcube, size_hi, step_hi, backend_fft)
        else:
            # The convolution is FFT-based, so the cube is padded for the
            # size of the PSF and the LSF (1 for the one not given)
            edge_hi = (0, 0, 0)
            if psf or lsf:
                psf_size_hi = psf.size(spat_step_hi) if psf else (1, 1)
                lsf_size_hi = lsf.size(spec_step_hi) if lsf else 1
                size_hi, edge_hi = backend_fft.fft_convolution_shape(
                    size_hi, psf_size_hi + (lsf_size_hi,))

        # The high-res grid: scale pixels for each low-res pixel, centred on
        # it, after edge_hi pixels of padding
        rpix_hi = tuple(
            (rpix_lo[i] + 0.5) * scale[i] - 0.5 + edge_hi[i] for i in range(3))
        grid_hi = gridutils.Grid(
            tuple(size_hi), dcube.grid().coords._replace(
                step=step_hi, rpix=rpix_hi), 2)

        # The shapes of the arrays are the reversed sizes
        shape_lo = size_lo[::-1]
        shape_hi = size_hi[::-1]

        # If they are convolved separately: the convolution of each channel
        # by its image of the PSF (or by the only one), and the LSF of each
        # channel
        self._psf_convolution = None
        self._lsf_kernels_hi = None
        self._lsf_size_hi = None
        self._zscratch_hi = None
        if separable:
            psf_images, lsf_kernels = _varying_kernels(
                dcube, grid_hi, lsf_size_hi)
            if psf_images is not None:
                self._psf_convolution = backend_fft.convolution(
                    DriverFFT.fft_convolution_shift(
                        psf_images, axes=(1, 2)).astype(fft_dtype),
                    shape_hi, axes=(1, 2), dtype=dtype)
            if lsf_kernels is not None:
                self._lsf_kernels_hi = driver.mem_copy_h2d(
                    lsf_kernels.astype(dtype))
                self._zscratch_hi = driver.mem_alloc_d(shape_hi, dtype)
            self._lsf_size_hi = lsf_size_hi

        # Otherwise, the convolution by the PSF/LSF cube (see _kernel_cube)
        self._convolution = None
        if (psf or lsf) and not separable:
            self._convolution = backend_fft.convolution(
                _kernel_cube(dcube, grid_hi).astype(fft_dtype, copy=False),
                shape_hi, dtype=dtype)

        # The response of the primary beam on the high-res grid, if there
        # is one: one image for all channels, or one for each channel of a
        # beam that varies with the frequency
        self._pbeam_hi = None
        beam = dcube.primary_beam()
        if beam is not None:
            velocities = grid_hi.zero()[2] \
                + step_hi[2] * np.arange(size_hi[2])
            beams = beam.at(velocities * _KMS, grid_hi.coords.rest) \
                if beam.varies() else [beam]
            pbeam_hi = np.array([x.response(grid_hi) for x in beams])
            self._pbeam_hi = driver.mem_copy_h2d(pbeam_hi.astype(dtype))

        # The low- and high-res data and weight cubes; without oversampling
        # and padding they are one
        self._dcube_lo = driver.mem_alloc_d(shape_lo, dtype)
        self._dcube_hi = self._dcube_lo
        driver.mem_fill(self._dcube_lo, 0)
        if size_lo != size_hi:
            self._dcube_hi = driver.mem_alloc_d(shape_hi, dtype)
            driver.mem_fill(self._dcube_hi, 0)
        self._wcube_lo = None
        self._wcube_hi = None
        if has_weights:
            self._wcube_lo = driver.mem_alloc_d(shape_lo, dtype)
            self._wcube_hi = self._wcube_lo
            driver.mem_fill(self._wcube_lo, 1)
            if size_lo != size_hi:
                self._wcube_hi = driver.mem_alloc_d(shape_hi, dtype)
                driver.mem_fill(self._wcube_hi, 1)

        # The mask cube, if masking is enabled: masking is done on the
        # low-res cube only
        self._mcube_lo = None
        if dcube.mask_cutoff() is not None:
            self._mcube_lo = driver.mem_alloc_d(shape_lo, dtype)
            driver.mem_fill(self._mcube_lo, 1)

        self._dcube = dcube
        self._varies = varies
        self._separable = separable
        self._grid_hi = grid_hi
        self._edge_hi = edge_hi
        self._has_weights = has_weights
        self._driver = driver
        self._backend_dmodel = driver.native_class('DModel', dtype)()

    def scratch_grid(self) -> gridutils.Grid:
        return self._grid_hi

    def scratch_edge(self) -> tuple[int, int, int]:
        return self._edge_hi

    def scratch_dcube(self) -> DeviceArray:
        return self._dcube_hi

    def scratch_wcube(self) -> DeviceArray | None:
        return self._wcube_hi

    def dcube(self) -> DeviceArray:
        return self._dcube_lo

    def wcube(self) -> DeviceArray | None:
        return self._wcube_lo

    def mcube(self) -> DeviceArray | None:
        return self._mcube_lo

    def evaluate(
            self,
            out_extra: dict[str, Any] | None,
            extra_lo: ExtraFunction,
            extra_hi: ExtraFunction,
            flux_scale: float | None = None
    ) -> None:
        """
        Record the high-res cube into the low-res one (see record) and mask
        it (see mask). The extra outputs on the low- and high-res grids go
        to out_extra as extra_lo and extra_hi make them from their data and
        grid (e.g. cube_extra or plain_extra).
        """
        self.record(flux_scale)
        self.mask()
        if out_extra is not None:
            self._extras(out_extra, extra_lo, extra_hi)

    def record(self, flux_scale: float | None = None) -> None:
        """
        Scale the flux of the high-res cube by flux_scale (if given; e.g.
        the calibration of an observation), attenuate it by the primary
        beam, convolve and downscale it into the low-res one: the data the
        instrument records.
        """
        dcube = self._dcube
        dcube_lo = self._dcube_lo
        dcube_hi = self._dcube_hi
        wcube_lo = self._wcube_lo
        wcube_hi = self._wcube_hi
        pbeam_hi = self._pbeam_hi
        has_weights = self._has_weights
        driver = self._driver
        backend_dmodel = self._backend_dmodel

        if flux_scale is not None:
            driver.math_mul(dcube_hi, flux_scale, out=dcube_hi)

        # The primary beam attenuates the light before the PSF
        if pbeam_hi is not None:
            driver.math_mul(dcube_hi, pbeam_hi, out=dcube_hi)

        # The convolution, of the weights too if they are smoothed (in
        # float32, the rounding of the FFT leaves noise of about 1e-7 of
        # the peak in the faint parts of the cube, which spoils their
        # moments: see dmodel_mmaps_moments; hence fft_dtype)
        cubes = [dcube_hi]
        if has_weights and dcube.smooth_weights():
            cubes.append(wcube_hi)
        if self._separable:
            # The PSF of each channel, then the LSF of each channel, in the
            # order the light meets them
            for cube in cubes:
                if self._psf_convolution is not None:
                    self._psf_convolution(cube)
                if self._lsf_kernels_hi is not None:
                    backend_dmodel.dcube_convolve_z(
                        self._lsf_kernels_hi, cube, self._zscratch_hi)
        elif self._convolution is not None:
            for cube in cubes:
                self._convolution(cube)

        # The downscaling, which also removes the padding: the weights
        # always need it, smoothed or not, to reach the low-res cube
        if dcube_lo is not dcube_hi:
            backend_dmodel.dcube_downscale(
                dcube.scale(), self._edge_hi, dcube_hi, dcube_lo)
            if has_weights:
                backend_dmodel.dcube_downscale(
                    dcube.scale(), self._edge_hi, wcube_hi, wcube_lo)

    def mask(self) -> None:
        """
        Mask the low-res cube where its values are not above mask_cutoff,
        if masking is enabled (the mask is applied, NaN, if mask_apply).
        """
        mask_cutoff = self._dcube.mask_cutoff()
        if mask_cutoff is not None:
            self._backend_dmodel.dcube_mask(
                mask_cutoff, self._dcube.mask_apply(), self._dcube_lo,
                self._mcube_lo, self._wcube_lo)

    def _extras(
            self,
            out_extra: dict[str, Any],
            extra_lo: ExtraFunction,
            extra_hi: ExtraFunction
    ) -> None:
        """Write the extra outputs (see evaluate)."""
        dcube = self._dcube
        step_lo = dcube.step()
        step_hi = self._grid_hi.coords.step
        psf = dcube.psf()
        lsf = dcube.lsf()
        dcube_lo = self._dcube_lo
        dcube_hi = self._dcube_hi
        wcube_lo = self._wcube_lo
        wcube_hi = self._wcube_hi
        mcube_lo = self._mcube_lo
        pbeam_hi = self._pbeam_hi
        has_weights = self._has_weights
        mask_cutoff = dcube.mask_cutoff()
        driver = self._driver

        def lo(data: DeviceArray) -> gridutils.GridData | np.ndarray:
            return extra_lo(driver.mem_copy_d2h(data), dcube.grid())

        def hi(data: DeviceArray) -> gridutils.GridData | np.ndarray:
            return extra_hi(driver.mem_copy_d2h(data), self._grid_hi)

        out_extra.update(dcube_lo=lo(dcube_lo), dcube_hi=hi(dcube_hi))
        if mask_cutoff is not None:
            out_extra.update(mcube_lo=lo(mcube_lo))
        if has_weights:
            out_extra.update(wcube_lo=lo(wcube_lo), wcube_hi=hi(wcube_hi))
        if self._varies:
            # The images of the PSF and the LSF of the channels, made again
            # (they are not kept on the host)
            psf_images, lsf_kernels = _varying_kernels(
                dcube, self._grid_hi, self._lsf_size_hi)
            if psf_images is not None:
                out_extra.update(psf_hi=psf_images)
            if lsf_kernels is not None:
                out_extra.update(lsf_hi=lsf_kernels)
        else:
            if psf:
                out_extra.update(
                    psf_lo=psf.asarray(step_lo[:2], rota=dcube.rota()),
                    psf_hi=psf.asarray(step_hi[:2], rota=dcube.rota()))
            if lsf:
                out_extra.update(
                    lsf_lo=lsf.asarray(step_lo[2]),
                    lsf_hi=lsf.asarray(step_hi[2]))
            if psf or lsf:
                # (made again: it is not kept)
                out_extra.update(pcube_hi=_kernel_cube(
                    dcube, self._grid_hi).astype(
                        dcube.fft_dtype(), copy=False))
        if pbeam_hi is not None:
            pbeam = driver.mem_copy_d2h(pbeam_hi)
            out_extra.update(pbeam_hi=pbeam[0] if len(pbeam) == 1 else pbeam)
