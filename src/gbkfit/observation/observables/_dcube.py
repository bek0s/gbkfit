from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import astropy.units as u
import numpy as np

import gbkfit.math
from gbkfit.driver import DeviceArray, Driver
from gbkfit.driver.fft import DriverFFT
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


class DCube:
    """
    A cube of the data of an observable (x, y and the spectral axis), of
    the given grid, made from a model evaluated on a finer grid (scale
    times finer along each axis): the primary beam attenuates it, the PSF
    and the LSF convolve it, it is downscaled to the grid, and masked
    where its values are not above mask_cutoff (the mask is applied, NaN,
    if mask_apply). The weights are smoothed by the PSF and the LSF if
    smooth_weights.
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
            scale: tuple[int, int, int],
            primary_beam: PrimaryBeam | None,
            psf: PSF | None,
            lsf: LSF | None,
            smooth_weights: bool,
            mask_cutoff: float | None,
            mask_apply: bool,
            dtype: np.dtype
    ):
        # The low-res grid (the grid of the data); the high-res one is made
        # by the plan
        coords = gridutils.Coords(
            step, rpix, rval, rota, gridutils.make_rest(rest))
        self._grid_lo = gridutils.Grid(size, coords, 2)
        self._scale = scale
        self._primary_beam = primary_beam
        self._psf = psf
        self._lsf = lsf
        self._smooth_weights = smooth_weights
        self._mask_cutoff = mask_cutoff
        self._mask_apply = mask_apply
        self._dtype = dtype

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
    primary beam and PSF/LSF cube.
    """

    def __init__(self, dcube: DCube, driver: Driver, has_weights: bool):
        size_lo = dcube.size()
        step_lo = dcube.step()
        rpix_lo = dcube.rpix()
        scale = dcube.scale()
        psf = dcube.psf()
        lsf = dcube.lsf()
        dtype = dcube.dtype()
        backend_fft = driver.fft(dtype)

        # The size and step of the high-res cube, before the padding
        size_hi = tuple(size_lo[i] * scale[i] for i in range(3))
        step_hi = tuple(step_lo[i] / scale[i] for i in range(3))
        spat_step_hi = step_hi[:2]
        spec_step_hi = step_hi[2]

        # A PSF or an LSF that varies along the spectral axis has one for
        # each channel: then the PSF is convolved along y and x, with an
        # image for each channel, and the LSF along z (see _varying_padding
        # and _varying_kernels)
        varies = bool(psf and psf.varies()) or bool(lsf and lsf.varies())
        if varies:
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

        # The images of the PSF (one for each channel, or one for all) and
        # the LSF of each channel, if one of them varies
        self._psf_images_hi = None
        self._lsf_kernels_hi = None
        self._lsf_size_hi = None
        self._zscratch_hi = None
        if varies:
            psf_images, lsf_kernels = _varying_kernels(
                dcube, grid_hi, lsf_size_hi)
            if psf_images is not None:
                self._psf_images_hi = driver.mem_copy_h2d(
                    backend_fft.fft_convolution_shift(
                        psf_images, axes=(1, 2)).astype(dtype))
            if lsf_kernels is not None:
                self._lsf_kernels_hi = driver.mem_copy_h2d(
                    lsf_kernels.astype(dtype))
                self._zscratch_hi = driver.mem_alloc_d(shape_hi, dtype)
            self._lsf_size_hi = lsf_size_hi

        # Otherwise, the PSF/LSF cube of the FFT-based convolution: the
        # product of the PSF and the LSF (a point for the one not given),
        # with its centre rolled to (0, 0, 0). Kernels of an even size are
        # offset by -1 pixel, since they mostly have a central peak.
        self._pcube_hi = None
        if (psf or lsf) and not varies:
            offset_hi = gbkfit.math.is_odd(size_hi) - 1
            psf_args = (spat_step_hi, size_hi[:2], offset_hi[:2], dcube.rota())
            lsf_args = (spec_step_hi, size_hi[2], offset_hi[2])
            psf_hi = psf.asarray(*psf_args) if psf \
                else PSFPoint().asarray(*psf_args)
            lsf_hi = lsf.asarray(*lsf_args) if lsf \
                else LSFPoint().asarray(*lsf_args)
            pcube_hi = (psf_hi * lsf_hi[:, None, None]).astype(dtype)
            self._pcube_hi = driver.mem_copy_h2d(
                backend_fft.fft_convolution_shift(pcube_hi))

        # The response of the primary beam on the high-res grid (one image
        # for all channels), if there is one
        self._pbeam_hi = None
        if dcube.primary_beam() is not None:
            pbeam_hi = dcube.primary_beam().response(grid_hi)
            self._pbeam_hi = driver.mem_copy_h2d(
                pbeam_hi[None].astype(dtype))

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
        self._grid_hi = grid_hi
        self._edge_hi = edge_hi
        self._has_weights = has_weights
        self._driver = driver
        self._backend_fft = backend_fft
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
            extra_hi: ExtraFunction
    ) -> None:
        """
        Attenuate by the primary beam, convolve, downscale and mask the
        high-res cube into the low-res one. The extra outputs on the low-
        and high-res grids go to out_extra as extra_lo and extra_hi make
        them from their data and grid (e.g. cube_extra or plain_extra).
        """
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
        pcube_hi = self._pcube_hi
        pbeam_hi = self._pbeam_hi
        has_weights = self._has_weights
        mask_cutoff = dcube.mask_cutoff()
        driver = self._driver
        backend_fft = self._backend_fft
        backend_dmodel = self._backend_dmodel

        # The primary beam attenuates the light before the PSF
        if pbeam_hi is not None:
            driver.math_mul(dcube_hi, pbeam_hi, out=dcube_hi)

        # The convolution, of the weights too if they are smoothed (in
        # float32, the rounding of the FFT leaves noise of about 1e-7 of
        # the peak in the faint parts of the cube, which spoils their
        # moments: see dmodel_mmaps_moments)
        cubes = [dcube_hi]
        if has_weights and dcube.smooth_weights():
            cubes.append(wcube_hi)
        if self._varies:
            # The PSF of each channel, then the LSF of each channel, in the
            # order the light meets them
            for cube in cubes:
                if self._psf_images_hi is not None:
                    backend_fft.fft_convolve_xy_cached(
                        cube, self._psf_images_hi)
                if self._lsf_kernels_hi is not None:
                    backend_dmodel.dcube_convolve_z(
                        self._lsf_kernels_hi, cube, self._zscratch_hi)
        elif psf or lsf:
            for cube in cubes:
                backend_fft.fft_convolve_cached(cube, pcube_hi)

        # The downscaling, which also removes the padding: the weights
        # always need it, smoothed or not, to reach the low-res cube
        if dcube_lo is not dcube_hi:
            backend_dmodel.dcube_downscale(
                dcube.scale(), self._edge_hi, dcube_hi, dcube_lo)
            if has_weights:
                backend_dmodel.dcube_downscale(
                    dcube.scale(), self._edge_hi, wcube_hi, wcube_lo)

        # The mask, applied to the data if asked
        if mask_cutoff is not None:
            backend_dmodel.dcube_mask(
                mask_cutoff, dcube.mask_apply(), dcube_lo, mcube_lo, wcube_lo)

        if out_extra is None:
            return

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
                out_extra.update(pcube_hi=driver.mem_copy_d2h(pcube_hi))
        if pbeam_hi is not None:
            out_extra.update(pbeam_hi=driver.mem_copy_d2h(pbeam_hi)[0])
