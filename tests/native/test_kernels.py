"""
Tests for the native (C++/CUDA) kernels, called through the driver
backends. Every test runs once per available driver.

Kernel outputs are compared against reference outputs stored in
tests/data/native_kernels. All drivers share the same references.
After an intended change in behaviour, regenerate them with:

    pytest tests/native --force-regen
"""

import pathlib

import numpy as np
import pytest

from gbkfit.model.gmodels import traits


DTYPE = np.float32

DATA_DIR = pathlib.Path(__file__).parents[1] / 'data'


@pytest.fixture
def original_datadir():
    """Read and write the reference outputs of this module in tests/data."""
    return DATA_DIR / 'native_kernels'


class Memory:
    """Copies arrays between the host and the memory of a driver."""

    def __init__(self, driver):
        self._driver = driver

    def to_device(self, array):
        if array is None:
            return None
        return self._driver.mem_copy_h2d(np.ascontiguousarray(array))

    def to_host(self, array):
        return np.array(self._driver.mem_copy_d2h(array))


def smooth_cube(shape):
    """A deterministic, smoothly varying test cube."""
    z, y, x = np.indices(shape, dtype=DTYPE)
    return np.sin(0.3 * x) * np.cos(0.2 * y) + 0.1 * z + 1


def check_against_reference(ndarrays_regression, basename, arrays):
    """Compare arrays with their stored reference, to float32 precision."""
    tolerances = {
        name: dict(rtol=1e-5, atol=1e-5 * float(np.abs(array).max()))
        for name, array in arrays.items()}
    ndarrays_regression.check(
        arrays, basename=basename, tolerances=tolerances)


def test_fft_roundtrip(driver):
    memory = Memory(driver)
    fft = driver.fft(DTYPE)
    shape = (16, 24, 32)
    data = smooth_cube(shape)
    data_r = memory.to_device(data)
    data_c = memory.to_device(
        np.zeros(fft.fft_complex_shape(shape), np.complex64))
    fft.fft_r2c(data_r, data_c)
    fft.fft_c2r(data_c, data_r)
    result = memory.to_host(data_r) / data.size
    np.testing.assert_allclose(result, data, rtol=1e-5, atol=1e-5)


def test_fft_convolve(driver, ndarrays_regression):
    memory = Memory(driver)
    fft = driver.fft(DTYPE)
    shape = (16, 24, 32)
    kernel = np.zeros(shape, DTYPE)
    kernel[8, 12, 16] = 0.5
    kernel[9, 12, 16] = 0.25
    kernel[8, 13, 17] = 0.25
    data = memory.to_device(smooth_cube(shape))
    fft.fft_convolve_cached(data, memory.to_device(kernel))
    check_against_reference(
        ndarrays_regression, 'fft_convolve',
        dict(result=memory.to_host(data)))


def test_dcube_downscale(driver, ndarrays_regression):
    memory = Memory(driver)
    dmodel = driver.native_class('DModel', DTYPE)()
    cube_hi = memory.to_device(smooth_cube((40, 60, 80)))
    cube_lo = memory.to_device(np.zeros((20, 20, 20), DTYPE))
    dmodel.dcube_downscale((4, 3, 2), (0, 0, 0), cube_hi, cube_lo)
    check_against_reference(
        ndarrays_regression, 'dcube_downscale',
        dict(result=memory.to_host(cube_lo)))


def test_arrays_are_checked(driver):
    # The native modules check the dtype, layout, device and shapes of
    # their arrays, instead of reading them as raw memory or converting
    # them to temporary copies (which would lose the outputs)
    memory = Memory(driver)
    dmodel = driver.native_class('DModel', DTYPE)()
    cube_hi = memory.to_device(smooth_cube((40, 60, 80)))

    def downscale_into(cube_lo, scale=(4, 3, 2)):
        dmodel.dcube_downscale(scale, (0, 0, 0), cube_hi, cube_lo)

    with pytest.raises(TypeError, match="incompatible"):
        downscale_into(memory.to_device(np.zeros((20, 20, 20), np.float64)))
    with pytest.raises(TypeError, match="incompatible"):
        downscale_into(memory.to_device(
            np.zeros((20, 20, 40), DTYPE))[:, :, ::2])
    with pytest.raises(TypeError, match="incompatible"):
        downscale_into(_array_on_other_device(driver, (20, 20, 20)))
    with pytest.raises(ValueError, match="does not fit"):
        downscale_into(memory.to_device(np.zeros((20, 20, 20), DTYPE)),
                       scale=(4, 4, 4))


def _array_on_other_device(driver, shape):
    """An array in host memory for cuda, and in device memory for host."""
    if driver.type() == 'cuda':
        return np.zeros(shape, DTYPE)
    cupy = pytest.importorskip('cupy')
    return cupy.zeros(shape, DTYPE)


# Wider than the tiles of spectra that the host kernels copy (128 pixels),
# so that each row also has a partial tile
SPECTRA_SHAPE = (40, 3, 130)


def gaussian_lines(amplitude, centre, sigma):
    """
    A cube of the shape SPECTRA_SHAPE of Gaussian lines of the given
    amplitude, centre and dispersion (in channels; arrays of the shape of
    a channel), sampled at the centres of the channels. The lines of the
    first column are faint, to be masked.
    """
    z = np.arange(SPECTRA_SHAPE[0])[:, None, None]
    amplitude = np.where(np.arange(SPECTRA_SHAPE[2]) == 0, 1e-3, amplitude)
    return amplitude * np.exp(-0.5 * ((z - centre) / sigma) ** 2)


def line_shapes():
    """The amplitudes, centres and dispersions (channels) of test lines."""
    y, x = np.indices(SPECTRA_SHAPE[1:])
    return 1 + 0.01 * x, 15 + 5 * np.sin(0.1 * x + y), 2 + 0.02 * x + y / 2


def test_mmaps_moments(driver):
    # The moments of skewed lines, with weights, against those computed
    # here in double precision
    memory = Memory(driver)
    dmodel = driver.native_class('DModel', DTYPE)()
    amplitude, centre, sigma = line_shapes()
    cube = gaussian_lines(amplitude, centre, sigma) \
        + gaussian_lines(0.3 * amplitude, centre + 6, sigma)
    weights = smooth_cube(SPECTRA_SHAPE)
    orders = (0, 1, 2, 3, 4)
    maps_shape = (len(orders),) + SPECTRA_SHAPE[1:]
    mmaps_d = memory.to_device(np.zeros(maps_shape, DTYPE))
    mmaps_m = memory.to_device(np.zeros(SPECTRA_SHAPE[1:], DTYPE))
    mmaps_w = memory.to_device(np.zeros(maps_shape, DTYPE))
    # Velocities near 0: far from 0, the float32 sums lose the precision
    # of moment 1, and the central moments with it
    step, zero = 5, -100
    dmodel.mmaps_moments(
        (1, 1, step), (0, 0, zero),
        memory.to_device(cube.astype(DTYPE)), memory.to_device(weights),
        0.1, memory.to_device(np.array(orders, np.int32)),
        mmaps_d, mmaps_m, mmaps_w)

    v = zero + step * np.arange(SPECTRA_SHAPE[0])[:, None, None]
    m0 = np.sum(cube, axis=0) * step
    m1 = np.sum(cube * v, axis=0) * step / m0
    central = {
        k: np.sum(cube * (v - m1) ** k, axis=0) * step / m0
        for k in (2, 3, 4)}
    expected = [m0, m1, np.sqrt(central[2]), central[3], central[4]]
    expected_w = np.sum(weights * cube, axis=0) * step / m0
    # The faint lines are masked: NaN and 0 in the mask
    valid = m0 > 0.1
    assert valid.sum() == valid.size - SPECTRA_SHAPE[1]
    np.testing.assert_array_equal(memory.to_host(mmaps_m), valid)
    result = memory.to_host(mmaps_d)
    for order, expected_map in zip(orders, expected):
        assert np.isnan(result[order][~valid]).all()
        np.testing.assert_allclose(
            result[order][valid], expected_map[valid], rtol=1e-4,
            atol=1e-4 * np.abs(expected_map[valid]).max())
    result_w = memory.to_host(mmaps_w)[0]
    assert np.isnan(result_w[~valid]).all()
    np.testing.assert_allclose(result_w[valid], expected_w[valid], rtol=1e-5)


@pytest.mark.parametrize('orders, masked', [((0, 1), False), ((0, 1, 2), True)])
def test_mmaps_moments_mask_negative_variances(driver, orders, masked):
    # A spectrum whose variance is negative (e.g. the faint ringing of a
    # convolution) has no dispersion: it is masked, for all the orders,
    # if they need it
    memory = Memory(driver)
    dmodel = driver.native_class('DModel', DTYPE)()
    cube = np.array([[[1, -1]], [[2, 3]], [[1, -1]]], DTYPE)
    mmaps_d = memory.to_device(np.zeros((len(orders), 1, 2), DTYPE))
    mmaps_m = memory.to_device(np.zeros((1, 2), DTYPE))
    dmodel.mmaps_moments(
        (1, 1, 1), (0, 0, -1), memory.to_device(cube), None, 0.1,
        memory.to_device(np.array(orders, np.int32)), mmaps_d, mmaps_m, None)
    np.testing.assert_array_equal(memory.to_host(mmaps_m), [[1, not masked]])
    result = memory.to_host(mmaps_d)
    assert np.isfinite(result[:, 0, 0]).all()
    assert np.isnan(result[:, 0, 1]).all() == masked

def test_mmaps_gaussian(driver):
    # Gaussian lines are fitted exactly: their flux, velocity and
    # dispersion, in the units of the spectral axis
    memory = Memory(driver)
    dmodel = driver.native_class('DModel', DTYPE)()
    amplitude, centre, sigma = line_shapes()
    cube = gaussian_lines(amplitude, centre, sigma)
    mmaps_d = memory.to_device(np.zeros((3,) + SPECTRA_SHAPE[1:], DTYPE))
    mmaps_m = memory.to_device(np.zeros(SPECTRA_SHAPE[1:], DTYPE))
    step, zero = 5, 1000
    dmodel.mmaps_gaussian(
        (1, 1, step), (0, 0, zero), memory.to_device(cube.astype(DTYPE)),
        0.1, memory.to_device(np.array([0, 1, 2], np.int32)),
        mmaps_d, mmaps_m)

    expected = [
        amplitude * sigma * np.sqrt(2 * np.pi) * step,
        zero + step * centre,
        step * sigma]
    valid = memory.to_host(mmaps_m).astype(bool)
    assert valid.sum() == valid.size - SPECTRA_SHAPE[1]
    assert not valid[:, 0].any()
    result = memory.to_host(mmaps_d)
    for result_map, expected_map in zip(result, expected):
        assert np.isnan(result_map[~valid]).all()
        np.testing.assert_allclose(
            result_map[valid], expected_map[valid], rtol=1e-3)


def _residual_inputs():
    """Observed and model data, errors, masks and weights for a residual."""
    n = 1000
    i = np.arange(n, dtype=DTYPE)
    return dict(
        obs_d=np.sin(0.01 * i) + 1,
        obs_e=np.cos(0.02 * i) + 1.5,
        obs_m=(i % 7 != 0).astype(DTYPE),
        mdl_d=np.sin(0.011 * i) + 1,
        mdl_w=np.cos(0.005 * i) ** 2,
        mdl_m=(i % 11 != 0).astype(DTYPE))


def test_residual(driver, ndarrays_regression):
    memory = Memory(driver)
    objective = driver.native_class('Objective', DTYPE)()
    inputs = {k: memory.to_device(v) for k, v in _residual_inputs().items()}
    residual = memory.to_device(np.zeros(1000, DTYPE))
    objective.residual(**inputs, weight=0.7, res=residual)
    check_against_reference(
        ndarrays_regression, 'residual',
        dict(result=memory.to_host(residual)))


def test_residual_of_masked_pixels_is_zero(driver):
    # Masked pixels have no residual, also where their data or model are
    # NaN (masked data, and the moments of masked spectra)
    memory = Memory(driver)
    objective = driver.native_class('Objective', DTYPE)()
    inputs = _residual_inputs()
    masked = (inputs['obs_m'] == 0) | (inputs['mdl_m'] == 0)
    inputs['obs_d'][inputs['obs_m'] == 0] = np.nan
    inputs['mdl_d'][inputs['mdl_m'] == 0] = np.nan
    residual = memory.to_device(np.zeros(1000, DTYPE))
    objective.residual(
        **{k: memory.to_device(v) for k, v in inputs.items()},
        weight=0.7, res=residual)
    result = memory.to_host(residual)
    np.testing.assert_array_equal(result[masked], 0)
    assert np.isfinite(result).all() and result[~masked].any()


def residual_sum(driver, values, squared):
    memory = Memory(driver)
    objective = driver.native_class('Objective', DTYPE)()
    total = memory.to_device(np.zeros(1, np.float64))
    objective.residual_sum(
        memory.to_device(values.astype(DTYPE)), squared, total)
    return memory.to_host(total)[0]


def test_residual_sum(driver):
    values = np.linspace(-1, 1, 1000, dtype=DTYPE)
    assert residual_sum(driver, values, True) == pytest.approx(
        np.sum(values.astype(np.float64) ** 2), rel=1e-6)
    assert residual_sum(driver, values, False) == pytest.approx(
        np.sum(np.abs(values.astype(np.float64))), rel=1e-6)


def test_residual_sum_precision(driver):
    # Accumulated or returned in float32, the small terms would be lost
    # next to the large one (float32 numbers near 1e8 are 8 apart)
    values = np.ones(1000)
    values[0] = 1e8
    assert residual_sum(driver, values, False) == 1e8 + 999


def _trait_set(driver, memory, trait, params=()):
    """A native set of one trait, with the given parameter values."""
    consts = [float(c) for c in trait.consts()]
    return driver.native_class('TraitSet', DTYPE)(
        uids=memory.to_device(np.array([trait.uid()], np.int32)),
        cvalues=memory.to_device(np.array(consts, DTYPE)),
        ccounts=memory.to_device(np.array([len(consts)], np.int32)),
        pvalues=memory.to_device(np.array(params, DTYPE)),
        pcounts=memory.to_device(np.array([len(params)], np.int32)))


def evaluate_smdisk(driver, thick):
    """
    Evaluate a smooth disk with an exponential brightness profile, an
    arctan rotation curve, and a uniform velocity dispersion. A thick
    disk also has a sech2 vertical brightness profile.
    """
    memory = Memory(driver)
    spat_size = (49, 49, 21 if thick else 1)
    spec_size = 61
    image_shape = spat_size[1::-1]
    cube_shape = spat_size[::-1]
    outputs = dict(
        image=memory.to_device(np.zeros(image_shape, DTYPE)),
        scube=memory.to_device(np.zeros((spec_size,) + image_shape, DTYPE)),
        velocity=memory.to_device(np.zeros(cube_shape, DTYPE)),
        dispersion=memory.to_device(np.zeros(cube_shape, DTYPE)),
        weight=memory.to_device(np.zeros(cube_shape, DTYPE)))

    def scalar(value):
        return memory.to_device(np.array([value], DTYPE))

    def trait_set(trait, params=()):
        return _trait_set(driver, memory, trait, params)

    # A disk without height traits is evaluated as a thin disk
    heights = dict(
        rht=trait_set(traits.BHTraitSech2(), (1.0,)),
        vht=trait_set(traits.VHTraitOne()),
        dht=trait_set(traits.DHTraitOne())) if thick else {}

    disk = driver.native_class('Disk', DTYPE)(
        loose=False, tilted=False,
        rnodes=memory.to_device(np.linspace(0, 20, 21, dtype=DTYPE)),
        vsys=scalar(10), xpos=scalar(0.5), ypos=scalar(-1.0),
        posa=scalar(30), incl=scalar(50),
        rpt=trait_set(traits.BPTraitExponential(), (1.0, 5.0)),
        vpt=trait_set(traits.VPTraitTanArctan(), (3.0, 200.0)),
        dpt=trait_set(traits.DPTraitUniform(), (25.0,)),
        **heights)
    driver.native_class('GModel', DTYPE).smdisk_evaluate(
        disk,
        spat_size=spat_size,
        spat_step=(1.0, 1.0, 1.0),
        spat_zero=(-24.0, -24.0, -(spat_size[2] // 2)),
        spat_rota=0,
        spec_size=spec_size, spec_step=10.0, spec_zero=-300.0,
        lines=memory.to_device(np.array([[0, 1, 1]], DTYPE)),
        image=outputs['image'], scube=outputs['scube'],
        vdata_cmp=outputs['velocity'], ddata_cmp=outputs['dispersion'],
        vdweight_cmp=outputs['weight'])
    # The velocity and dispersion are weighted sums; their means are 0
    # where the disk is not
    result = {name: memory.to_host(array) for name, array in outputs.items()}
    weight = result.pop('weight')
    for name in ('velocity', 'dispersion'):
        result[name] = np.divide(
            result[name], weight, out=np.zeros_like(weight),
            where=weight != 0)
    return result


def test_smdisk_thin(driver, ndarrays_regression):
    check_against_reference(
        ndarrays_regression, 'smdisk_thin',
        evaluate_smdisk(driver, thick=False))


def test_smdisk_thick(driver, ndarrays_regression):
    check_against_reference(
        ndarrays_regression, 'smdisk_thick',
        evaluate_smdisk(driver, thick=True))


def disk_image(driver, rnodes=21, ntraits=1, heights=False, vpt=False,
               vht=False, monte_carlo=False):
    """
    Evaluate into an image a native disk of rnodes radial nodes and
    ntraits uniform density traits (with uniform height traits, with
    heights), and optionally a velocity trait (with its height trait, with
    vht), as a Monte Carlo disk with monte_carlo.
    """
    memory = Memory(driver)

    def scalar(value):
        return memory.to_device(np.array([value], DTYPE))

    def trait_set(trait, params, n):
        consts = [float(c) for c in trait.consts()]
        return driver.native_class('TraitSet', DTYPE)(
            uids=memory.to_device(np.full(n, trait.uid(), np.int32)),
            cvalues=memory.to_device(np.array(consts * n, DTYPE)),
            ccounts=memory.to_device(np.full(n, len(consts), np.int32)),
            pvalues=memory.to_device(np.array(params * n, DTYPE)),
            pcounts=memory.to_device(np.full(n, len(params), np.int32)))
    options = dict(
        loose=False, tilted=False,
        rnodes=memory.to_device(np.linspace(0, 20, rnodes, dtype=DTYPE)),
        vsys=scalar(0), xpos=scalar(0), ypos=scalar(0),
        posa=scalar(0), incl=scalar(0),
        rpt=trait_set(traits.BPTraitUniform(), [1.0], ntraits))
    if heights:
        options['rht'] = trait_set(traits.BHTraitUniform(), [1.0], ntraits)
    if vpt:
        options['vpt'] = trait_set(traits.VPTraitTanUniform(), [100.0], 1)
    if vht:
        options['vht'] = trait_set(traits.VHTraitOne(), [], 1)
    disk = driver.native_class('Disk', DTYPE)(**options)
    image = memory.to_device(np.zeros((8, 8), DTYPE))
    grid = dict(
        spat_size=(8, 8, 4 if heights else 1), spat_step=(1.0, 1.0, 1.0),
        spat_zero=(-4.0, -4.0, -2.0), spat_rota=0,
        spec_size=1, spec_step=1.0, spec_zero=0.0, image=image)
    gmodel = driver.native_class('GModel', DTYPE)
    if monte_carlo:
        gmodel.mcdisk_evaluate(
            disk, cloud_flux=memory.to_device(np.ones(1, DTYPE)), seed=0,
            nclouds=10, ncloudscsum=memory.to_device(np.array([10], np.int32)),
            has_analytical_integral=memory.to_device(
                np.ones(ntraits, bool)),
            **grid)
    else:
        gmodel.smdisk_evaluate(disk, **grid)
    return memory.to_host(image)


@pytest.mark.parametrize('options, message', [
    # The kernels keep the values of the traits of a set in arrays of 4
    (dict(ntraits=5), "at most 4 traits"),
    (dict(rnodes=1), "at least two radial nodes"),
    # A thick disk multiplies each polar trait by its height trait
    (dict(heights=True, vpt=True), "needs the height traits"),
    (dict(monte_carlo=True), "Monte Carlo disk needs density height")])
def test_disks_are_checked(driver, options, message):
    # The kernels trust the arrays of a disk: what they cannot handle is
    # an error, not a read out of bounds
    assert disk_image(driver).any()
    with pytest.raises(ValueError, match=message):
        disk_image(driver, **options)
