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


def residual_sum(driver, values, squared):
    memory = Memory(driver)
    objective = driver.native_class('Objective', DTYPE)()
    total = memory.to_device(np.zeros(1, DTYPE))
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
    # Accumulated in float32, the small terms would be lost next to the
    # large one (float32 numbers near 1e8 are 8 apart)
    values = np.ones(1000)
    values[0] = 1e8
    assert residual_sum(driver, values, False) == np.float32(1e8 + 999)


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
        dispersion=memory.to_device(np.zeros(cube_shape, DTYPE)))

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
        image=outputs['image'], scube=outputs['scube'],
        vdata_cmp=outputs['velocity'], ddata_cmp=outputs['dispersion'])
    return {name: memory.to_host(array) for name, array in outputs.items()}


def test_smdisk_thin(driver, ndarrays_regression):
    check_against_reference(
        ndarrays_regression, 'smdisk_thin',
        evaluate_smdisk(driver, thick=False))


def test_smdisk_thick(driver, ndarrays_regression):
    check_against_reference(
        ndarrays_regression, 'smdisk_thick',
        evaluate_smdisk(driver, thick=True))
