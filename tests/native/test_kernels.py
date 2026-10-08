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
    fft = driver.backends().fft(DTYPE)
    shape = (16, 24, 32)
    data = smooth_cube(shape)
    data_r = memory.to_device(data)
    data_c = memory.to_device(
        np.zeros(fft.fft_complex_shape(list(shape)), np.complex64))
    fft.fft_r2c(data_r, data_c)
    fft.fft_c2r(data_c, data_r)
    result = memory.to_host(data_r) / data.size
    np.testing.assert_allclose(result, data, rtol=1e-5, atol=1e-5)


def test_fft_convolve(driver, ndarrays_regression):
    memory = Memory(driver)
    fft = driver.backends().fft(DTYPE)
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
    dmodel = driver.backends().dmodel(DTYPE)
    cube_hi = memory.to_device(smooth_cube((40, 60, 80)))
    cube_lo = memory.to_device(np.zeros((20, 20, 20), DTYPE))
    dmodel.dcube_downscale((4, 3, 2), (0, 0, 0), cube_hi, cube_lo)
    check_against_reference(
        ndarrays_regression, 'dcube_downscale',
        dict(result=memory.to_host(cube_lo)))


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
    objective = driver.backends().objective(DTYPE)
    inputs = {k: memory.to_device(v) for k, v in _residual_inputs().items()}
    residual = memory.to_device(np.zeros(1000, DTYPE))
    objective.residual(**inputs, weight=0.7, res=residual)
    check_against_reference(
        ndarrays_regression, 'residual',
        dict(result=memory.to_host(residual)))


def test_residual_sum(driver, request):
    if driver.type() == 'cuda':
        request.applymarker(pytest.mark.xfail(
            raises=AttributeError,
            reason="residual_sum is not bound in the cuda module yet"))
    memory = Memory(driver)
    objective = driver.backends().objective(DTYPE)
    values = np.linspace(-1, 1, 1000, dtype=DTYPE)
    total = memory.to_device(np.zeros(1, DTYPE))
    objective.residual_sum(memory.to_device(values), True, total)
    assert memory.to_host(total)[0] == pytest.approx(
        np.sum(values ** 2), rel=1e-5)


def _trait_args(kind, memory, trait=None, params=()):
    """
    Pack a single trait into the five arrays the native code expects,
    as keyword arguments named after the trait kind (e.g. 'rpt_uids').
    Without a trait, all five arrays are None.
    """
    arrays = (None,) * 5
    if trait is not None:
        consts = [float(c) for c in trait.consts()]
        arrays = (
            np.array([trait.uid()], np.int32),
            np.array(consts, DTYPE), np.array([len(consts)], np.int32),
            np.array(params, DTYPE), np.array([len(params)], np.int32))
    names = ('uids', 'cvalues', 'ccounts', 'pvalues', 'pcounts')
    return {
        f'{kind}_{name}': memory.to_device(array)
        for name, array in zip(names, arrays)}


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

    # A disk without height traits is evaluated as a thin disk
    brightness_height = traits.BHTraitSech2() if thick else None
    velocity_height = traits.VHTraitOne() if thick else None
    dispersion_height = traits.DHTraitOne() if thick else None

    gmodel = driver.backends().gmodel(DTYPE)
    gmodel.smdisk_evaluate(
        loose=False, tilted=False,
        rnodes=memory.to_device(np.linspace(0, 20, 21, dtype=DTYPE)),
        vsys=scalar(10), xpos=scalar(0.5), ypos=scalar(-1.0),
        posa=scalar(30), incl=scalar(50),
        **_trait_args(
            'rpt', memory, traits.BPTraitExponential(), (1.0, 5.0)),
        **_trait_args('rht', memory, brightness_height, (1.0,)),
        **_trait_args(
            'vpt', memory, traits.VPTraitTanArctan(), (3.0, 200.0)),
        **_trait_args('vht', memory, velocity_height),
        **_trait_args(
            'dpt', memory, traits.DPTraitUniform(), (25.0,)),
        **_trait_args('dht', memory, dispersion_height),
        **_trait_args('zpt', memory),
        **_trait_args('spt', memory),
        **_trait_args('wpt', memory),
        opacity=None,
        spat_size=spat_size,
        spat_step=(1.0, 1.0, 1.0),
        spat_zero=(-24.0, -24.0, -(spat_size[2] // 2)),
        spat_rota=0,
        spec_size=spec_size, spec_step=10.0, spec_zero=-300.0,
        image=outputs['image'], scube=outputs['scube'],
        wdata=None, wdata_cmp=None,
        rdata=None, rdata_cmp=None,
        ordata=None, ordata_cmp=None,
        vdata_cmp=outputs['velocity'], ddata_cmp=outputs['dispersion'])
    return {name: memory.to_host(array) for name, array in outputs.items()}


def test_smdisk_thin(driver, ndarrays_regression):
    check_against_reference(
        ndarrays_regression, 'smdisk_thin',
        evaluate_smdisk(driver, thick=False))


def test_smdisk_thick(driver, ndarrays_regression, request):
    if driver.type() == 'cuda':
        request.applymarker(pytest.mark.xfail(
            reason="known bug: the cuda thick disk only evaluates the "
                   "z=0 slice (it launches nx*ny threads, not nx*ny*nz)"))
    check_against_reference(
        ndarrays_regression, 'smdisk_thick',
        evaluate_smdisk(driver, thick=True))
