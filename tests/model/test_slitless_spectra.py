"""
Tests for slitless spectra (slitless_spectra): the image of a grism,
where the light of each channel of a spectral cube of the sky lands moved
by the trace of the disperser.
"""

import copy

import gbkfit.params
import numpy as np
import pytest
from gbkfit.observation.observables.slitless_spectra import pixel_footprint
from modelutils import observation_group


GALAXY = dict(components=[dict(
    type='smdisk', loose=False, tilted=False,
    rnodes=list(range(0, 10)),
    density=dict(ptraits=dict(type='exponential')),
    kinematics=dict(
        vptraits=dict(type='arctan', direction='tan'),
        dptraits=dict(type='uniform')),
    emission=dict(type='lines'))])

PROPERTIES = dict(
    vsys=0, xpos=0.3, ypos=-0.6, posa=50, incl=60,
    pt_a=1, pt_s=3, vpt_rt=2, vpt_vt=150, dpt_a=20)

# A trace of one pixel along x per channel of 30 km/s at 4 um (a channel
# spans 4 um * 30 / c = 4e-4 um), centred on the image at the centre of
# the spectral axis
SIZE, NZ, STEP = (16, 12), 21, 30
SLITLESS = dict(
    type='slitless_spectra', size=list(SIZE), spec_size=NZ,
    spec_step=STEP, spec_rest='4 um', trace=dict(
        type='polynomial', wave='4 um', dx=[0, 2.99792458e5 / (4 * STEP)]))


def evaluate(driver, observable):
    group = observation_group([
        dict(driver=dict(type=driver.type()), galaxy=GALAXY,
             line_of_sight=dict(type='flat'), observable=observable)])
    params = gbkfit.params.ParamSpace(group.pdescs(), PROPERTIES)
    data = group.model_h(params.evaluate())[0]
    return next(iter(data.values()))['d'].copy()


def test_pixel_footprint_of_a_still_pixel_is_its_overlaps():
    p, q, weights = pixel_footprint((0.3, 0), (0.3, 0))
    assert dict(zip(zip(p, q), weights)) == pytest.approx(
        {(0, 0): 0.7, (1, 0): 0.3})


def test_pixel_footprint_of_a_pixel_moved_by_one_pixel():
    # A unit box spread over a unit segment is a triangle of base 2
    p, q, weights = pixel_footprint((-0.5, 0), (0.5, 0))
    assert dict(zip(zip(p, q), weights)) == pytest.approx(
        {(-1, 0): 1 / 8, (0, 0): 3 / 4, (1, 0): 1 / 8})


def test_pixel_footprint_of_a_slanted_segment_matches_sampling():
    start, end = (0.2, 0.1), (3.7, 0.9)
    p, q, weights = pixel_footprint(start, end)
    assert weights.sum() == pytest.approx(1)
    rng = np.random.default_rng(1)
    t, ux, uy = rng.random((3, 400_000))
    x = np.rint(start[0] + t * (end[0] - start[0]) + ux - 0.5)
    y = np.rint(start[1] + t * (end[1] - start[1]) + uy - 0.5)
    sampled = [np.mean((x == a) & (y == b)) for a, b in zip(p, q)]
    np.testing.assert_allclose(sampled, weights, atol=3e-3)


@pytest.mark.parametrize('psf', [None, dict(type='gauss', sigma=1.5)])
def test_slitless_spectra_are_the_cube_dispersed(driver, psf):
    # Channel m from the centre spans dx from m - 1/2 to m + 1/2, so each
    # pixel of it lands 1/8, 3/4 and 1/8 at m - 1, m and m + 1. The sky of
    # the model reaches (NZ - 1) / 2 + 1 pixels beyond the image on each
    # side along x: the cube of pixel_spectra on that sky, dispersed by
    # hand, is the image
    reach = (NZ - 1) // 2 + 1
    width, height = SIZE
    image = evaluate(driver, SLITLESS | dict(psf=psf))
    cube = evaluate(driver, dict(
        type='pixel_spectra', size=[width + 2 * reach, height, NZ],
        step=[1, 1, STEP],
        rpix=[(width - 1) / 2 + reach, (height - 1) / 2, (NZ - 1) / 2],
        psf=psf))
    expected = np.zeros(image.shape)
    for k in range(NZ):
        m = k - (NZ - 1) // 2
        for s, weight in ((-1, 1 / 8), (0, 3 / 4), (1, 1 / 8)):
            # (the detector pixel x gets the sky pixel x + reach - m - s)
            start = reach - m - s
            expected += weight * STEP * cube[k, :, start:start + width]
    assert image.shape == (height, width)
    np.testing.assert_allclose(
        image, expected, rtol=1e-4, atol=1e-5 * expected.max())


def test_slitless_spectra_objective_residual(driver):
    # Data equal to the model plus 1, with errors of 2: every residual
    # (model - data) / error is -0.5
    from gbkfit.dataset import Data, DatasetSlitlessSpectra
    from gbkfit.model import galaxy_parser
    from gbkfit.objective import Objective
    from gbkfit.observation import (
        Flat, Observation, ObservationGroup, observable_parser)
    model = evaluate(driver, SLITLESS)
    dataset = DatasetSlitlessSpectra(
        Data(model + 1, error=np.full_like(model, 2)))
    options = {k: v for k, v in SLITLESS.items() if k not in ('type', 'size')}
    observable = observable_parser.load(
        dict(type='slitless_spectra') | copy.deepcopy(options), dataset)
    observation = Observation(observable, Flat(), driver, data=dataset)
    group = ObservationGroup(
        [galaxy_parser.load(copy.deepcopy(GALAXY))], [observation])
    objective = Objective(group)
    params = gbkfit.params.ParamSpace(group.pdescs(), PROPERTIES)
    residual = objective.residual_nddata_h(params.evaluate(), False)
    np.testing.assert_allclose(residual[0]['spectra'], -0.5, rtol=1e-5)


def test_slitless_spectra_refuse_an_lsf_and_frequencies():
    with pytest.raises(RuntimeError, match="no LSF"):
        observation_group([dict(
            driver=dict(type='host'), galaxy=GALAXY,
            line_of_sight=dict(type='flat'),
            observable=SLITLESS | dict(lsf=dict(type='gauss', sigma=15)))])
    with pytest.raises(RuntimeError, match="rest wavelength"):
        observation_group([dict(
            driver=dict(type='host'), galaxy=GALAXY,
            line_of_sight=dict(type='flat'),
            observable=SLITLESS | dict(spec_rest='1420 MHz'))])


def test_grismconf_traces_follow_their_parameter():
    # The offsets of a wavelength are those of the t that gives it, of
    # polynomials of the place on the detector (from the origin of the
    # image): here w(t) = 4 + 0.1 t + 0.01 t^2 + 1e-5 x0 (um), dx(t) = 3 +
    # 0.01 x0 - 0.02 y0 + 50 t and dy(t) = 1e-3 x0 y0 t
    from gbkfit.observation.observables.traces import TraceGrismConf
    trace = TraceGrismConf(
        displ=[[4, 1e-5, 0], [0.1], [0.01]],
        dispx=[[3, 0.01, -0.02], [50]],
        dispy=[[0], [0, 0, 0, 0, 1e-3, 0]], origin=[100, 200])
    t = np.array([0.1, 0.5, 0.9])
    x, y = np.array([0.0, 10, 20]), np.array([5.0, 0, 30])
    x0, y0 = x + 100, y + 200
    wavelength = (4 + 0.1 * t + 0.01 * t ** 2 + 1e-5 * x0) * 1e-6
    dx, dy = trace.offsets(wavelength, x, y)
    np.testing.assert_allclose(dx, 3 + 0.01 * x0 - 0.02 * y0 + 50 * t,
                               rtol=1e-12)
    np.testing.assert_allclose(dy, 1e-3 * x0 * y0 * t, rtol=1e-10)


def test_grismconf_traces_constant_over_the_detector(driver):
    # A trace of GRISMCONF that is the same at every place is that of the
    # polynomials: w(t) = 4 + t (um) and dx(t) = c / (4 STEP) t
    grismconf = SLITLESS | dict(trace=dict(
        type='grismconf', displ=[[4], [1]],
        dispx=[[0], [2.99792458e5 / (4 * STEP)]], dispy=[[0]]))
    np.testing.assert_allclose(
        evaluate(driver, grismconf), evaluate(driver, SLITLESS),
        rtol=1e-9, atol=1e-12)


def test_traces_that_vary_across_the_detector_move_each_pixel():
    # Of a trace whose dx grows with the place, dx(t) = 0.05 x0 + 4 t, the
    # light of each pixel and channel lands about the middle of its
    # segment, at its own offset
    from gbkfit.observation.observables import traces
    from gbkfit.observation.observables.slitless_spectra import (
        SlitlessSpectra)
    trace = traces.TraceGrismConf(
        displ=[[4], [1e-4]], dispx=[[0, 0.05, 0], [4]], dispy=[[0]])
    observable = SlitlessSpectra(
        size=[40, 6], spec_step=30, spec_rest='4 um', trace=trace,
        spec_size=3)
    edges = observable._channel_edges()
    offset, size = observable._sky(edges)
    matrix = observable._dispersion(edges, size, offset).tocsc()
    nx, ny = size
    # (channels about -16 and +16 pixels, of places that land inside)
    for channel, x0, j in ((0, 25, 2), (2, 10, 3)):
        i = x0 - offset[0]
        column = matrix[:, [channel * nx * ny + j * nx + i]].toarray()[:, 0]
        pixels = np.flatnonzero(column)
        t = (edges[channel] * 1e6 - 4) / 1e-4
        middle = x0 + 0.05 * x0 + 4 * t.mean()
        np.testing.assert_allclose(
            np.average(pixels % 40, weights=column[pixels]), middle,
            atol=1e-9)
        np.testing.assert_allclose(column.sum(), 30, rtol=1e-12)


def test_slitless_spectra_have_amplitudes_per_row(driver):
    # Free amplitudes scale each line of the image along the dispersion
    # (the rows: the trace runs along x), as geko's row amplitudes: of data
    # that are the model times a scale for each row, of no noise but a
    # small error, the best amplitudes, (sum m d / e^2 + 1/s^2) / (sum m^2
    # / e^2 + 1/s^2), are those scales where the rows have light
    from gbkfit.dataset import Data, DatasetSlitlessSpectra
    from gbkfit.model import galaxy_parser
    from gbkfit.objective import Objective
    from gbkfit.observation import (
        Flat, LikelihoodGaussian, Observation, ObservationGroup,
        observable_parser)
    model = evaluate(driver, SLITLESS)
    scales = 1 + 0.3 * np.sin(np.arange(model.shape[0]))
    error = 1e-3
    dataset = DatasetSlitlessSpectra(Data(
        model * scales[:, None], error=np.full_like(model, error)))
    options = {k: v for k, v in SLITLESS.items() if k not in ('type', 'size')}
    observable = observable_parser.load(
        dict(type='slitless_spectra') | copy.deepcopy(options), dataset)
    assert observable.spectra_axis() == 1
    observation = Observation(
        observable, Flat(), driver, data=dataset, likelihood=(
            LikelihoodGaussian(amplitudes='per_spectrum',
                               amplitude_sigma=0.5)))
    group = ObservationGroup(
        [galaxy_parser.load(copy.deepcopy(GALAXY))], [observation])
    objective = Objective(group)
    params = gbkfit.params.ParamSpace(group.pdescs(), PROPERTIES)
    objective.residual_nddata_h(params.evaluate(), False)
    amplitudes = objective.amplitudes()[0]
    a = (model ** 2).sum(axis=1) / error ** 2 + 1 / 0.5 ** 2
    b = (model * dataset['spectra'].data()).sum(axis=1) / error ** 2 \
        + 1 / 0.5 ** 2
    assert amplitudes.shape == (model.shape[0],)
    np.testing.assert_allclose(amplitudes, b / a, rtol=1e-4)
    bright = model.sum(axis=1) > 0.1 * model.sum(axis=1).max()
    np.testing.assert_allclose(amplitudes[bright], scales[bright], rtol=1e-3)
