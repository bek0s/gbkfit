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
from modelutils import group_plan, observation_group


GALAXY = dict(components=[dict(
    type='disk', loose=False, tilted=False,
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


def evaluate(driver, observable, properties=PROPERTIES):
    plan = group_plan([
        dict(driver=dict(type=driver.type()), galaxy=GALAXY,
             line_of_sight=dict(type='flat'),
             observable=copy.deepcopy(observable))])
    params = gbkfit.params.ParamSpace(plan.group().pdescs(), properties)
    data = plan.model_h(params.evaluate())[0]
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


def test_pixel_footprint_of_a_box_matches_sampling():
    # (a part of a pixel: e.g. the part of a pixel of the sky open in a
    # shutter)
    start, end, size = (0.2, 0.1), (3.7, 0.9), (0.3, 0.6)
    p, q, weights = pixel_footprint(start, end, size)
    assert weights.sum() == pytest.approx(1)
    rng = np.random.default_rng(1)
    t, ux, uy = rng.random((3, 400_000))
    x = np.rint(start[0] + t * (end[0] - start[0]) + (ux - 0.5) * size[0])
    y = np.rint(start[1] + t * (end[1] - start[1]) + (uy - 0.5) * size[1])
    sampled = [np.mean((x == a) & (y == b)) for a, b in zip(p, q)]
    np.testing.assert_allclose(sampled, weights, atol=3e-3)


def test_pixel_footprint_on_resampled_pixels_matches_sampling():
    # Pixels resampled by drizzle (of a square kernel of a pixel) take the
    # light of each pixel of the detector by overlap: that of the box moved
    # by a uniform offset of up to half a pixel more
    start, end, size = (0.2, 0.1), (3.7, 0.9), (0.3, 0.6)
    p, q, weights = pixel_footprint(start, end, size, 'triangle')
    assert weights.sum() == pytest.approx(1)
    rng = np.random.default_rng(2)
    t, ux, uy, vx, vy = rng.random((5, 400_000))
    x = np.rint(start[0] + t * (end[0] - start[0]) + (ux - 0.5) * size[0]
                + vx - 0.5)
    y = np.rint(start[1] + t * (end[1] - start[1]) + (uy - 0.5) * size[1]
                + vy - 0.5)
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
        Flat, ObservationGroup, ObservationSky, observable_parser)
    model = evaluate(driver, SLITLESS)
    dataset = DatasetSlitlessSpectra(
        Data(model + 1, error=np.full_like(model, 2)))
    options = {k: v for k, v in SLITLESS.items() if k not in ('type', 'size')}
    observable = observable_parser.load(
        dict(type='slitless_spectra') | copy.deepcopy(options), dataset)
    observation = ObservationSky(observable, Flat(), data=dataset)
    group = ObservationGroup(
        [galaxy_parser.load(copy.deepcopy(GALAXY))], [observation])
    objective = Objective(group.plan(driver))
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


def test_grismconf_traces_from_files(tmp_path):
    # A GRISMCONF file (of two orders, with comments and other keys) gives
    # the trace of its coefficients, of the order asked
    from gbkfit.observation.observables.traces import (
        TraceGrismConf, trace_parser)
    path = tmp_path / 'grism.conf'
    path.write_text(
        "# Wavelength unit: Micron\nNAXIS 2048 2048\n\nBEAM_+1\n"
        "DISPL_+1_0 4 1e-5 0\nDISPL_+1_1 0.1\nDISPL_+1_2 0.01\n"
        "DISPX_+1_0 3 0.01 -0.02\nDISPX_+1_1 50\n"
        "DISPY_+1_0 0\nDISPY_+1_1 0 0 0 0 1e-3 0\n"
        "SENSITIVITY_+1 sens.fits\nXRANGE_+1 -125.00 125.00\n\n"
        "BEAM_+2\nDISPL_+2_0 2\nDISPL_+2_1 1\nDISPX_+2_0 0\n"
        "DISPX_+2_1 99\nDISPY_+2_0 0\n")
    trace = trace_parser.load(dict(
        type='grismconf', file=str(path), origin=[100, 200]))
    expected = TraceGrismConf(
        displ=[[4, 1e-5, 0], [0.1], [0.01]],
        dispx=[[3, 0.01, -0.02], [50]],
        dispy=[[0], [0, 0, 0, 0, 1e-3, 0]], origin=[100, 200])
    wavelength, x, y = np.array([4.05e-6, 4.1e-6]), [0.0, 10], [5.0, 30]
    np.testing.assert_allclose(
        trace.offsets(wavelength, x, y), expected.offsets(wavelength, x, y))
    second = TraceGrismConf.read(str(path), order=2)
    np.testing.assert_allclose(second.offsets(2.5e-6, 0.0, 0.0), [49.5, 0])
    with pytest.raises(RuntimeError, match="no order '-1'"):
        TraceGrismConf.read(str(path), order=-1)
    path.write_text("# Wavelength unit: Angstrom\nBEAM_+1\n")
    with pytest.raises(RuntimeError, match="not micron"):
        TraceGrismConf.read(str(path))


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


def test_slitless_spectra_have_flux_scales_per_row(driver):
    # Solved flux scales per row scale each line of the image along the
    # dispersion (the rows: the trace runs along x), as geko's row
    # amplitudes: of data that are the model times a scale for each row,
    # of no noise but a small error, the best scales, (sum m d / e^2 +
    # 1/s^2) / (sum m^2 / e^2 + 1/s^2), are those scales where the rows
    # have light
    from gbkfit.dataset import Data, DatasetSlitlessSpectra
    from gbkfit.model import galaxy_parser
    from gbkfit.objective import Objective
    from gbkfit.observation import (
        Flat, ObservationGroup, ObservationSky, observable_parser)
    model = evaluate(driver, SLITLESS)
    scales = 1 + 0.3 * np.sin(np.arange(model.shape[0]))
    error = 1e-3
    dataset = DatasetSlitlessSpectra(Data(
        model * scales[:, None], error=np.full_like(model, error)))
    options = {k: v for k, v in SLITLESS.items() if k not in ('type', 'size')}
    observable = observable_parser.load(
        dict(type='slitless_spectra') | copy.deepcopy(options), dataset)
    assert observable.flux_scales()['per_row'].axis == 1
    observation = ObservationSky(
        observable, Flat(), data=dataset, flux_scale='per_row', name='grism')
    group = ObservationGroup(
        [galaxy_parser.load(copy.deepcopy(GALAXY))], [observation])
    params = gbkfit.params.ParamSpace(group.pdescs(), PROPERTIES | dict(
        grism_flux_scale=dict(solve=dict(
            type='gaussian', mean=1, sigma=0.5, minimum=0))))
    objective = Objective(group.plan(driver), params.solves())
    found = np.array(list(
        objective.evaluate(params.evaluate()).solved.values()))
    a = (model ** 2).sum(axis=1) / error ** 2 + 1 / 0.5 ** 2
    b = (model * dataset['spectra'].data()).sum(axis=1) / error ** 2 \
        + 1 / 0.5 ** 2
    assert found.shape == (model.shape[0],)
    np.testing.assert_allclose(found, b / a, rtol=1e-4)
    bright = model.sum(axis=1) > 0.1 * model.sum(axis=1).max()
    np.testing.assert_allclose(found[bright], scales[bright], rtol=1e-3)


def disperse_by_hand(cube, rows, column, width):
    # The light of the sky pixels (rows, column) of a cube on the grid of
    # the image, dispersed as in test_slitless_spectra_are_the_cube_dispersed
    image = np.zeros((cube.shape[1], width))
    for k in range(NZ):
        m = k - (NZ - 1) // 2
        for s, weight in ((-1, 1 / 8), (0, 3 / 4), (1, 1 / 8)):
            if 0 <= column + m + s < width:
                image[rows, column + m + s] += \
                    weight * STEP * cube[k, rows, column]
    return image


def test_shutters_let_through_the_light_of_their_sky(driver):
    # Two shutters over the column 5 of the image (x from -3 to -2 arcsec
    # of its centre, 7.5), over the rows 3-4 and 6-7, with a bar over the
    # row 5: the image is the light of the sky under them, dispersed. And
    # a shutter over all the sky lets all of it through
    width, height = SIZE
    shutters = SLITLESS | dict(aperture=dict(
        type='shutters', size=[1, 2], pitch=3, count=2,
        centre=[-2.5, -0.5]))
    cube = evaluate(driver, dict(
        type='pixel_spectra', size=[width, height, NZ], step=[1, 1, STEP]))
    np.testing.assert_allclose(
        evaluate(driver, shutters),
        disperse_by_hand(cube, [3, 4, 6, 7], 5, width),
        rtol=1e-4, atol=1e-6 * cube.max())
    everything = SLITLESS | dict(aperture=dict(
        type='shutters', size=[1000, 1000]))
    np.testing.assert_allclose(
        evaluate(driver, everything), evaluate(driver, SLITLESS),
        rtol=1e-6, atol=1e-9)


def test_a_finer_sky_places_the_light_within_its_pixels(driver):
    # Of the same light (the model on 6 x 6 points of each pixel), the sky
    # sampled by 1, 3 and 6 pixels for each pixel of the image: the light
    # within the pixels moves along the trace with them, so the images
    # converge, with the same light in total
    images = [
        evaluate(driver, SLITLESS | dict(
            oversampling=[6 // k] * 2, sky_sampling=k))
        for k in (1, 3, 6)]
    np.testing.assert_allclose(
        [image.sum() for image in images], images[2].sum(), rtol=1e-3)
    errors = [np.abs(image - images[2]).max() for image in images[:2]]
    assert errors[1] < 0.2 * errors[0]


def test_nods_subtract_the_other_exposures(driver):
    # Three exposures of the sky moved by 0 and +-3 pixels along the slit,
    # of weights 1, -1/2 and -1/2: the image of the galaxy moved likewise
    shutters = SLITLESS | dict(aperture=dict(
        type='shutters', size=[3, 2], pitch=3, count=3))
    nods = shutters | dict(exposures=[
        dict(weight=1), dict(sky_offset=[0, 3], weight=-0.5),
        dict(sky_offset=[0, -3], weight=-0.5)])

    def moved(dy):
        return evaluate(driver, shutters, PROPERTIES | dict(
            ypos=PROPERTIES['ypos'] + dy))
    expected = moved(0) - (moved(3) + moved(-3)) / 2
    np.testing.assert_allclose(
        evaluate(driver, nods), expected, rtol=1e-5,
        atol=1e-6 * np.abs(expected).max())
    # The image of an exposure can be moved too (one row up)
    up = evaluate(driver, SLITLESS | dict(
        exposures=dict(image_offset=[0, 1])))
    np.testing.assert_allclose(
        up[1:], evaluate(driver, SLITLESS)[:-1], rtol=1e-5, atol=1e-9)


def test_an_aperture_lets_the_spectrograph_blur_the_spectra(driver):
    # Behind an aperture, the LSF of the instrument blurs the spectra along
    # the trace, keeping their light
    shutter = SLITLESS | dict(aperture=dict(type='shutters', size=[3, 6]))
    sharp = evaluate(driver, shutter)
    blurred = evaluate(driver, shutter | dict(
        lsf=dict(type='gauss', sigma=40)))
    assert blurred.sum() == pytest.approx(sharp.sum(), rel=1e-3)
    assert blurred.max() < 0.9 * sharp.max()


def test_resampled_spectra_spread_the_light_by_a_pixel_more(driver):
    # (of the same light in total, of a galaxy within the image)
    compact = PROPERTIES | dict(pt_s=1, vpt_vt=60)
    plain = evaluate(driver, SLITLESS, compact)
    resampled = evaluate(driver, SLITLESS | dict(resampled=True), compact)
    assert resampled.sum() == pytest.approx(plain.sum(), rel=1e-3)
    assert resampled.max() < plain.max()


def test_apertures_and_exposures_from_configurations():
    from gbkfit.observation import observable_parser
    info = SLITLESS | dict(
        aperture=dict(type='shutters', size=[0.2, 0.46], pitch=0.53,
                      count=3, centre=[0.1, 0]),
        exposures=[dict(), dict(sky_offset=[0, 0.53], weight=-1)],
        sky_sampling=4, resampled=True)
    observable = observable_parser.load(copy.deepcopy(info))
    assert observable.aperture().rectangles().shape == (3, 4)
    assert [e.weight() for e in observable.exposures()] == [1, -1]
    dumped = observable_parser.dump(observable)
    assert observable_parser.dump(observable_parser.load(dumped)) == dumped
    with pytest.raises(RuntimeError, match="pitch"):
        observable_parser.load(SLITLESS | dict(aperture=dict(
            type='shutters', size=[0.2, 0.46], count=3)))
    with pytest.raises(RuntimeError, match="sky_sampling"):
        observable_parser.load(SLITLESS | dict(sky_sampling=0))


def polynomial_table(wave, x, y, dy=0.0):
    # The trace of SLITLESS (dx = c / (4 STEP) per um about 4 um), sampled
    # at nodes, with a dy of dy x y
    from gbkfit.observation.observables.traces import TraceTable
    w, yy, xx = np.meshgrid(wave, y, x, indexing='ij')
    dx = 2.99792458e5 / (4 * STEP) * (w - 4e-6) / 1e-6
    return TraceTable(wave, x, y, dx, dy * xx * yy)


def test_trace_tables_interpolate_their_nodes():
    wave = np.linspace(3.99e-6, 4.01e-6, 9)
    table = polynomial_table(
        wave, np.arange(-5.0, 25), np.arange(-5.0, 20), dy=1e-3)
    # (at the nodes, between them, and beyond the places: those of the edge)
    dx, dy = table.offsets(wave[[2, 2]], [3.0, 3.5], [4.0, 4.5])
    np.testing.assert_allclose(
        dx, 2.99792458e5 / (4 * STEP) * (wave[2] - 4e-6) / 1e-6, atol=1e-9)
    np.testing.assert_allclose(dy, [1e-3 * 12, 1e-3 * 3.5 * 4.5], rtol=1e-6)
    np.testing.assert_allclose(
        table.offsets(4e-6, 100.0, -100.0)[1], 1e-3 * 24 * -5, rtol=1e-9)
    with pytest.raises(RuntimeError, match="covers the wavelengths"):
        table.offsets(4.02e-6, 0.0, 0.0)


def test_trace_tables_disperse_as_their_traces(driver, tmp_path):
    # A table of the trace of SLITLESS, from a file, gives its image
    from gbkfit.observation import observable_parser
    edges = 4e-6 * (1 + np.array([-1, 1]) * STEP * (NZ / 2 + 1) / 2.99792458e5)
    table = polynomial_table(
        np.linspace(*edges, 12), np.arange(-15.0, 31), np.arange(-3.0, 15))
    table.write(str(tmp_path / 'trace.fits'))
    tabled = SLITLESS | dict(trace=dict(
        type='table', file=str(tmp_path / 'trace.fits')))
    expected = evaluate(driver, SLITLESS)
    np.testing.assert_allclose(
        evaluate(driver, tabled), expected, rtol=1e-5,
        atol=1e-6 * expected.max())
    observable = observable_parser.load(copy.deepcopy(tabled))
    dumped = observable_parser.dump(observable, prefix=str(tmp_path / 'd_'))
    assert dumped['trace']['file'] == str(tmp_path / 'd_trace.fits')
    again = observable_parser.load(dumped)
    np.testing.assert_allclose(
        again.trace().offsets(4e-6, 3.0, 4.0),
        observable.trace().offsets(4e-6, 3.0, 4.0))
