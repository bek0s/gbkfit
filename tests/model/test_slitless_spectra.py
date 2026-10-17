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
    spec_step=STEP, spec_rest='4 um', trace_wave='4 um',
    trace_dx=[0, 2.99792458e5 / (4 * STEP)])


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
