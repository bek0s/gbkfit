"""
Tests for the geometry and velocity conventions of the disk models.
"""

import numpy as np
import pytest


def kinematics_2d_model(driver, **component):
    """A pixel_spectra model with one thin smooth disk component."""
    return dict(
        driver=dict(type=driver.type()),
        dmodel=dict(type='pixel_spectra', size=[32, 32, 41], step=[1, 1, 10]),
        gmodel=dict(type='kinematics_2d', components=[dict(
            type='smdisk',
            rnodes=list(range(0, 12)),
            bptraits=dict(type='exponential'),
            vptraits=dict(type='tan_uniform'),
            dptraits=dict(type='uniform'),
            **component)]))


def test_loose_disk_systemic_velocity(driver, evaluate_models):
    # A loose disk without rotation: the line-of-sight velocity of the
    # disk must be the systemic velocity everywhere, regardless of the
    # (also node-wise) centre position.
    model = kinematics_2d_model(driver, loose=True, tilted=False)
    properties = dict(
        vsys=50, xpos=3, ypos=-2, posa=0, incl=45,
        bpt_a=1, bpt_s=4, vpt_vt=0, dpt_a=10)
    _, extra = evaluate_models([model], properties)
    velocity = extra['observation0_gmodel_component0_vdata'].data
    on_disk = np.isfinite(velocity)
    assert on_disk.sum() > 100
    np.testing.assert_allclose(velocity[on_disk], 50)


def kinematics_3d_model(driver, disk, rota, psf=None):
    """
    An scube model with one thick disk, on a grid rotated by rota, and
    with the given PSF.
    """
    component = dict(
        type=disk, loose=False, tilted=False,
        rnodes=list(range(0, 16, 2)),
        bptraits=dict(type='exponential'),
        bhtraits=dict(type='sech2'),
        vptraits=dict(type='tan_arctan'),
        dptraits=dict(type='uniform'))
    if disk == 'mcdisk':
        component['cflux'] = 2e-5
    return dict(
        driver=dict(type=driver.type()),
        dmodel=dict(
            type='pixel_spectra', size=[33, 33, 41], step=[1, 1, 10], rota=rota,
            psf=psf),
        gmodel=dict(type='kinematics_3d', components=[component]))


@pytest.mark.parametrize('disk', ['smdisk', 'mcdisk'])
def test_grid_rotation(driver, evaluate_models, disk):
    # rota rotates the grid on the sky, counterclockwise like a position
    # angle, so a grid rotated by rota sees a centred disk with position
    # angle posa like an unrotated grid sees one with posa - rota
    properties = dict(
        vsys=0, xpos=0, ypos=0, incl=60,
        bpt_a=1, bpt_s=4, bht_s=1, vpt_rt=2, vpt_vt=150, dpt_a=20)

    def evaluate(posa, rota):
        data, _ = evaluate_models(
            [kinematics_3d_model(driver, disk, rota)],
            properties | dict(posa=posa))
        return data[0]['spectra']['d'].copy()

    rotated = evaluate(posa=70, rota=40)
    expected = evaluate(posa=30, rota=0)
    # The smooth disk is sampled at the same points; the clouds of the
    # Monte Carlo disk can land in a neighbouring pixel due to rounding
    tolerance = 1e-5 if disk == 'smdisk' else 2e-3
    difference = np.linalg.norm(rotated - expected) / np.linalg.norm(expected)
    assert difference < tolerance


def test_grid_rotation_with_an_elongated_beam(driver, evaluate_models):
    # The position angle of the PSF is on the sky too, so a grid rotated
    # by rota sees a beam with position angle posa like an unrotated grid
    # sees one with posa - rota
    properties = dict(
        vsys=0, xpos=0, ypos=0, incl=60,
        bpt_a=1, bpt_s=4, bht_s=1, vpt_rt=2, vpt_vt=150, dpt_a=20)

    def evaluate(posa, beam_posa, rota):
        beam = dict(type='gauss', sigma=1.5, ratio=0.4, posa=beam_posa)
        data, _ = evaluate_models(
            [kinematics_3d_model(driver, 'smdisk', rota, beam)],
            properties | dict(posa=posa))
        return data[0]['spectra']['d'].copy()

    rotated = evaluate(posa=70, beam_posa=50, rota=40)
    expected = evaluate(posa=30, beam_posa=10, rota=0)
    difference = np.linalg.norm(rotated - expected) / np.linalg.norm(expected)
    assert difference < 1e-5


def test_image_psf_matches_the_analytic_psf(driver, evaluate_models):
    # A PSF image sampled from an analytic PSF gives the same model as
    # that PSF, on a grid that is not square (an image PSF used to be
    # transposed, and off-centre by half a pixel)
    from astropy.io import fits
    from gbkfit.instrument import PSFGauss
    gauss = dict(type='gauss', sigma=1.5, ratio=0.6, posa=30)
    image = PSFGauss(1.5, 0.6, 30).asarray((1, 1), (21, 21))
    fits.writeto('psf.fits', image)
    properties = dict(
        vsys=0, xpos=0, ypos=0, posa=60, incl=60,
        bpt_a=1, bpt_s=4, bht_s=1, vpt_rt=2, vpt_vt=150, dpt_a=20)

    def evaluate(psf):
        model = kinematics_3d_model(driver, 'smdisk', 0, psf)
        # The convolution pads the grid to 128 x 64
        model['dmodel']['size'] = [61, 17, 41]
        data, _ = evaluate_models([model], properties)
        return data[0]['spectra']['d'].copy()

    expected = evaluate(gauss)
    actual = evaluate(dict(type='image', file='psf.fits'))
    difference = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert difference < 1e-3


def uniform_disk_flux(driver, evaluate_models, rnodes, loose):
    """The flux of a face-on uniform disk of brightness 1 per arcsec^2."""
    nodes = len(rnodes)
    model = dict(
        driver=dict(type=driver.type()),
        dmodel=dict(type='pixel_brightness', size=[80, 80], step=[0.2, 0.2]),
        gmodel=dict(type='intensity_2d', components=[dict(
            type='smdisk', loose=loose, tilted=False, rnodes=rnodes, rstep=1,
            bptraits=dict(type='uniform'))]))
    centre = [0] * nodes if loose else 0
    properties = dict(xpos=centre, ypos=centre, posa=0, incl=0, bpt_a=1)
    data, _ = evaluate_models([model], properties)
    return data[0]['brightness']['d'].sum() * 0.2 * 0.2


@pytest.mark.parametrize('loose', [False, True])
@pytest.mark.parametrize('rnodes', [[0, 3, 6.7], [0.5, 3, 6]])
def test_disk_ends_at_its_last_node(driver, evaluate_models, rnodes, loose):
    # The rings of the disk (of width up to rstep = 1) cover exactly the
    # radii from the first node to the last one, even when that range is
    # not a whole number of rsteps
    flux = uniform_disk_flux(driver, evaluate_models, rnodes, loose)
    area = np.pi * (rnodes[-1] ** 2 - rnodes[0] ** 2)
    assert flux == pytest.approx(area, rel=0.01)


@pytest.mark.parametrize('incl', [90, 120])
def test_thin_disk_seen_from_below(driver, evaluate_models, incl):
    # A thin disk is seen from below at inclinations above 90 degrees:
    # an axisymmetric one looks as it does at 180 - incl. At 90 degrees
    # (edge-on), its light is on its major axis, and not negative (in
    # float32 the cosine of 90 degrees is negative). The edge of the disk
    # is between pixels, which the rounding of the cosines would move.
    model = dict(
        driver=dict(type=driver.type()),
        dmodel=dict(type='pixel_brightness', size=[33, 33]),
        gmodel=dict(type='intensity_2d', components=[dict(
            type='smdisk', loose=False, tilted=False,
            rnodes=[1.05 * i for i in range(14)],
            bptraits=dict(type='exponential'))]))

    def image(incl):
        properties = dict(
            xpos=0, ypos=0, posa=0, incl=incl, bpt_a=1, bpt_s=4)
        data, _ = evaluate_models([model], properties)
        return data[0]['brightness']['d']
    below = image(incl)
    assert (below >= 0).all()
    if incl != 90:
        np.testing.assert_allclose(
            below, image(180 - incl), rtol=1e-5, atol=1e-6 * below.max())


def test_loose_disk_centres_its_rings(driver, evaluate_models):
    # A face-on loose thin disk whose centre moves with radius, xpos =
    # 0.2 r: the ring of radius r is centred at 0.2 r, so the point at x
    # on the x axis is on the ring x / 1.2 (x > 0) or -x / 0.8 (x < 0),
    # where the image has the brightness of that radius
    rnodes = list(range(0, 13))
    model = dict(
        driver=dict(type=driver.type()),
        dmodel=dict(type='pixel_brightness', size=[41, 41], step=[0.5, 0.5]),
        gmodel=dict(type='intensity_2d', components=[dict(
            type='smdisk', loose=True, tilted=False, rnodes=rnodes,
            bptraits=dict(type='exponential'))]))
    properties = dict(
        xpos=[0.2 * r for r in rnodes], ypos=[0] * len(rnodes),
        posa=0, incl=0, bpt_a=1, bpt_s=4)
    data, _ = evaluate_models([model], properties)
    row = data[0]['brightness']['d'][20]
    x = (np.arange(41) - 20) * 0.5
    radius = np.where(x > 0, x / 1.2, -x / 0.8)
    inside = radius < 11
    np.testing.assert_allclose(
        row[inside], np.exp(-radius[inside] / 4), rtol=1e-3)
