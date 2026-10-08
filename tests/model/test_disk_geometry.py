"""
Tests for the geometry and velocity conventions of the disk models.
"""

import numpy as np
import pytest


def kinematics_2d_model(driver, **component):
    """An scube model with one thin smooth disk component."""
    return dict(
        driver=dict(type=driver.type()),
        dmodel=dict(type='scube', size=[32, 32, 41], step=[1, 1, 10]),
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
    velocity = extra['model0_gmodel_component0_vdata']
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
            type='scube', size=[33, 33, 41], step=[1, 1, 10], rota=rota,
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
        return data[0]['scube']['d'].copy()

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
        return data[0]['scube']['d'].copy()

    rotated = evaluate(posa=70, beam_posa=50, rota=40)
    expected = evaluate(posa=30, beam_posa=10, rota=0)
    difference = np.linalg.norm(rotated - expected) / np.linalg.norm(expected)
    assert difference < 1e-5


def test_image_psf_matches_the_analytic_psf(driver, evaluate_models):
    # A PSF image sampled from an analytic PSF gives the same model as
    # that PSF, on a grid that is not square (an image PSF used to be
    # transposed, and off-centre by half a pixel)
    from astropy.io import fits
    from gbkfit.psflsf.psfs import PSFGauss
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
        return data[0]['scube']['d'].copy()

    expected = evaluate(gauss)
    actual = evaluate(dict(type='image', data='psf.fits'))
    difference = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert difference < 1e-3
