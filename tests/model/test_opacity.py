"""
Tests for opacity components: they absorb the light of the components
behind them along the line of sight.
"""

import copy

import gbkfit.model
import gbkfit.params
import numpy as np
import pytest
from modelutils import observation_group


DISK = dict(
    type='smdisk', loose=False, tilted=False, rnodes=list(range(0, 12)))

PROPERTIES = dict(
    vsys=0, xpos=0, ypos=0, posa=30, incl=60,
    bpt_a=1, bpt_s=4, bht_s=1, vpt_rt=2, vpt_vt=150, dpt_a=20,
    ocmp_xpos=0, ocmp_ypos=0, ocmp_posa=30, ocmp_incl=60,
    ocmp_opt_s=4, ocmp_oht_s=1)


def kinematics_3d_model(driver, with_opacity):
    gmodel = dict(type='kinematics_3d', components=[dict(
        **DISK,
        bptraits=dict(type='exponential'),
        bhtraits=dict(type='sech2'),
        vptraits=dict(type='tan_arctan'),
        dptraits=dict(type='uniform'))])
    if with_opacity:
        gmodel['opacity_components'] = [dict(
            **DISK,
            optraits=dict(type='exponential'),
            ohtraits=dict(type='sech2'))]
    return dict(
        driver=dict(type=driver.type()),
        dmodel=dict(type='scube', size=[32, 32, 41], step=[1, 1, 10]),
        gmodel=gmodel)


def evaluate(driver, with_opacity, opacity, times=1):
    """Evaluate a thick disk (with or without opacity) several times."""
    model_group = observation_group([kinematics_3d_model(driver, with_opacity)])
    properties = dict(PROPERTIES, ocmp_opt_a=opacity)
    if not with_opacity:
        properties = {
            k: v for k, v in properties.items() if not k.startswith('ocmp')}
    params = gbkfit.params.EvaluationParams(model_group.pdescs(), properties)
    values = params.evaluate()
    return [
        copy.deepcopy(model_group.model_h(values)[0]['scube']['d'])
        for _ in range(times)]


def assert_same_model(actual, desired):
    """
    Compare two model cubes to float32 precision. Thick disks are not
    bitwise reproducible on the host, because the threads add to the
    cube in a different order every time.
    """
    np.testing.assert_allclose(
        actual, desired, rtol=1e-5, atol=1e-6 * np.abs(desired).max())


def test_zero_opacity_has_no_effect(driver):
    without, = evaluate(driver, with_opacity=False, opacity=0)
    with_zero, = evaluate(driver, with_opacity=True, opacity=0)
    assert_same_model(with_zero, without)


def test_opacity_absorbs_flux(driver):
    without, = evaluate(driver, with_opacity=False, opacity=0)
    absorbed, = evaluate(driver, with_opacity=True, opacity=0.05)
    assert absorbed.sum() < 0.99 * without.sum()


def test_opacity_does_not_accumulate(driver):
    first, second = evaluate(
        driver, with_opacity=True, opacity=0.05, times=2)
    assert_same_model(second, first)


def face_on_optical_depth(
        driver, evaluate_models, disk_type,
        optraits=(dict(type='exponential'),),
        opacity_properties=dict(ocmp_opt_a=0.5, ocmp_opt_s=4, ocmp_oht_s=1)):
    """
    The optical depth of each line of sight through a face-on opacity disk
    of the given type and opacity traits (each with a sech2 height trait,
    and their properties): the sum of the opacity cube along the z axis.
    """
    gmodel = dict(
        type='intensity_3d', size_z=40, step_z=0.25,
        components=[dict(
            **DISK,
            bptraits=dict(type='exponential'),
            bhtraits=dict(type='sech2'))],
        opacity_components=[dict(
            DISK, type=disk_type,
            optraits=list(optraits),
            ohtraits=[dict(type='sech2')] * len(optraits))])
    if disk_type == 'mcdisk':
        gmodel['opacity_components'][0]['cflux'] = 1e-4
    model = dict(
        driver=dict(type=driver.type()),
        dmodel=dict(type='image', size=[24, 24]),
        gmodel=gmodel)
    properties = dict(
        xpos=0, ypos=0, posa=0, incl=0, bpt_a=1, bpt_s=4, bht_s=1,
        ocmp_xpos=0, ocmp_ypos=0, ocmp_posa=0, ocmp_incl=0,
        **opacity_properties)
    _, extra = evaluate_models([model], properties)
    return extra['observation0_gmodel_total_odata'].data.sum(axis=0)


def test_opacity_traits_give_the_face_on_optical_depth(
        driver, evaluate_models):
    # The polar trait is the optical depth of the disk seen face-on, and
    # the height trait distributes it along z (it integrates to 1), in
    # both types of disk
    y, x = np.indices((24, 24)) - 11.5
    radius = np.hypot(x, y)
    expected = 0.5 * np.exp(-radius / 4)
    inside = radius < 10
    smdisk = face_on_optical_depth(driver, evaluate_models, 'smdisk')
    np.testing.assert_allclose(smdisk[inside], expected[inside], rtol=1e-2)
    # The clouds of the Monte Carlo disk add up to the same optical depth
    mcdisk = face_on_optical_depth(driver, evaluate_models, 'mcdisk')
    np.testing.assert_allclose(
        mcdisk[inside].sum(), smdisk[inside].sum(), rtol=1e-2)


def test_monte_carlo_opacity_mixtures(driver, evaluate_models):
    # A mixture of blobs and an exponential profile: the clouds of the
    # Monte Carlo disk add up to the optical depth of the smooth disk
    optraits = (dict(type='mixture_gauss', nblobs=1), dict(type='exponential'))
    properties = dict(
        ocmp_opt_r=[3.0], ocmp_opt_t=[0.0], ocmp_opt_a=[0.5],
        ocmp_opt_s=[2.0], ocmp_opt_q=[1.0], ocmp_opt_p=[0.0],
        ocmp_opt1_a=0.2, ocmp_opt1_s=3.0, ocmp_oht_s=1, ocmp_oht1_s=1)
    smdisk, mcdisk = (
        face_on_optical_depth(
            driver, evaluate_models, disk_type, optraits, properties)
        for disk_type in ('smdisk', 'mcdisk'))
    np.testing.assert_allclose(mcdisk.sum(), smdisk.sum(), rtol=1e-2)


def test_the_near_side_is_where_outflows_approach(driver, evaluate_models):
    # A thick disk with an outflow and a thin layer of absorbers at its
    # midplane. Along the lines of sight of the near half of the disk, its
    # light behind the absorbers comes from closer to the centre (it is
    # brighter) than in the far half: the near half is the more absorbed.
    # Its outflow approaches the viewer (negative velocities).
    nodes = list(range(0, 16, 2))
    disk = dict(DISK, rnodes=nodes)
    gmodel = dict(type='kinematics_3d', components=[dict(
        disk, bptraits=dict(type='exponential'),
        bhtraits=dict(type='sech2'),
        vptraits=dict(type='nw_rad_uniform'),
        dptraits=dict(type='uniform'))])
    properties = dict(
        vsys=0, xpos=0, ypos=0, posa=0, incl=60, bpt_a=1, bpt_s=4,
        bht_s=1, vpt_vr=[50.0] * len(nodes), dpt_a=20)

    def evaluate(gmodel, properties):
        model = dict(
            driver=dict(type=driver.type()),
            dmodel=dict(type='scube', size=[32, 32, 41], step=[1, 1, 10]),
            gmodel=gmodel)
        data, _ = evaluate_models([model], properties)
        return data[0]['scube']['d']
    clear = evaluate(gmodel, properties)
    dusty = evaluate(
        gmodel | dict(opacity_components=[dict(
            disk, optraits=dict(type='exponential'),
            ohtraits=dict(type='sech2'))]),
        properties | dict(
            ocmp_xpos=0, ocmp_ypos=0, ocmp_posa=0, ocmp_incl=60,
            ocmp_opt_a=2.0, ocmp_opt_s=4, ocmp_oht_s=0.2))
    # The halves of the disk on either side of its major axis (x = 0)
    halves = (slice(4, 14), slice(18, 28))
    velocity = np.arange(-20, 21) * 10.0
    transmitted = [dusty[:, :, h].sum() / clear[:, :, h].sum() for h in halves]
    mean_velocity = [
        np.sum(velocity[:, None, None] * clear[:, :, h]) / clear[:, :, h].sum()
        for h in halves]
    near, far = np.argsort(transmitted)
    assert mean_velocity[near] < 0 < mean_velocity[far]

def image_and_optical_depth(driver, evaluate_models, opacity, step_z):
    """
    The image of an inclined thick disk whose absorbers have the same
    distribution as its light, with the given face-on optical depth at
    the centre, on a z axis of the given step. Also return the optical
    depth of each line of sight (the sum of the opacity cube along z).
    """
    traits = dict(
        bptraits=dict(type='exponential'), bhtraits=dict(type='sech2'))
    opacity_traits = dict(
        optraits=dict(type='exponential'), ohtraits=dict(type='sech2'))
    model = dict(
        driver=dict(type=driver.type()),
        dmodel=dict(type='image', size=[32, 32]),
        gmodel=dict(
            type='intensity_3d', size_z=round(24 / step_z), step_z=step_z,
            components=[dict(DISK, **traits)],
            opacity_components=[dict(DISK, **opacity_traits)]))
    properties = dict(
        xpos=0, ypos=0, posa=30, incl=60, bpt_a=1, bpt_s=4, bht_s=1,
        ocmp_xpos=0, ocmp_ypos=0, ocmp_posa=30, ocmp_incl=60,
        ocmp_opt_a=opacity, ocmp_opt_s=4, ocmp_oht_s=1)
    data, extra = evaluate_models([model], properties)
    tau = extra['observation0_gmodel_total_odata'].data.sum(axis=0)
    return data[0]['image']['d'], tau


@pytest.mark.parametrize('step_z', [0.25, 1, 2])
def test_mixed_absorbers_let_through_the_exact_fraction(
        driver, evaluate_models, step_z):
    # When the absorbers have the same distribution as the emitters, a
    # line of sight of optical depth tau lets through (1 - exp(-tau)) / tau
    # of its light, whatever the size of the voxels
    clear, _ = image_and_optical_depth(driver, evaluate_models, 0, step_z)
    dimmed, tau = image_and_optical_depth(
        driver, evaluate_models, 2, step_z)
    assert tau.max() > 2
    visible = clear > 1e-3 * clear.max()
    np.testing.assert_allclose(
        dimmed[visible] / clear[visible],
        -np.expm1(-tau[visible]) / tau[visible], rtol=1e-4)
