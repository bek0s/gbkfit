"""
Tests for the components with opacity (dust): they absorb the light of
the components behind them along the line of sight.
"""

import copy

import gbkfit.model
import gbkfit.params
import numpy as np
import pytest
from modelutils import observation_group


DISK = dict(
    type='smdisk', loose=False, tilted=False, rnodes=list(range(0, 12)))

# The density of a thick exponential disk
THICK_DENSITY = dict(
    ptraits=dict(type='exponential'), htraits=dict(type='sech2', zmax=8))

BAND = dict(type='band')
LINES = dict(type='lines')
DUST = dict(type='dust')

PROPERTIES = dict(
    gas_vsys=0, gas_xpos=0, gas_ypos=0, gas_posa=30, gas_incl=60,
    gas_pt_a=1, gas_pt_s=4, gas_ht_s=1, gas_vpt_rt=2, gas_vpt_vt=150,
    gas_dpt_a=20,
    dust_xpos=0, dust_ypos=0, dust_posa=30, dust_incl=60,
    dust_pt_s=4, dust_ht_s=1)


def spectral_cube_case(driver, with_opacity):
    """
    A spectral cube, on a grid, of a galaxy of a thick disk of gas, and
    if with_opacity, of a thick disk of dust.
    """
    components = [dict(
        DISK, name='gas', density=THICK_DENSITY,
        kinematics=dict(
            vptraits=dict(type='arctan', direction='tan'),
            dptraits=dict(type='uniform')),
        emission=LINES)]
    if with_opacity:
        components.append(dict(
            DISK, name='dust', density=THICK_DENSITY, opacity=DUST))
    return dict(
        driver=dict(type=driver.type()),
        observable=dict(
            type='pixel_spectra', size=[32, 32, 41], step=[1, 1, 10]),
        line_of_sight=dict(type='grid'),
        galaxy=dict(components=components))


def evaluate(driver, with_opacity, opacity, times=1):
    """Evaluate a thick disk (with or without dust) several times."""
    obs_group = observation_group([spectral_cube_case(driver, with_opacity)])
    properties = dict(PROPERTIES, dust_pt_a=opacity)
    if not with_opacity:
        properties = {
            k: v for k, v in properties.items() if not k.startswith('dust_')}
    params = gbkfit.params.ParamSpace(obs_group.pdescs(), properties)
    values = params.evaluate()
    return [
        copy.deepcopy(obs_group.model_h(values)[0]['spectra']['d'])
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
        driver, evaluate_cases, disk_type,
        ptraits=(dict(type='exponential'),),
        dust_properties=dict(dust_pt_a=0.5, dust_pt_s=4, dust_ht_s=1)):
    """
    The optical depth of each line of sight through a face-on disk of
    dust of the given type and density polar traits (each with a sech2
    height trait, and their properties): the sum of the opacity cube
    along the depth axis.
    """
    dust = dict(
        DISK, type=disk_type, name='dust',
        density=dict(
            ptraits=list(ptraits),
            htraits=[dict(type='sech2', zmax=8)] * len(ptraits)),
        opacity=DUST)
    if disk_type == 'mcdisk':
        dust['cflux'] = 1e-4
    stars = dict(DISK, name='stars', density=THICK_DENSITY, emission=BAND)
    case = dict(
        driver=dict(type=driver.type()),
        observable=dict(type='pixel_brightness', size=[24, 24]),
        line_of_sight=dict(
            type='grid', depth=dict(extent=40 * 0.25, step=0.25)),
        galaxy=dict(components=[stars, dust]))
    properties = dict(
        stars_xpos=0, stars_ypos=0, stars_posa=0, stars_incl=0,
        stars_pt_a=1, stars_pt_s=4, stars_ht_s=1,
        dust_xpos=0, dust_ypos=0, dust_posa=0, dust_incl=0,
        **dust_properties)
    _, extra = evaluate_cases([case], properties)
    return extra.get('total_odata').data.sum(axis=0)


def test_dust_density_gives_the_face_on_optical_depth(
        driver, evaluate_cases):
    # The polar trait is the optical depth of the disk seen face-on, and
    # the height trait distributes it along z (it integrates to 1), in
    # both types of disk
    y, x = np.indices((24, 24)) - 11.5
    radius = np.hypot(x, y)
    expected = 0.5 * np.exp(-radius / 4)
    inside = radius < 10
    smdisk = face_on_optical_depth(driver, evaluate_cases, 'smdisk')
    np.testing.assert_allclose(smdisk[inside], expected[inside], rtol=1e-2)
    # The clouds of the Monte Carlo disk add up to the same optical depth
    mcdisk = face_on_optical_depth(driver, evaluate_cases, 'mcdisk')
    np.testing.assert_allclose(
        mcdisk[inside].sum(), smdisk[inside].sum(), rtol=1e-2)


def test_monte_carlo_opacity_mixtures(driver, evaluate_cases):
    # A mixture of blobs and an exponential profile: the clouds of the
    # Monte Carlo disk add up to the optical depth of the smooth disk
    ptraits = (dict(type='mixture_gauss', nblobs=1), dict(type='exponential'))
    properties = dict(
        dust_pt_r=[3.0], dust_pt_t=[0.0], dust_pt_a=[0.5],
        dust_pt_s=[2.0], dust_pt_q=[1.0], dust_pt_p=[0.0],
        dust_pt1_a=0.2, dust_pt1_s=3.0, dust_ht_s=1, dust_ht1_s=1)
    smdisk, mcdisk = (
        face_on_optical_depth(
            driver, evaluate_cases, disk_type, ptraits, properties)
        for disk_type in ('smdisk', 'mcdisk'))
    np.testing.assert_allclose(mcdisk.sum(), smdisk.sum(), rtol=1e-2)


def test_the_near_side_is_where_outflows_approach(driver, evaluate_cases):
    # A thick disk with an outflow and a thin layer of absorbers at its
    # midplane. Along the lines of sight of the near half of the disk, its
    # light behind the absorbers comes from closer to the centre (it is
    # brighter) than in the far half: the near half is the more absorbed.
    # Its outflow approaches the viewer (negative velocities).
    nodes = list(range(0, 16, 2))
    disk = dict(DISK, rnodes=nodes)
    gas = dict(
        disk, name='gas', density=THICK_DENSITY,
        kinematics=dict(
            vptraits=dict(type='nw_uniform', direction='rad'),
            dptraits=dict(type='uniform')),
        emission=LINES)
    properties = dict(
        gas_vsys=0, gas_xpos=0, gas_ypos=0, gas_posa=0, gas_incl=60,
        gas_pt_a=1, gas_pt_s=4, gas_ht_s=1,
        gas_vpt_vr=[50.0] * len(nodes), gas_dpt_a=20)

    def evaluate(components, properties):
        case = dict(
            driver=dict(type=driver.type()),
            observable=dict(
                type='pixel_spectra', size=[32, 32, 41], step=[1, 1, 10]),
            line_of_sight=dict(type='grid'),
            galaxy=dict(components=components))
        data, _ = evaluate_cases([case], properties)
        return data[0]['spectra']['d']
    clear = evaluate([gas], properties)
    dusty = evaluate(
        [gas, dict(disk, name='dust', density=THICK_DENSITY, opacity=DUST)],
        properties | dict(
            dust_xpos=0, dust_ypos=0, dust_posa=0, dust_incl=60,
            dust_pt_a=2.0, dust_pt_s=4, dust_ht_s=0.2))
    # The halves of the disk on either side of its major axis (x = 0)
    halves = (slice(4, 14), slice(18, 28))
    velocity = np.arange(-20, 21) * 10.0
    transmitted = [dusty[:, :, h].sum() / clear[:, :, h].sum() for h in halves]
    mean_velocity = [
        np.sum(velocity[:, None, None] * clear[:, :, h]) / clear[:, :, h].sum()
        for h in halves]
    near, far = np.argsort(transmitted)
    assert mean_velocity[near] < 0 < mean_velocity[far]

def image_and_optical_depth(driver, evaluate_cases, opacity, depth_step):
    """
    The image of an inclined thick disk whose dust has the same
    distribution as its light, with the given face-on optical depth at
    the centre, on a depth axis of the given step. Also return the
    optical depth of each line of sight (the sum of the opacity cube
    along the depth axis).
    """
    case = dict(
        driver=dict(type=driver.type()),
        observable=dict(type='pixel_brightness', size=[32, 32]),
        line_of_sight=dict(
            type='grid', depth=dict(extent=24.0, step=depth_step)),
        galaxy=dict(components=[
            dict(DISK, name='stars', density=THICK_DENSITY, emission=BAND),
            dict(DISK, name='dust', density=THICK_DENSITY, opacity=DUST)]))
    properties = dict(
        stars_xpos=0, stars_ypos=0, stars_posa=30, stars_incl=60,
        stars_pt_a=1, stars_pt_s=4, stars_ht_s=1,
        dust_xpos=0, dust_ypos=0, dust_posa=30, dust_incl=60,
        dust_pt_a=opacity, dust_pt_s=4, dust_ht_s=1)
    data, extra = evaluate_cases([case], properties)
    tau = extra.get('total_odata').data.sum(axis=0)
    return data[0]['brightness']['d'], tau


@pytest.mark.parametrize('depth_step', [0.25, 1, 2])
def test_mixed_absorbers_let_through_the_exact_fraction(
        driver, evaluate_cases, depth_step):
    # When the absorbers have the same distribution as the emitters, a
    # line of sight of optical depth tau lets through (1 - exp(-tau)) / tau
    # of its light, whatever the size of the voxels
    clear, _ = image_and_optical_depth(
        driver, evaluate_cases, 0, depth_step)
    dimmed, tau = image_and_optical_depth(
        driver, evaluate_cases, 2, depth_step)
    assert tau.max() > 2
    visible = clear > 1e-3 * clear.max()
    np.testing.assert_allclose(
        dimmed[visible] / clear[visible],
        -np.expm1(-tau[visible]) / tau[visible], rtol=1e-4)


THIN_DENSITY = dict(ptraits=dict(type='exponential'))

THIN_GAS = dict(
    DISK, name='gas', density=THIN_DENSITY,
    kinematics=dict(
        vptraits=dict(type='arctan', direction='tan'), dptraits=dict(type='uniform')),
    emission=LINES)

THIN_GAS_PROPERTIES = dict(
    gas_vsys=0, gas_xpos=0, gas_ypos=0, gas_posa=30, gas_incl=60,
    gas_pt_a=1, gas_pt_s=4, gas_vpt_rt=2, gas_vpt_vt=150, gas_dpt_a=20)


@pytest.mark.parametrize('observable, key', [
    (dict(type='pixel_brightness', size=[32, 32]), 'brightness'),
    (dict(type='pixel_spectra', size=[32, 32, 41], step=[1, 1, 10]),
     'spectra')])
def test_a_grid_sees_a_thin_disk_as_the_sky_does(
        driver, evaluate_cases, observable, key):
    # Without dust, the depth of the plane of the disk changes nothing
    component = THIN_GAS if key == 'spectra' else dict(
        DISK, name='gas', density=THIN_DENSITY, emission=BAND)
    properties = {
        name: value for name, value in THIN_GAS_PROPERTIES.items()
        if key == 'spectra' or not name.startswith(('gas_v', 'gas_d'))}
    models = [
        evaluate_cases([dict(
            driver=dict(type=driver.type()), observable=observable,
            line_of_sight=line_of_sight,
            galaxy=dict(components=[component]))], properties)[0][0]
        for line_of_sight in (dict(type='flat'), dict(type='grid'))]
    np.testing.assert_array_equal(models[1][key]['d'], models[0][key]['d'])


def face_on_dust_case(driver, components):
    """
    An image of a galaxy of the given components and a face-on thick disk
    of dust, symmetric about the plane of the sky, on a depth axis of an
    even number of voxels (that plane is between two of them).
    """
    dust = dict(DISK, name='dust', density=THICK_DENSITY, opacity=DUST)
    return dict(
        driver=dict(type=driver.type()),
        observable=dict(type='pixel_brightness', size=[24, 24]),
        line_of_sight=dict(
            type='grid', depth=dict(extent=40 * 0.25, step=0.25)),
        galaxy=dict(components=list(components) + [dust]))


FACE_ON_DUST = dict(
    dust_xpos=0, dust_ypos=0, dust_posa=0, dust_incl=0,
    dust_pt_a=0.5, dust_pt_s=4, dust_ht_s=1)


def test_dust_dims_a_thin_disk_in_front_of_its_plane(driver, evaluate_cases):
    # The light of a face-on thin disk in the middle of the dust crosses
    # half of it
    stars = dict(DISK, name='stars', density=THIN_DENSITY, emission=BAND)
    properties = dict(
        stars_xpos=0, stars_ypos=0, stars_posa=0, stars_incl=0,
        stars_pt_a=1, stars_pt_s=4)
    clear, _ = evaluate_cases(
        [face_on_dust_case(driver, [stars])],
        properties | FACE_ON_DUST | dict(dust_pt_a=0))
    dimmed, extra = evaluate_cases(
        [face_on_dust_case(driver, [stars])], properties | FACE_ON_DUST)
    tau = extra.get('total_odata').data.sum(axis=0)
    clear, dimmed = (data[0]['brightness']['d'] for data in (clear, dimmed))
    visible = clear > 1e-3 * clear.max()
    assert tau[visible].max() > 0.4
    np.testing.assert_allclose(
        dimmed[visible] / clear[visible], np.exp(-tau[visible] / 2),
        rtol=1e-5)


@pytest.mark.parametrize('depth, fraction', [
    (None, 0.5), (-100, 0), (0, 0.5), (100, 1)])
def test_dust_dims_a_point_in_front_of_its_depth(
        driver, evaluate_cases, depth, fraction):
    # A point at the centre of a pixel, by default in the plane of the
    # centre of its galaxy (in the middle of the dust), or at a depth in
    # front of the dust, in it, or behind it: it crosses that fraction of
    # the optical depth of its line of sight
    nucleus = dict(type='point', name='nucleus', emission=BAND)
    properties = dict(nucleus_xpos=0.5, nucleus_ypos=0.5, nucleus_flux=1)
    if depth is not None:
        nucleus['depth'] = True
        properties['nucleus_depth'] = depth
    data, extra = evaluate_cases(
        [face_on_dust_case(driver, [nucleus])], properties | FACE_ON_DUST)
    tau = extra.get('total_odata').data.sum(axis=0)[12, 12]
    assert data[0]['brightness']['d'][12, 12] == pytest.approx(
        np.exp(-fraction * tau), rel=1e-5)


# Extinction laws whose ratios differ much between halpha (6564.61 A) and
# nii6583 (6585.27 A): 1.031 and 0.721 (A), 1.469 and 1.779 (B)
TABLE_A = dict(type='table', wavelengths=[0.65, 0.66], ratios=[2.0, 0.5])
TABLE_B = dict(type='table', wavelengths=[0.65, 0.66], ratios=[0.5, 2.0])
RATIOS = {
    'A': dict(halpha=2 - 1.5 * 0.6461, nii6583=2 - 1.5 * 0.8527),
    'B': dict(halpha=0.5 + 1.5 * 0.6461, nii6583=0.5 + 1.5 * 0.8527)}

# The channels of each line (km/s; nii6583 is at 943.5 km/s)
LINE_WINDOWS = dict(halpha=(-300, 300), nii6583=(643, 1243))


def lines_in_dust(driver, evaluate_cases, emitter, laws, lines, opacity):
    """
    The spectral cube of halpha and nii6583 of a face-on thin disk or a
    point (emitter) in the middle of face-on disks of dust, one of each of
    the given laws, of the given face-on optical depth at V at the centre,
    seen in the given lines. Also return the optical depth at V of each
    disk of dust along each line of sight.
    """
    emission = dict(type='lines', lines=['halpha', 'nii6583'])
    if emitter == 'disk':
        gas = dict(
            DISK, name='gas', density=THIN_DENSITY, emission=emission,
            kinematics=dict(
                vptraits=dict(type='arctan', direction='tan'),
                dptraits=dict(type='uniform')))
        properties = dict(
            gas_vsys=0, gas_xpos=0, gas_ypos=0, gas_posa=0, gas_incl=0,
            gas_pt_a=1, gas_pt_s=4, gas_vpt_rt=2, gas_vpt_vt=150,
            gas_dpt_a=20)
    else:
        gas = dict(type='point', name='gas', emission=emission)
        properties = dict(
            gas_xpos=0.5, gas_ypos=0.5, gas_flux=1, gas_vsys=0, gas_disp=20)
    properties['gas_nii6583_ratio'] = 1
    dusts = [
        dict(DISK, name=f'dust{i}', density=THICK_DENSITY,
             opacity=dict(type='dust', law=law))
        for i, law in enumerate(laws)]
    for i in range(len(laws)):
        properties |= {
            f'dust{i}_{key.removeprefix("dust_")}': value
            for key, value in FACE_ON_DUST.items()}
        properties[f'dust{i}_pt_a'] = opacity
    case = dict(
        driver=dict(type=driver.type()),
        observable=dict(
            type='pixel_spectra', size=[24, 24, 81], step=[1, 1, 25],
            rval=[0, 0, 500], rest='6564.61 Angstrom'),
        line_of_sight=dict(
            type='grid', depth=dict(extent=40 * 0.25, step=0.25)),
        galaxy=dict(components=[gas] + dusts),
        **({} if lines is None else dict(lines=lines)))
    data, extra = evaluate_cases([case], properties)
    taus = [
        extra.get('odata', component=f'dust{i}').data.sum(axis=0)
        for i in range(len(laws))]
    return data[0]['spectra']['d'], taus


def line_flux(cube, line):
    """The flux of a line in each spaxel: the sum of its channels."""
    velocity = 500 + (np.arange(81) - 40) * 25
    low, high = LINE_WINDOWS[line]
    return cube[(velocity > low) & (velocity < high)].sum(axis=0)


@pytest.mark.parametrize('emitter', ['disk', 'point'])
@pytest.mark.parametrize('laws, lines', [
    (('A',), None),
    # Two laws: a cube of opacity for each, or one for one wavelength
    (('A', 'B'), None),
    (('A', 'B'), ['halpha'])])
def test_dust_dims_each_line_by_its_law(
        driver, evaluate_cases, emitter, laws, lines):
    # The light of each line crosses half of the dust, whose optical depth
    # at the line is that at V times the ratio of its law there
    tables = dict(A=TABLE_A, B=TABLE_B)
    args = (
        driver, evaluate_cases, emitter, [tables[law] for law in laws],
        lines)
    clear, _ = lines_in_dust(*args, opacity=0)
    dimmed, taus = lines_in_dust(*args, opacity=0.5)
    for line in lines or ['halpha', 'nii6583']:
        tau = sum(RATIOS[law][line] * tau for law, tau in zip(laws, taus))
        flux = line_flux(clear, line)
        visible = flux > 1e-3 * flux.max()
        np.testing.assert_allclose(
            line_flux(dimmed, line)[visible] / flux[visible],
            np.exp(-tau[visible] / 2), rtol=1e-4)


def test_dust_of_a_law_needs_named_lines(driver):
    from gbkfit.utils.parseutils import ConfigError
    case = spectral_cube_case(driver, with_opacity=True)
    case['galaxy']['components'][1]['opacity'] = dict(
        type='dust', law=dict(type='ccm'))
    with pytest.raises(ConfigError, match="must name its lines"):
        observation_group([case])
