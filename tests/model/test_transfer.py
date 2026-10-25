"""
Tests for radiative transfer along the line of sight (Transfer): the
fields of the light of the emitters, which the march reads.
"""

import numpy as np
import pytest
from modelutils import observation_group

from gbkfit.model import Extras, galaxy_parser
from gbkfit.observation import DepthGroup
from gbkfit.utils import gridutils


# A spectral axis of velocities (the fields do not depend on it)
SPECTRAL = gridutils.Grid((11,), gridutils.Coords((10,), (5,), (0,), 0), 0)

DISK_PARAMS = dict(
    vsys=0, xpos=0.5, ypos=-1, posa=30, incl=60, pt_a=1, pt_s=3, ht_s=1,
    vpt_rt=2, vpt_vt=150, dpt_a=20)


def disk(type_, thick=True, **options):
    """The configuration of a disk of gas, thick or thin."""
    density = dict(ptraits=dict(type='exponential'))
    if thick:
        density.update(htraits=dict(type='sech2', zmax=8))
    return dict(
        type=type_, loose=False, tilted=False, rnmin=0, rnmax=10, rnsep=1,
        density=density,
        kinematics=dict(
            vptraits=dict(type='arctan', direction='tan'), dptraits=dict(type='uniform')),
        emission=dict(type='lines')) | options


def native_grid(size):
    """A grid of voxels of size 1, centred on 0."""
    return dict(
        spat_size=size, spat_step=(1.0, 1.0, 1.0),
        spat_zero=tuple(-(n / 2 - 0.5) for n in size), spat_rota=0.0,
        slab_zero=0, spec_size=SPECTRAL.size[0],
        spec_step=SPECTRAL.coords.step[0], spec_zero=SPECTRAL.zero()[0])


def component_plan(driver, component):
    """The plan of the light of a component (of a galaxy of it alone)."""
    galaxy = galaxy_parser.load(dict(components=[component]))
    return galaxy.components()[0].plan(
        driver, SPECTRAL, np.dtype(np.float32), None, None, True, None,
        None, 0.0)


def evaluate_fields(driver, component, params, grid):
    """
    Return the fields of a component on a grid, and its extra outputs.
    """
    plan = component_plan(driver, component)
    fields = driver.mem_alloc_d(
        plan.fields_layout().shape(grid), np.float32)
    extra = {}
    plan.evaluate_fields(params, grid, fields, extra)
    return driver.mem_copy_d2h(fields), extra


@pytest.mark.parametrize('disk_type', ['smdisk', 'mcdisk'])
def test_the_fields_of_a_thick_disk_are_its_light_and_kinematics(
        driver, disk_type):
    options = dict(cflux=1e-3) if disk_type == 'mcdisk' else {}
    (j, v, s, *_), extra = evaluate_fields(
        driver, disk(disk_type, **options), DISK_PARAMS,
        native_grid((24, 24, 24)))
    on = extra['bdata'] != 0
    np.testing.assert_allclose(j, extra['bdata'], rtol=1e-5)
    np.testing.assert_allclose(v[on], extra['vdata'][on], atol=1e-3)
    if disk_type == 'smdisk':
        np.testing.assert_allclose(s[on], extra['ddata'][on], rtol=1e-6)
    else:
        # The line of the clouds of a voxel is as wide as the sum of
        # theirs: the spread of their velocities widens it
        assert np.all(s[on] >= extra['ddata'][on] * (1 - 1e-5))
        assert s[on].mean() > extra['ddata'][on].mean() + 0.1


def test_the_clouds_keep_their_dispersion_at_a_large_systemic_velocity(
        driver):
    # Clouds of one velocity: the line of each voxel is theirs (float32
    # would lose it in the moments of velocities of 3000)
    params = DISK_PARAMS | dict(vsys=3000, vpt_vt=0, dpt_a=10)
    (j, v, s, _), _ = evaluate_fields(
        driver, disk('mcdisk', cflux=1e-3), params, native_grid((24, 24, 24)))
    on = j != 0
    np.testing.assert_allclose(v[on], 3000, rtol=1e-6)
    np.testing.assert_allclose(s[on], 10, rtol=1e-4)


def test_a_thin_disk_is_a_layer_where_its_plane_crosses(driver):
    params = DISK_PARAMS | dict(xpos=0, ypos=0, posa=0)
    grid = native_grid((24, 24, 8))
    component = disk('smdisk', thick=False)
    (j, v, s, position), extra = evaluate_fields(
        driver, component, params, grid)
    image = driver.mem_alloc_d((24, 24), np.float32)
    driver.mem_fill(image, 0)
    component_plan(driver, component).evaluate_emission(
        params, grid, dict(image=image), None)
    np.testing.assert_array_equal(j, driver.mem_copy_d2h(image))
    # The plane of the disk is at the depth |x| tan(incl) from the plane
    # of the sky (x along its minor axis), even beyond the depth axis
    on = j != 0
    x = np.broadcast_to(grid['spat_zero'][0] + np.arange(24), (24, 24))
    depth = grid['spat_zero'][2] + position
    np.testing.assert_allclose(
        np.abs(depth[on]), np.abs(x[on]) * np.tan(np.radians(60)),
        atol=1e-4)
    assert np.abs(depth[on]).max() > 4


def test_a_point_is_a_layer_at_its_depth(driver):
    params = dict(xpos=0.25, ypos=0.5, depth=2, flux=3, vsys=50, disp=-10)
    point = dict(type='point', emission=dict(type='lines'), depth=True)
    grid = native_grid((6, 6, 8))
    (j, v, s, position), _ = evaluate_fields(driver, point, params, grid)
    on = j != 0
    assert on.sum() == 2 and j.sum() == pytest.approx(3)
    np.testing.assert_array_equal(v[on], 50)
    np.testing.assert_array_equal(s[on], 10)
    np.testing.assert_array_equal(position[on], 2 + 3.5)
    np.testing.assert_array_equal(np.stack((v, s, position))[:, ~on], 0)


# A galaxy of gas, stars and dust, seen through a grid and through
# radiative transfer
LINES = dict(type='lines')
KINEMATICS = dict(
    vptraits=dict(type='arctan', direction='tan'), dptraits=dict(type='uniform'))
THICK = dict(
    ptraits=dict(type='exponential'), htraits=dict(type='sech2', zmax=8))
THIN = dict(ptraits=dict(type='exponential'))
GAS = dict(
    type='smdisk', name='gas', loose=False, tilted=False, rnmin=0, rnmax=10,
    rnsep=1, density=THICK, kinematics=KINEMATICS, emission=LINES)
DUST = dict(
    type='smdisk', name='dust', loose=False, tilted=False, rnmin=0,
    rnmax=10, rnsep=1, density=THICK, opacity=dict(type='dust'))
NUCLEUS = dict(type='point', name='nucleus', emission=LINES, depth=True)
OUTFLOW = dict(
    type='smcone', name='outflow', rnodes=[1, 3, 6],
    density=dict(
        ptraits=dict(type='uniform'), htraits=dict(type='gauss', zmax=1)),
    kinematics=dict(
        vptraits=dict(type='uniform', direction='out'),
        dptraits=dict(type='uniform')),
    emission=LINES)

PROPERTIES = dict(
    gas_vsys=0, gas_xpos=0.5, gas_ypos=-1, gas_posa=30, gas_incl=60,
    gas_pt_a=1, gas_pt_s=3, gas_ht_s=1, gas_vpt_rt=2, gas_vpt_vt=150,
    gas_dpt_a=20, dust_xpos=0, dust_ypos=0, dust_posa=40, dust_incl=70,
    dust_pt_a=0.3, dust_pt_s=4, dust_ht_s=0.5, nucleus_xpos=0.3,
    nucleus_ypos=0.6, nucleus_depth=0.7, nucleus_flux=20, nucleus_vsys=30,
    nucleus_disp=25)

# Extinction laws whose ratios differ much between halpha and nii6583
LAWS = [
    dict(type='table', wavelengths=[0.65, 0.66], ratios=[2.0, 0.5]),
    dict(type='table', wavelengths=[0.65, 0.66], ratios=[0.5, 2.0])]


def cubes_of(evaluate_cases, driver, components, properties, **observation):
    """
    The model data of a galaxy of the given components seen through a
    grid and through radiative transfer (a spectral cube by default).
    """
    observation = dict(
        observable=dict(
            type='pixel_spectra', size=[24, 24, 31], step=[1, 1, 10])
    ) | observation
    models = []
    for kind in ('grid', 'transfer'):
        case = dict(
            observation, driver=dict(type=driver.type()),
            line_of_sight=dict(type=kind),
            galaxy=dict(components=components))
        data, _ = evaluate_cases([case], of_model([case], properties))
        models.append(next(iter(data[0].values()))['d'])
    return models


def of_model(cases, properties):
    """
    Those of the properties that the model of cases has (the tests share
    properties between models with and without some components or
    heights).
    """
    pdescs = observation_group(cases).pdescs()
    return {key: value for key, value in properties.items() if key in pdescs}


def assert_same_model(actual, desired):
    # (to 1e-5 of the peak: on a grid, the lines of a point have tails
    # beyond the 5 dispersions where the march, as the disks, cuts them)
    np.testing.assert_allclose(
        actual, desired, rtol=1e-5, atol=1e-5 * np.abs(desired).max())


@pytest.mark.parametrize('components', [
    [GAS],
    [GAS, DUST],
    [GAS | dict(density=THIN), DUST],
    [NUCLEUS, DUST],
    [GAS, NUCLEUS, DUST, GAS | dict(name='gas2', density=THIN)]],
    ids=['thick', 'dust', 'thin', 'point', 'all'])
def test_transfer_gives_what_a_grid_gives(driver, evaluate_cases, components):
    properties = PROPERTIES | {
        key.replace('gas_', 'gas2_'): value
        for key, value in PROPERTIES.items() if key.startswith('gas_')}
    grid, transfer = cubes_of(evaluate_cases, driver, components, properties)
    assert grid.max() > 0
    assert_same_model(transfer, grid)


def test_transfer_dims_each_line_by_its_law(driver, evaluate_cases):
    # Two laws and two lines: a cube of opacity for each law, and the
    # light of each line dimmed by its ratios
    gas = GAS | dict(emission=dict(type='lines', lines=['halpha', 'nii6583']))
    dusts = [
        DUST | dict(name=f'dust{i}', opacity=dict(type='dust', law=law))
        for i, law in enumerate(LAWS)]
    properties = PROPERTIES | dict(gas_nii6583_ratio=0.5) | {
        f'dust{i}_{key.removeprefix("dust_")}': value
        for i in range(2) for key, value in PROPERTIES.items()
        if key.startswith('dust_')}
    grid, transfer = cubes_of(
        evaluate_cases, driver, [gas] + dusts, properties,
        observable=dict(
            type='pixel_spectra', size=[24, 24, 81], step=[1, 1, 25],
            rval=[0, 0, 500], rest='6564.61 Angstrom'))
    assert_same_model(transfer, grid)


def test_transfer_gives_what_a_grid_gives_for_an_image(
        driver, evaluate_cases):
    stars = GAS | dict(name='stars', emission=dict(type='band'))
    del stars['kinematics']
    properties = PROPERTIES | dict(
        stars_xpos=0.5, stars_ypos=-1, stars_posa=30, stars_incl=60,
        stars_pt_a=1, stars_pt_s=3, stars_ht_s=1)
    grid, transfer = cubes_of(
        evaluate_cases, driver, [stars, DUST], properties,
        observable=dict(type='pixel_brightness', size=[24, 24]))
    assert_same_model(transfer, grid)


def test_transfer_keeps_the_moments_of_monte_carlo_clouds(
        driver, evaluate_cases):
    # The clouds of a voxel become one line of their flux, mean velocity
    # and width: the moments of each spaxel are kept
    gas = GAS | dict(type='mcdisk', cflux=1e-3)
    grid, transfer = cubes_of(
        evaluate_cases, driver, [gas, DUST], PROPERTIES,
        observable=dict(
            type='pixel_spectra', size=[24, 24, 61], step=[1, 1, 10]))
    velocity = (np.arange(61) - 30) * 10.0
    flux = grid.sum(axis=0)
    on = flux > 1e-3 * flux.max()

    def moments(cube):
        spectra = cube[:, on]
        flux = spectra.sum(axis=0)
        mean = velocity @ spectra / flux
        return flux, mean, np.sqrt(velocity ** 2 @ spectra / flux - mean ** 2)

    for desired, actual, rtol in zip(
            moments(grid), moments(transfer), (1e-5, 1e-5, 1e-4)):
        np.testing.assert_allclose(
            actual, desired, rtol=rtol, atol=rtol * np.abs(desired).max())


def line_opacity(gas, coefficients):
    """A component with line opacity of the given coefficient traits."""
    return gas | dict(opacity=dict(type='line', coefficients=coefficients))


def transfer_cube(evaluate_cases, driver, components, properties):
    """The spectral cube of components seen through radiative transfer."""
    case = dict(
        driver=dict(type=driver.type()), line_of_sight=dict(type='transfer'),
        observable=dict(
            type='pixel_spectra', size=[24, 24, 61], step=[1, 1, 10]),
        galaxy=dict(components=components))
    names = {component['name'] for component in components}
    data, _ = evaluate_cases([case], {
        key: value for key, value in properties.items()
        if key.split('_')[0] in names})
    return data[0]['spectra']['d']


@pytest.mark.parametrize('disk_type', ['smdisk', 'mcdisk'])
def test_line_opacity_saturates_at_the_source_function(
        driver, evaluate_cases, disk_type):
    # Each voxel's optical depth at each channel is its light there over S
    # (a uniform coefficient 1/S), so the light of each channel is S (1 -
    # exp(-I / S)), I that without line opacity, whatever the geometry
    gas = GAS | (dict(type='mcdisk', cflux=1e-3) if disk_type == 'mcdisk'
                 else {})
    thin = transfer_cube(evaluate_cases, driver, [gas], PROPERTIES)
    source = thin.max() / 3
    for coefficient, expected in (
            (0, thin), (1 / source, source * -np.expm1(-thin / source))):
        cube = transfer_cube(
            evaluate_cases, driver,
            [line_opacity(gas, dict(type='uniform'))],
            PROPERTIES | dict(gas_op_a=coefficient))
        assert_same_model(cube, expected)
    assert cube.sum() < 0.8 * thin.sum()


@pytest.mark.parametrize('separation, dims', [(300, False), (0, True)])
def test_lines_absorb_only_at_their_velocities(
        driver, evaluate_cases, separation, dims):
    # Two face-on disks (each of one velocity on every line of sight):
    # far apart in velocity, each is seen as if alone; at one velocity,
    # the one behind is dimmed
    disks = [
        line_opacity(GAS | dict(name=name), dict(type='uniform'))
        for name in ('gas', 'gas2')]
    properties = {}
    for name, vsys in (('gas', -separation / 2), ('gas2', separation / 2)):
        properties |= {
            key.replace('gas_', f'{name}_'): value
            for key, value in PROPERTIES.items() if key.startswith('gas_')}
        properties |= {
            f'{name}_vsys': vsys, f'{name}_incl': 0, f'{name}_op_a': 100}
    both = transfer_cube(evaluate_cases, driver, disks, properties)
    alone = sum(
        transfer_cube(evaluate_cases, driver, [disk], properties)
        for disk in disks)
    if dims:
        assert both.sum() < 0.95 * alone.sum()
    else:
        assert_same_model(both, alone)


def test_named_lines_take_a_coefficient_each(driver, evaluate_cases):
    from gbkfit.model import galaxy_parser
    emission = dict(type='lines', lines=['halpha', 'nii6583'])
    coefficients = dict(
        halpha=dict(type='uniform'),
        nii6583=[dict(type='uniform'), dict(type='exponential')])
    galaxy = galaxy_parser.load(dict(components=[line_opacity(
        GAS | dict(emission=emission), coefficients)]))
    assert {'gas_op_halpha_a', 'gas_op_nii6583_a', 'gas_op1_nii6583_a',
            'gas_op1_nii6583_s'} <= set(galaxy.pdescs())
    dumped = galaxy_parser.dump(galaxy)
    assert galaxy_parser.dump(galaxy_parser.load(dumped)) == dumped


def test_a_uniform_coefficient_is_not_discouraged(caplog):
    # (unlike a uniform density: one source function is common)
    import logging
    caplog.set_level(logging.WARNING)
    galaxy_parser.load(dict(components=[
        line_opacity(GAS, dict(type='uniform'))]))
    assert 'discouraged' not in caplog.text


@pytest.mark.parametrize('component, line_of_sight, message', [
    (line_opacity(GAS, dict(type='uniform')), 'grid',
     "needs a line of sight of radiative transfer"),
    (line_opacity(GAS, dict(halpha=dict(type='uniform'))), 'transfer',
     "one line without a name"),
    (line_opacity(
        GAS | dict(emission=dict(type='lines', lines=['halpha', 'nii6583'])),
        dict(halpha=dict(type='uniform'))), 'transfer',
     "needs the coefficients of every line"),
    (line_opacity(GAS | dict(density=THIN), dict(type='uniform')), 'transfer',
     "needs density height traits"),
    (line_opacity(
        GAS | dict(emission=dict(type='band')), dict(type='uniform')),
     'transfer', "needs emission lines")])
def test_line_opacity_errors(driver, component, line_of_sight, message):
    from gbkfit.utils.parseutils import ConfigError
    from modelutils import group_plan
    case = dict(
        driver=dict(type=driver.type()),
        line_of_sight=dict(type=line_of_sight),
        observable=dict(
            type='pixel_spectra', size=[24, 24, 31], step=[1, 1, 10]),
        galaxy=dict(components=[component]))
    with pytest.raises(ConfigError, match=message):
        group_plan([case])


@pytest.mark.parametrize('disk_type', ['smdisk', 'mcdisk'])
def test_slabs_give_the_same_model(driver, evaluate_cases, disk_type):
    # A thick disk with line opacity, a thin disk, a point, a cone and
    # dust, on the depth axis at once, a plane at a time, and in slabs of
    # 7 planes
    gas = line_opacity(GAS, dict(type='uniform')) | (
        dict(type='mcdisk', cflux=1e-3) if disk_type == 'mcdisk' else {})
    components = [
        gas, NUCLEUS, DUST, GAS | dict(name='gas2', density=THIN), OUTFLOW]
    properties = PROPERTIES | dict(gas_op_a=0.5) | {
        key.replace('gas_', 'gas2_'): value
        for key, value in PROPERTIES.items() if key.startswith('gas_')} \
        | dict(
            outflow_vsys=0, outflow_xpos=0.5, outflow_ypos=-1,
            outflow_posa=30, outflow_incl=60, outflow_wall=30,
            outflow_pt_a=0.2, outflow_ht_s=0.4, outflow_vpt_vo=200,
            outflow_dpt_a=40)
    models = []
    for slab in (None, 1, 7):
        case = dict(
            driver=dict(type=driver.type()),
            line_of_sight=dict(type='transfer') | (
                dict(slab=slab) if slab is not None else {}),
            observable=dict(
                type='pixel_spectra', size=[24, 24, 31], step=[1, 1, 10]),
            galaxy=dict(components=components))
        data, extra = evaluate_cases([case], of_model([case], properties))
        models.append((data[0]['spectra']['d'], extra))
    (cube, extra), *others = models
    for other_cube, other_extra in others:
        assert_same_model(other_cube, cube)
        assert names(other_extra) == names(extra)
        for name, ids in (
                ('total_odata', {}), ('total_bdata', {}),
                ('vdata', dict(component='gas')),
                ('vdata', dict(component='outflow'))):
            value = extra.get(name, **ids).data
            np.testing.assert_allclose(
                other_extra.get(name, **ids).data, value, rtol=1e-5,
                atol=1e-6 * np.nanmax(np.abs(value)))


def names(extra):
    """Return the names and the ids of the extra outputs."""
    return {(x.name, tuple(sorted(x.ids.items()))) for x in extra}


def galaxies_model(
        driver, galaxies, properties, depth_order=None, with_extras=True,
        **observation):
    """
    The model data of an observation of several galaxies through
    radiative transfer (a spectral cube by default), front to back in the
    given order (that of the galaxies if None), and its extras (none
    without with_extras: the evaluation of fits). The galaxies take those
    of the properties that they have (see of_model).
    """
    import gbkfit.params
    from gbkfit.observation import ObservationGroup, observation_parser
    observation = dict(
        type='sky',
        observable=dict(
            type='pixel_spectra', size=[24, 24, 31], step=[1, 1, 10]),
        line_of_sight=dict(type='transfer'),
        galaxies=[galaxy['name'] for galaxy in galaxies]) | observation
    if depth_order is None:
        depth_order = [galaxy['name'] for galaxy in galaxies]
    group = ObservationGroup(
        galaxy_parser.load(galaxies),
        observation_parser.load([observation]), depth_order)
    plan = group.plan(driver)
    pdescs = group.pdescs()
    params = gbkfit.params.ParamSpace(pdescs, {
        key: value for key, value in properties.items() if key in pdescs})
    extra = Extras() if with_extras else None
    data = plan.model_h(params.evaluate(), extra)
    return next(iter(data[0].values()))['d'], extra


def galaxy_properties(galaxy, properties):
    """The properties of a galaxy of the given name (prefixed)."""
    return {f'{galaxy}_{key}': value for key, value in properties.items()}


@pytest.mark.parametrize('dust_in_front', [True, False])
def test_the_dust_of_a_galaxy_dims_the_galaxies_behind_it(
        driver, dust_in_front):
    # An image of a face-on disk of stars at z = 1, and a galaxy of face-on
    # dust at z = 0.5 of a law whose ratio is 3 - 2.5 (wavelength - 0.2):
    # in front of the stars, the dust dims them at the wavelength of the
    # band in its own rest frame; behind them, not at all
    from gbkfit.model import LineCatalogue
    stars = dict(
        GAS, name='stars', density=THIN, emission=dict(type='band'),
        kinematics=None)
    dust = DUST | dict(opacity=dict(type='dust', law=dict(
        type='table', wavelengths=[0.2, 1.0], ratios=[3.0, 1.0])))
    galaxies = [
        dict(name='back', components=[stars], redshift=1.0),
        dict(name='front', components=[dust], redshift=0.5)]
    properties = galaxy_properties('back', dict(
        stars_xpos=0, stars_ypos=0, stars_posa=0, stars_incl=0,
        stars_pt_a=1, stars_pt_s=4))
    properties |= galaxy_properties('front', dict(
        dust_xpos=0, dust_ypos=0, dust_posa=0, dust_incl=0, dust_pt_a=0.5,
        dust_pt_s=4, dust_ht_s=1))
    order = ['front', 'back'] if dust_in_front else ['back', 'front']
    image = dict(
        observable=dict(type='pixel_brightness', size=[24, 24]),
        band='sdss_r')
    clear, _ = galaxies_model(driver, galaxies[:1], properties, **image)
    model, extra = galaxies_model(
        driver, galaxies, properties, order, **image)
    tau = extra.get('total_odata', galaxy='front').data.sum(axis=0)
    wavelength = LineCatalogue().band('sdss_r').wavelength().to_value('um')
    ratio = 3 - 2.5 * (wavelength / 1.5 - 0.2)
    expected = clear * np.exp(-ratio * tau) if dust_in_front else clear
    assert tau.max() > 0.1
    assert_same_model(model, expected)


def test_line_opacity_continues_into_the_galaxies_behind(driver):
    # Two galaxies of gas at one place on the sky; the one in front has
    # line opacity of a uniform coefficient 1/S, so its light is S (1 -
    # e^-t), t = I / S of its light without line opacity, and that of the
    # one behind is dimmed by e^-t (see
    # test_line_opacity_saturates_at_the_source_function). Swapping their
    # order swaps the roles. (Without extras: the evaluation of fits.)
    gas = line_opacity(GAS, dict(type='uniform'))
    galaxies = [dict(name=name, components=[gas]) for name in ('a', 'b')]
    properties = {}
    for name, posa in (('a', 30), ('b', 120)):
        properties |= galaxy_properties(name, {
            key: value for key, value in PROPERTIES.items()
            if key.startswith('gas_')} | dict(gas_posa=posa, gas_op_a=0))

    def model(order, **values):
        return galaxies_model(
            driver, galaxies, properties | values, order,
            with_extras=False)[0]

    thin_a = model(['a', 'b'], b_gas_pt_a=0)
    thin_b = model(['a', 'b'], a_gas_pt_a=0)
    source = thin_a.max() / 3
    dimmed = np.exp(-thin_a / source)
    in_front = model(['a', 'b'], a_gas_op_a=1 / source)
    behind = model(['b', 'a'], a_gas_op_a=1 / source)
    assert_same_model(in_front, source * (1 - dimmed) + dimmed * thin_b)
    assert_same_model(behind, source * (1 - dimmed) + thin_b)
    assert in_front.sum() < 0.95 * behind.sum()


def test_galaxies_without_opacity_add_up_in_any_order(driver):
    # Their light just adds up, in any order, behind a galaxy with dust
    galaxies = [
        dict(name='a', components=[GAS, DUST]),
        dict(name='b', components=[GAS]),
        dict(name='c', components=[GAS])]
    properties = {}
    for name in ('a', 'b', 'c'):
        values = PROPERTIES if name == 'a' else {
            key: value for key, value in PROPERTIES.items()
            if key.startswith('gas_')}
        properties |= galaxy_properties(name, values)
    model, _ = galaxies_model(driver, galaxies, properties, ['a', 'b', 'c'])
    swapped, _ = galaxies_model(
        driver, galaxies, properties, ['a', 'c', 'b'])
    assert_same_model(swapped, model)


def test_a_group_at_one_place_is_a_galaxy_of_all(driver):
    # Two galaxies of a group at separation 0 (gas; and dust with a point
    # in it) are one galaxy of all their components: the dust dims the
    # gas and the point voxel by voxel. The march of the group writes its
    # extras under its name.
    gas = {k: v for k, v in PROPERTIES.items() if k.startswith('gas_')}
    rest = {k: v for k, v in PROPERTIES.items() if k not in gas}
    one, _ = galaxies_model(
        driver, [dict(name='one', components=[GAS, DUST, NUCLEUS])],
        galaxy_properties('one', PROPERTIES))
    galaxies = [
        dict(name='a', components=[GAS]),
        dict(name='b', components=[DUST, NUCLEUS])]
    properties = galaxy_properties('a', gas) \
        | galaxy_properties('b', rest) | dict(b_separation=0)
    pair, extra = galaxies_model(
        driver, galaxies, properties, [DepthGroup('pair', ('a', 'b'))])
    assert_same_model(pair, one)
    assert extra.find('total_unabsorbed', group='pair')
    assert extra.find('bdata', galaxy='a', component='gas')


@pytest.mark.parametrize('separation', [100, -100])
def test_a_group_far_apart_is_its_galaxies_in_order(driver, separation):
    # Far apart (by whole voxels), the galaxies of a group are the
    # galaxies in the order of their separation, and leaving out more of
    # the empty stretch between them changes nothing
    galaxies = [
        dict(name='a', components=[GAS, DUST]),
        dict(name='b', components=[GAS])]
    properties = galaxy_properties('a', PROPERTIES) | galaxy_properties(
        'b', {k: v for k, v in PROPERTIES.items() if k.startswith('gas_')})

    def model(order, **values):
        return galaxies_model(
            driver, galaxies, properties | values, order,
            with_extras=False)[0]

    group = [DepthGroup('pair', ('a', 'b'))]
    ordered = model(['a', 'b'] if separation > 0 else ['b', 'a'])
    assert_same_model(model(group, b_separation=separation), ordered)
    assert_same_model(model(group, b_separation=3 * separation), ordered)


@pytest.mark.parametrize('line_of_sight, depth_order, message', [
    ('grid', ['a', 'b'], "needs a line of sight of radiative transfer"),
    ('transfer', ['a'], r"must give the place of the galaxies \['b'\]"),
    ('transfer', ['a', 'c'], r"names unknown galaxies \['c'\]"),
    ('transfer', ['a', 'b', 'a'], "repeats galaxies"),
    ('transfer', [DepthGroup('pair', ('a',)), 'b'], "at least two galaxies"),
    ('transfer', [DepthGroup('a', ('a', 'b'))], "the name of a galaxy")])
def test_several_galaxies_errors(
        driver, line_of_sight, depth_order, message):
    from gbkfit.utils.parseutils import ConfigError
    galaxies = [
        dict(name='a', components=[GAS, DUST]),
        dict(name='b', components=[DUST])]
    with pytest.raises(ConfigError, match=message):
        galaxies_model(
            driver, galaxies, {}, depth_order,
            line_of_sight=dict(type=line_of_sight))


def transfer_case(driver, components, nchannels=31, step=10, **transfer):
    """A case of a spectral cube of components seen through transfer."""
    return dict(
        driver=dict(type=driver.type()),
        line_of_sight=dict(type='transfer') | transfer,
        observable=dict(
            type='pixel_spectra', size=[24, 24, nchannels],
            step=[1, 1, step]),
        galaxy=dict(components=components))


def evaluate_transfer(evaluate_cases, case, properties):
    """The spectral cube and the extras of a case of transfer_case."""
    names = {component['name'] for component in case['galaxy']['components']}
    data, extra = evaluate_cases([case], {
        key: value for key, value in properties.items()
        if key.split('_')[0] in names})
    return data[0]['spectra']['d'], extra


def test_subchannels_march_each_channel_at_finer_channels(
        driver, evaluate_cases):
    # A face-on disk of narrow lines with line opacity of a uniform
    # coefficient 1/S (each channel is S (1 - exp(-I / S)), see
    # test_line_opacity_saturates_at_the_source_function): with 4
    # sub-channels, each channel is the mean of 4 channels 4 times
    # narrower; light without line opacity is the same with any number
    gas = line_opacity(GAS, dict(type='uniform'))
    properties = PROPERTIES | dict(gas_incl=0, gas_dpt_a=4)

    def cube(nchannels, step, subchannels, coefficient):
        return evaluate_transfer(evaluate_cases, transfer_case(
            driver, [gas], nchannels, step, subchannels=subchannels),
            properties | dict(gas_op_a=coefficient))[0]
    fine_thin = cube(31 * 4, 2.5, 1, 0)
    source = fine_thin.max() / 3
    expected = (source * -np.expm1(-fine_thin / source)).reshape(
        (31, 4) + fine_thin.shape[1:]).mean(axis=1)
    assert_same_model(cube(31, 10, 4, 1 / source), expected)
    assert np.abs(cube(31, 10, 1, 1 / source) - expected).max() \
        > 0.01 * expected.max()
    assert_same_model(cube(31, 10, 4, 0), cube(31, 10, 1, 0))


def test_transfer_extras(driver, evaluate_cases):
    # Gas with line opacity of a uniform coefficient 1/S, and dust: the
    # light without opacity is that of the gas alone, and the optical
    # depth of its lines at each channel is that light over S
    thin, _ = evaluate_transfer(
        evaluate_cases, transfer_case(driver, [GAS]), PROPERTIES)
    source = thin.max() / 3
    _, extra = evaluate_transfer(evaluate_cases, transfer_case(
        driver, [line_opacity(GAS, dict(type='uniform')), DUST],
        subchannels=2), PROPERTIES | dict(gas_op_a=1 / source))
    unabsorbed = extra.get('total_unabsorbed')
    line_tau = extra.get('total_line_tau')
    assert unabsorbed.spectral_axis == line_tau.spectral_axis == 2
    assert_same_model(unabsorbed.data, thin)
    assert_same_model(line_tau.data, thin / source)
    assert not extra.find('total_obdata')


def test_subchannels_option():
    from gbkfit.observation import Transfer
    from gbkfit.utils.parseutils import ConfigError
    assert Transfer(subchannels=4).dump()['subchannels'] == 4
    for subchannels in (0, 65, 2.5):
        with pytest.raises(ConfigError, match="from 1 to 64"):
            Transfer(subchannels=subchannels)


@pytest.mark.parametrize('incl, dispersion, subchannels, warnings', [
    (0, 20, 1, []),
    (0, 4, 1, ["raise the option subchannels of the line of sight to 4"]),
    (0, 4, 4, []),
    (90, 5, 4, ["does not resolve line opacity along the line of sight"])])
def test_a_coarse_march_of_line_opacity_warns(
        driver, evaluate_cases, caplog, incl, dispersion, subchannels,
        warnings):
    # Optically thick lines in channels of 10 km/s. Face-on (one velocity
    # on each line of sight), lines narrower than the channels need
    # sub-channels; edge-on, of a rotation curve flat almost from the
    # centre, the velocity changes along the line of sight by more than
    # the dispersion between adjacent voxels
    import logging
    properties = PROPERTIES | dict(
        gas_incl=incl, gas_dpt_a=dispersion, gas_vpt_rt=0.5)
    thin, _ = evaluate_transfer(
        evaluate_cases, transfer_case(driver, [GAS]), properties)
    caplog.set_level(logging.WARNING)
    evaluate_transfer(evaluate_cases, transfer_case(
        driver, [line_opacity(GAS, dict(type='uniform'))],
        subchannels=subchannels), properties | dict(gas_op_a=3 / thin.max()))
    logged = [
        record.getMessage() for record in caplog.records
        if 'radiative transfer does not resolve' in record.getMessage()]
    assert len(logged) == len(warnings)
    for warning, message in zip(warnings, logged):
        assert warning in message


@pytest.mark.parametrize('incl, depth_step, lsf_sigma, warns', [
    (90, None, None, True),
    (90, 0.05, None, False),
    (90, None, 100, False),
    (0, None, None, False)])
def test_a_coarse_depth_axis_warns_about_the_light(
        driver, evaluate_cases, caplog, incl, depth_step, lsf_sigma, warns):
    # Lines narrower than the change of the velocity between adjacent
    # voxels add up as a comb (edge-on, of a rotation curve flat almost
    # from the centre), unless a finer depth axis resolves it or the
    # instrument smooths it; face-on, each line of sight has one velocity
    import logging
    case = transfer_case(driver, [GAS])
    case['line_of_sight'] = dict(type='grid')
    if depth_step is not None:
        case['line_of_sight']['depth'] = dict(step=depth_step)
    if lsf_sigma is not None:
        case['observable'] |= dict(lsf=dict(type='gauss', sigma=lsf_sigma))
    caplog.set_level(logging.WARNING)
    evaluate_transfer(evaluate_cases, case, PROPERTIES | dict(
        gas_incl=incl, gas_dpt_a=5, gas_vpt_rt=0.5))
    assert warns == ('does not resolve the velocity of the light'
                     in caplog.text)


def test_lines_of_zero_dispersion_are_left_out_of_the_checks(
        driver, evaluate_cases):
    # No resolution resolves them: the checks of an evaluation with extras
    # (which divide by the dispersion) leave out optically thick lines of
    # zero dispersion
    gas = line_opacity(GAS, dict(type='uniform'))
    _, extra = evaluate_transfer(
        evaluate_cases, transfer_case(driver, [gas]),
        PROPERTIES | dict(gas_op_a=50, gas_dpt_a=0))
    assert extra.get('total_line_tau').data.max() > 1
