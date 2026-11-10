"""
Tests for the options of the components of galaxies (their parts, roles
and traits), and their errors.
"""


import numpy as np
import pytest
from gbkfit.config import ConfigError, strict_mode
from gbkfit.grid import Coords, Grid
from gbkfit.model import BandEmission, LineEmission, galaxy_parser
from gbkfit.model._component_plans import opacity_cubes
from gbkfit.observation import Depth, LineOfSightGrid
from modelutils import located_message


EXPONENTIAL = dict(type='exponential')
SECH2 = dict(type='sech2', zmax=8)
ARCTAN = dict(type='arctan', direction='tan')
UNIFORM = dict(type='uniform')

DISK = dict(loose=False, tilted=False, rnodes=[0, 2, 4, 6, 8])

THIN = dict(rptraits=EXPONENTIAL)
THICK = dict(rptraits=EXPONENTIAL, rhtraits=SECH2)
KINEMATICS = dict(vptraits=ARCTAN, dptraits=UNIFORM)

BAND = dict(type='band')
LINES = dict(type='lines')
DUST = dict(type='dust')

# Each kind of disk component, with its required options
COMPONENTS = dict(
    band_smdisk_thin=dict(type='thin_disk', **THIN, emission=BAND),
    band_smdisk_thick=dict(type='thick_disk', **THICK, emission=BAND),
    band_mcdisk=dict(
        type='thick_disk', clouds=dict(cflux=1e-3), **THICK,
        emission=BAND),
    lines_smdisk_thin=dict(
        type='thin_disk', **THIN, **KINEMATICS,
        emission=LINES),
    lines_smdisk_thick=dict(
        type='thick_disk', **THICK, **KINEMATICS,
        emission=LINES),
    lines_mcdisk=dict(
        type='thick_disk', clouds=dict(cflux=1e-3), **THICK,
        **KINEMATICS, emission=LINES),
    dust_smdisk=dict(type='thick_disk', **THICK, opacity=DUST),
    dust_mcdisk=dict(
        type='thick_disk', clouds=dict(cflux=1e-3), **THICK,
        opacity=DUST))

# The options of a thick disk that a thin one does not take, with a value
THICK_OPTIONS = [
    ('opacity', DUST),
    ('clouds', dict(cflux=1e-3)),
    ('zptraits', dict(type='nw_uniform', zmax=1)),
    ('rhtraits', SECH2),
    ('vhtraits', dict(type='one')),
    ('dhtraits', dict(type='one')),
    ('equilibrium', dict(type='central'))]

# Each kind of component with its required trait option
REQUIRED_TRAITS = [(name, 'rptraits') for name in COMPONENTS]

# Each kind of thick component (a thick disk)
THICK_COMPONENTS = [
    name for name, component in COMPONENTS.items()
    if component['type'] == 'thick_disk']

# Each kind of component that emits lines
LINE_COMPONENTS = [
    name for name, component in COMPONENTS.items()
    if component.get('emission') == LINES]


def updated(info, options):
    """The info with its options updated; an option set to ... is left out."""
    info = info | options
    return {key: value for key, value in info.items() if value is not ...}


def component_info(name, **options):
    """
    The configuration of a component of the given kind, with its options
    updated (see updated).
    """
    return updated(dict(DISK, **COMPONENTS[name]), options)


def load_error(component):
    """
    The error message of loading a galaxy with a bad component: of a
    mistake that the parsing finds (ConfigError), or a constructor
    (ValueError, TypeError).
    """
    with pytest.raises((ConfigError, ValueError, TypeError)) as error:
        galaxy_parser.load(dict(components=[component]))
    return located_message(error.value)


def component_error(name, **options):
    """
    The error message of loading a galaxy with a component of the given
    kind with bad options, which names the type of the component.
    """
    message = load_error(component_info(name, **options))
    assert message.startswith(
        f"components[0] [{COMPONENTS[name]['type']}]: ")
    return message


@pytest.mark.parametrize('name', COMPONENTS)
def test_component_loads(name):
    galaxy_parser.load(dict(components=[component_info(name)]))


@pytest.mark.parametrize('name, key', REQUIRED_TRAITS)
def test_required_traits_cannot_be_missing(name, key):
    message = component_error(name, **{key: ...})
    assert f"missing required options: '{key}'" in message


@pytest.mark.parametrize('name, key', REQUIRED_TRAITS)
def test_required_traits_cannot_be_null(name, key):
    message = component_error(name, **{key: None})
    assert f"option '{key}' cannot be null" in message


@pytest.mark.parametrize('name, key', REQUIRED_TRAITS)
def test_required_traits_cannot_be_empty(name, key):
    message = component_error(name, **{key: []})
    assert f"at least one of {key} is required" in message


@pytest.mark.parametrize('key', ['vptraits', 'dptraits'])
@pytest.mark.parametrize('name', LINE_COMPONENTS)
def test_a_disk_that_moves_takes_velocity_and_dispersion_traits(name, key):
    message = component_error(name, **{key: ...})
    assert "takes both its velocity traits (vptraits) and its dispersion " \
        "traits (dptraits)" in message


@pytest.mark.parametrize('rhtraits', [[], [SECH2, SECH2, SECH2]])
@pytest.mark.parametrize('name', THICK_COMPONENTS)
def test_height_traits_are_one_shared_or_one_for_each(name, rhtraits):
    message = component_error(
        name, rptraits=[EXPONENTIAL, EXPONENTIAL], rhtraits=rhtraits)
    assert (
        f"rhtraits takes one trait, which every one of the rptraits shares, "
        f"or one for each of them (2); {len(rhtraits)} were given") \
        in message


@pytest.mark.parametrize('rhtraits', [SECH2, [SECH2]])
def test_one_height_trait_is_shared_by_every_polar_trait(rhtraits):
    # One trait, alone or in a list of one: one set of parameters, which
    # Studio shows with the trait; its configuration keeps one
    galaxy = galaxy_parser.load(dict(components=[component_info(
        'band_smdisk_thick', rptraits=[EXPONENTIAL, EXPONENTIAL],
        rhtraits=rhtraits)]))
    names = set(galaxy.pdescs())
    assert {'rpt_s', 'rpt2_s', 'rht_s'} <= names and 'rht2_s' not in names
    assert galaxy.param_owners()['rht_s'] == ('components', 0, 'rhtraits', 0)
    dumped = galaxy.dump()['components'][0]
    assert dumped['rhtraits'] == [SECH2]
    assert galaxy_parser.load(dict(components=[dumped])).dump() \
        == galaxy.dump()


def test_a_shared_height_trait_is_one_for_each_polar_trait(
        driver, evaluate_cases):
    # The same model as a copy of it for each polar trait, of the same
    # parameters
    rptraits = [EXPONENTIAL, dict(type='gauss')]

    def evaluate(rhtraits, values):
        disk = component_info(
            'band_smdisk_thick', rptraits=rptraits, rhtraits=rhtraits)
        case = dict(
            driver=dict(type=driver.type()), galaxy=dict(components=[disk]),
            observable=dict(
                type='pixel_brightness', size=[24, 24], step=[1, 1]),
            line_of_sight=dict(type='grid', depth=dict(step=0.5)))
        data, _ = evaluate_cases([case], dict(
            xpos=0.5, ypos=-0.5, posa=30, incl=70, rpt_a=1, rpt_s=3,
            rpt2_a=0.5, rpt2_s=2) | values)
        return data[0]['brightness']['d']
    shared = evaluate(SECH2, dict(rht_s=0.7))
    each = evaluate([SECH2, SECH2], dict(rht_s=0.7, rht2_s=0.7))
    np.testing.assert_allclose(shared, each, rtol=1e-6)
    other = evaluate([SECH2, SECH2], dict(rht_s=0.7, rht2_s=1.5))
    assert not np.allclose(shared, other, rtol=1e-3)


@pytest.mark.parametrize('name', COMPONENTS)
def test_a_disk_is_thin_or_thick_by_its_type(name):
    galaxy = galaxy_parser.load(dict(components=[component_info(name)]))
    thick = COMPONENTS[name]['type'] == 'thick_disk'
    for seen_by in (BandEmission, LineEmission):
        assert galaxy.components()[0].thick(seen_by) == thick


@pytest.mark.parametrize('rhtraits, message', [
    (..., "missing required options: 'rhtraits'"),
    (None, "option 'rhtraits' cannot be null")])
@pytest.mark.parametrize('name', THICK_COMPONENTS)
def test_a_thick_disk_needs_height_traits(name, rhtraits, message):
    assert message in component_error(name, rhtraits=rhtraits)


@pytest.mark.parametrize('option, value', THICK_OPTIONS)
def test_a_thin_disk_takes_no_options_of_a_thick_disk(option, value):
    name = 'lines_smdisk_thin'
    message = component_error(name, **{option: value})
    assert (
        f"options of a thick disk (type: thick_disk), which a thin disk "
        f"does not take: '{option}'") in message


def test_a_lenient_thin_disk_leaves_out_the_options_of_a_thick_disk(caplog):
    component = component_info(
        'lines_smdisk_thin', equilibrium=dict(type='central'),
        zptraits=dict(type='nw_uniform', zmax=1))
    with strict_mode(False):
        galaxy = galaxy_parser.load(dict(components=[component]))
    dumped = galaxy.components()[0].dump()
    assert 'zptraits' not in dumped
    assert 'equilibrium' not in dumped
    assert "a thin disk does not take: 'equilibrium', 'zptraits'" \
        in caplog.text


@pytest.mark.parametrize('name', LINE_COMPONENTS)
def test_a_disk_that_emits_lines_needs_kinematics(name):
    message = component_error(name, vptraits=..., dptraits=...)
    assert "a disk that emits lines needs kinematics" in message


@pytest.mark.parametrize('name', COMPONENTS)
def test_a_component_needs_emission_or_opacity(name):
    message = component_error(name, emission=..., opacity=...)
    assert "a component needs emission, opacity or both" in message


def test_a_point_needs_emission():
    # A point has no depth, so it has no opacity
    message = load_error(dict(type='point'))
    assert message == (
        "components[0] [point]: missing required options: 'emission'")


@pytest.mark.parametrize('name', COMPONENTS)
def test_optional_traits_can_be_null(name):
    galaxy_parser.load(dict(components=[
        component_info(name, sptraits=None)]))


@pytest.mark.parametrize('name', ['lines_smdisk_thick', 'lines_mcdisk'])
def test_thick_disk_with_several_velocity_traits(name):
    # Rotation and radial motions, each with the default height trait.
    # The disk needs a height trait for each velocity trait to evaluate.
    from gbkfit.driver.drivers.host import DriverHost
    galaxy = galaxy_parser.load(dict(components=[component_info(
        name, vptraits=[ARCTAN, dict(type='nw_uniform', direction='rad')])]))
    assert {'vpt_vt', 'vpt2_vr'} <= set(galaxy.pdescs())
    params = {
        pname: np.ones(pdesc.size()) if pdesc.type() == 'vector' else 1.0
        for pname, pdesc in galaxy.pdescs().items()}
    scube = np.zeros((11, 16, 16), np.float32)
    grid = Grid((16, 16, 11), Coords(
        (1, 1, 10), (7.5, 7.5, 5), (0, 0, 0), 0), 2)
    spatial = grid.spatial()
    plan = galaxy.plan(
        DriverHost(), spatial, grid.spectral(),
        LineOfSightGrid(Depth(extent='legacy')).depth_axis(spatial, 0),
        np.float32, (0,),
        opacity_cubes([galaxy], [(0,)], None, None, np.float32)[0],
        None, None, None, None)
    plan.evaluate(params, {}, dict(scube=scube), None)
    assert scube.sum() > 0


def test_errors_have_paths():
    component = component_info('lines_smdisk_thick', vptraits=[
        ARCTAN, dict(type='nw_uniform', direction='rad', sampling='subringz')])
    assert load_error(component).startswith(
        "components[0].vptraits[1] [nw_uniform]: "
        "option 'sampling' must be")


def test_at_most_max_traits_of_a_kind():
    # The native kernels take at most MAX_TRAITS traits of each kind
    from gbkfit.model.components._nodal import MAX_TRAITS
    message = component_error(
        'band_smdisk_thin', rptraits=[EXPONENTIAL] * (MAX_TRAITS + 1))
    assert f"at most {MAX_TRAITS} rptraits" in message


@pytest.mark.parametrize('trait, message', [
    (dict(type='nw_harmonic', order=-1), "'order' must be at least 0"),
    (dict(type='nw_harmonic', order=1.5), "'order' must be of type int"),
    (dict(type='mixture_gauss', nblobs=0), "'nblobs' must be at least 1")])
def test_orders_and_blobs_are_checked(trait, message):
    # The kernels read the phases of the orders above 0, and the
    # parameters of each blob
    assert message in load_error(component_info(
        'band_smdisk_thin', rptraits=trait))


@pytest.mark.parametrize('zmax', [0, -1])
def test_zmax_must_be_a_positive_length(zmax):
    message = load_error(component_info(
        'band_smdisk_thick', rhtraits=dict(type='sech2', zmax=zmax)))
    assert "'zmax' must be finite and greater than 0" in message
