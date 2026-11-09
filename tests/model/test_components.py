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

THIN = dict(ptraits=EXPONENTIAL)
THICK = dict(ptraits=EXPONENTIAL, htraits=SECH2)
KINEMATICS = dict(type='traits', vptraits=ARCTAN, dptraits=UNIFORM)

BAND = dict(type='band')
LINES = dict(type='lines')
DUST = dict(type='dust')

# Each kind of disk component, with its required options
COMPONENTS = dict(
    band_smdisk_thin=dict(type='thin_disk', density=THIN, emission=BAND),
    band_smdisk_thick=dict(type='thick_disk', density=THICK, emission=BAND),
    band_mcdisk=dict(
        type='thick_disk', clouds=dict(cflux=1e-3), density=THICK,
        emission=BAND),
    lines_smdisk_thin=dict(
        type='thin_disk', density=THIN, kinematics=KINEMATICS,
        emission=LINES),
    lines_smdisk_thick=dict(
        type='thick_disk', density=THICK, kinematics=KINEMATICS,
        emission=LINES),
    lines_mcdisk=dict(
        type='thick_disk', clouds=dict(cflux=1e-3), density=THICK,
        kinematics=KINEMATICS, emission=LINES),
    dust_smdisk=dict(type='thick_disk', density=THICK, opacity=DUST),
    dust_mcdisk=dict(
        type='thick_disk', clouds=dict(cflux=1e-3), density=THICK,
        opacity=DUST))

# The options of a thick disk that a thin one does not take, with the
# part of the disk they are in (None: the disk itself) and a value
THICK_OPTIONS = [
    (None, 'opacity', DUST),
    (None, 'clouds', dict(cflux=1e-3)),
    (None, 'zptraits', dict(type='nw_uniform', zmax=1)),
    ('density', 'htraits', SECH2),
    ('kinematics', 'vhtraits', dict(type='one')),
    ('kinematics', 'dhtraits', dict(type='one')),
    ('kinematics', 'equilibrium', dict(type='central'))]

# The required trait options of each part of a disk
PART_TRAITS = dict(
    density=['ptraits'],
    kinematics=['vptraits', 'dptraits'])

# Each required trait option of each kind of component, with its part
REQUIRED_TRAITS = [
    (name, part, key)
    for name, component in COMPONENTS.items()
    for part, keys in PART_TRAITS.items() if part in component
    for key in keys]

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


def part_info(name, part, **options):
    """
    The configuration of a component of the given kind, with the options
    of its part (density or kinematics) updated (see updated).
    """
    component = component_info(name)
    return component | {part: updated(component[part], options)}


def load_error(component):
    """
    The error message of loading a galaxy with a bad component: of a
    mistake that the parsing finds (ConfigError), or a constructor
    (ValueError, TypeError).
    """
    with pytest.raises((ConfigError, ValueError, TypeError)) as error:
        galaxy_parser.load(dict(components=[component]))
    return located_message(error.value)


def part_error(name, part, **options):
    """
    The error message of loading a galaxy with a component of the given
    kind with bad options of its part, which names the part.
    """
    message = load_error(part_info(name, part, **options))
    assert message.startswith(f"components[0].{part}")
    return message


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


@pytest.mark.parametrize('name, part, key', REQUIRED_TRAITS)
def test_required_traits_cannot_be_missing(name, part, key):
    message = part_error(name, part, **{key: ...})
    assert f"missing required options: '{key}'" in message


@pytest.mark.parametrize('name, part, key', REQUIRED_TRAITS)
def test_required_traits_cannot_be_null(name, part, key):
    message = part_error(name, part, **{key: None})
    assert f"option '{key}' cannot be null" in message


@pytest.mark.parametrize('name, part, key', REQUIRED_TRAITS)
def test_required_traits_cannot_be_empty(name, part, key):
    message = part_error(name, part, **{key: []})
    assert f"at least one of {key} is required" in message


@pytest.mark.parametrize('htraits', [[], [SECH2, SECH2]])
@pytest.mark.parametrize('name', THICK_COMPONENTS)
def test_height_traits_must_match_polar_traits(name, htraits):
    message = part_error(name, 'density', htraits=htraits)
    assert (
        f"the number of htraits must be equal to "
        f"the number of ptraits ({len(htraits)} != 1)") in message


@pytest.mark.parametrize('name', COMPONENTS)
def test_a_disk_is_thin_or_thick_by_its_type(name):
    galaxy = galaxy_parser.load(dict(components=[component_info(name)]))
    thick = COMPONENTS[name]['type'] == 'thick_disk'
    for seen_by in (BandEmission, LineEmission):
        assert galaxy.components()[0].thick(seen_by) == thick


@pytest.mark.parametrize('htraits, message', [
    (..., "missing required options: 'htraits'"),
    (None, "option 'htraits' cannot be null")])
@pytest.mark.parametrize('name', THICK_COMPONENTS)
def test_a_thick_disk_needs_height_traits(name, htraits, message):
    assert message in part_error(name, 'density', htraits=htraits)


@pytest.mark.parametrize('part, option, value', THICK_OPTIONS)
def test_a_thin_disk_takes_no_options_of_a_thick_disk(part, option, value):
    name = 'lines_smdisk_thin'
    message = part_error(name, part, **{option: value}) if part \
        else component_error(name, **{option: value})
    assert (
        f"options of a thick disk (type: thick_disk), which a thin disk "
        f"does not take: '{option}'") in message


def test_a_lenient_thin_disk_leaves_out_the_options_of_a_thick_disk(caplog):
    component = part_info(
        'lines_smdisk_thin', 'kinematics', equilibrium=dict(type='central'))
    component |= dict(zptraits=dict(type='nw_uniform', zmax=1))
    with strict_mode(False):
        galaxy = galaxy_parser.load(dict(components=[component]))
    dumped = galaxy.components()[0].dump()
    assert 'zptraits' not in dumped
    assert 'equilibrium' not in dumped['kinematics']
    assert "a thin disk does not take: 'zptraits'" in caplog.text
    assert "a thin disk does not take: 'equilibrium'" in caplog.text


@pytest.mark.parametrize('name', LINE_COMPONENTS)
def test_a_disk_that_emits_lines_needs_kinematics(name):
    message = component_error(name, kinematics=...)
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
    galaxy = galaxy_parser.load(dict(components=[part_info(
        name, 'kinematics', vptraits=[ARCTAN, dict(type='nw_uniform', direction='rad')])]))
    assert {'vpt_vt', 'vpt1_vr'} <= set(galaxy.pdescs())
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
    component = part_info('lines_smdisk_thick', 'kinematics', vptraits=[
        ARCTAN, dict(type='nw_uniform', direction='rad', sampling='subringz')])
    assert load_error(component).startswith(
        "components[0].kinematics.vptraits[1] [nw_uniform]: "
        "option 'sampling' must be")


def test_at_most_max_traits_of_a_kind():
    # The native kernels take at most MAX_TRAITS traits of each kind
    from gbkfit.model.components._nodal import MAX_TRAITS
    message = part_error(
        'band_smdisk_thin', 'density',
        ptraits=[EXPONENTIAL] * (MAX_TRAITS + 1))
    assert f"at most {MAX_TRAITS} ptraits" in message


@pytest.mark.parametrize('trait, message', [
    (dict(type='nw_harmonic', order=-1), "'order' must be at least 0"),
    (dict(type='nw_harmonic', order=1.5), "'order' must be of type int"),
    (dict(type='mixture_gauss', nblobs=0), "'nblobs' must be at least 1")])
def test_orders_and_blobs_are_checked(trait, message):
    # The kernels read the phases of the orders above 0, and the
    # parameters of each blob
    assert message in part_error('band_smdisk_thin', 'density', ptraits=trait)


@pytest.mark.parametrize('zmax', [0, -1])
def test_zmax_must_be_a_positive_length(zmax):
    message = part_error(
        'band_smdisk_thick', 'density', htraits=dict(type='sech2', zmax=zmax))
    assert "'zmax' must be finite and greater than 0" in message
