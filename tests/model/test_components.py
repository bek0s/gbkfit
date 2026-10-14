"""
Tests for the options of the components of galaxies (their parts, roles
and traits), and their errors.
"""


import numpy as np
import pytest
from gbkfit.model import galaxy_parser
from gbkfit.model.components._component_plans import opacity_cubes
from gbkfit.observation import Depth, Grid
from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import ConfigError


EXPONENTIAL = dict(type='exponential')
SECH2 = dict(type='sech2', zmax=8)
ARCTAN = dict(type='tan_arctan')
UNIFORM = dict(type='uniform')

DISK = dict(loose=False, tilted=False, rnodes=[0, 2, 4, 6, 8])

THIN = dict(ptraits=EXPONENTIAL)
THICK = dict(ptraits=EXPONENTIAL, htraits=SECH2)
KINEMATICS = dict(vptraits=ARCTAN, dptraits=UNIFORM)

BAND = dict(type='band')
LINES = dict(type='lines')
DUST = dict(type='dust')

# Each kind of disk component, with its required options
COMPONENTS = dict(
    band_smdisk_thin=dict(type='smdisk', density=THIN, emission=BAND),
    band_smdisk_thick=dict(type='smdisk', density=THICK, emission=BAND),
    band_mcdisk=dict(
        type='mcdisk', cflux=1e-3, density=THICK, emission=BAND),
    lines_smdisk_thin=dict(
        type='smdisk', density=THIN, kinematics=KINEMATICS, emission=LINES),
    lines_smdisk_thick=dict(
        type='smdisk', density=THICK, kinematics=KINEMATICS, emission=LINES),
    lines_mcdisk=dict(
        type='mcdisk', cflux=1e-3, density=THICK, kinematics=KINEMATICS,
        emission=LINES),
    dust_smdisk=dict(type='smdisk', density=THICK, opacity=DUST),
    dust_mcdisk=dict(type='mcdisk', cflux=1e-3, density=THICK, opacity=DUST))

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

# Each kind of thick component (with density height traits)
THICK_COMPONENTS = [
    name for name, component in COMPONENTS.items()
    if 'htraits' in component['density']]

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
    """The error message of loading a galaxy with a bad component."""
    with pytest.raises(ConfigError) as error:
        galaxy_parser.load(dict(components=[component]))
    return str(error.value)


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
    assert f"option '{key}' is required but not provided" in message


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


@pytest.mark.parametrize('htraits', [..., None])
@pytest.mark.parametrize('name', ['band_smdisk_thick', 'lines_smdisk_thick'])
def test_a_smooth_disk_without_height_traits_is_thin(name, htraits):
    galaxy = galaxy_parser.load(dict(components=[
        part_info(name, 'density', htraits=htraits)]))
    assert not galaxy.components()[0].thick()


@pytest.mark.parametrize('name, message', [
    ('band_mcdisk', "a Monte Carlo disk needs density height traits"),
    ('lines_mcdisk', "a Monte Carlo disk needs density height traits"),
    ('dust_smdisk', "a disk with opacity needs density height traits"),
    ('dust_mcdisk', "a Monte Carlo disk needs density height traits")])
def test_thick_components_need_height_traits(name, message):
    assert message in component_error(name, density=THIN)


@pytest.mark.parametrize('name, options, message', [
    ('band_smdisk_thin', dict(zptraits=dict(type='nw_uniform', zmax=1)),
     "vertical distortion traits (zptraits) need density height traits"),
    ('lines_smdisk_thin',
     dict(kinematics=KINEMATICS | dict(vhtraits=dict(type='one'))),
     "the height traits of the kinematics need density height traits"),
    ('lines_smdisk_thin',
     dict(kinematics=KINEMATICS | dict(dhtraits=dict(type='one'))),
     "the height traits of the kinematics need density height traits")])
def test_thick_options_need_a_thick_disk(name, options, message):
    assert message in component_error(name, **options)


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
        "components[0] [point]: option 'emission' is required but not "
        "provided")


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
        name, 'kinematics', vptraits=[ARCTAN, dict(type='nw_rad_uniform')])]))
    assert {'vpt_vt', 'vpt1_vr'} <= set(galaxy.pdescs())
    params = {
        pname: np.ones(pdesc.size()) if pdesc.type() == 'vector' else 1.0
        for pname, pdesc in galaxy.pdescs().items()}
    scube = np.zeros((11, 16, 16), np.float32)
    grid = gridutils.Grid((16, 16, 11), gridutils.Coords(
        (1, 1, 10), (7.5, 7.5, 5), (0, 0, 0), 0), 2)
    spatial = grid.spatial()
    plan = galaxy.plan(
        DriverHost(), spatial, grid.spectral(),
        Grid(Depth(extent='legacy')).depth_axis(spatial, 0),
        np.float32, (0,),
        opacity_cubes([galaxy], [(0,)], None, None, np.float32)[0],
        None, None, None)
    plan.evaluate(params, {}, dict(scube=scube), None)
    assert scube.sum() > 0


def test_errors_have_paths():
    component = part_info('lines_smdisk_thick', 'kinematics', vptraits=[
        ARCTAN, dict(type='nw_rad_uniform', sampling='subringz')])
    assert load_error(component).startswith(
        "components[0].kinematics.vptraits[1] [nw_rad_uniform]: "
        "sampling must be")


def test_at_most_4_traits_of_a_kind():
    # The native kernels take at most 4 traits of each kind
    message = part_error(
        'band_smdisk_thin', 'density', ptraits=[EXPONENTIAL] * 5)
    assert "at most 4 ptraits" in message


@pytest.mark.parametrize('trait, message', [
    (dict(type='nw_harmonic', order=-1), "order must be an integer"),
    (dict(type='nw_harmonic', order=1.5), "'order' must be of type int"),
    (dict(type='mixture_gauss', nblobs=0), "nblobs must be at least 1")])
def test_orders_and_blobs_are_checked(trait, message):
    # The kernels read the phases of the orders above 0, and the
    # parameters of each blob
    assert message in part_error('band_smdisk_thin', 'density', ptraits=trait)


@pytest.mark.parametrize('zmax', [0, -1])
def test_zmax_must_be_a_positive_length(zmax):
    message = part_error(
        'band_smdisk_thick', 'density', htraits=dict(type='sech2', zmax=zmax))
    assert "zmax must be a positive length" in message
