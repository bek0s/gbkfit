"""
Tests for the options of the galaxies and of the lines of sight of their
observations: which components each kind of observation sees, and which
options they accept and reject.
"""

import logging
import re

import pytest
from gbkfit.model import galaxy_parser
from gbkfit.model.components._nodal import parse_component_rnode_args
from gbkfit.observation import Depth
from gbkfit.utils import gridutils, parseutils
from modelutils import config_group_plan


EXPONENTIAL = dict(type='exponential')
SECH2 = dict(type='sech2', zmax=8)
ARCTAN = dict(type='arctan', direction='tan')
UNIFORM = dict(type='uniform')

DISK = dict(loose=False, tilted=False, rnodes=[0, 2, 4, 6, 8])

THIN = dict(ptraits=EXPONENTIAL)
THICK = dict(ptraits=EXPONENTIAL, htraits=SECH2)
KINEMATICS = dict(type='traits', vptraits=ARCTAN, dptraits=UNIFORM)

# A component of each kind, with its required options
BAND_THIN = dict(DISK, type='disk', density=THIN, emission=dict(type='band'))
BAND_THICK = BAND_THIN | dict(density=THICK)
LINES_THIN = BAND_THIN | dict(
    kinematics=KINEMATICS, emission=dict(type='lines'))
LINES_THICK = LINES_THIN | dict(density=THICK)
DUST = dict(DISK, type='disk', density=THICK, opacity=dict(type='dust'))

COMPONENTS = dict(
    band_thin=BAND_THIN, band_thick=BAND_THICK, lines_thin=LINES_THIN,
    lines_thick=LINES_THICK, dust=DUST)

FLAT = dict(type='flat')
GRID = dict(type='grid')

IMAGE = dict(type='pixel_brightness', size=[8, 8])
CUBE = dict(type='pixel_spectra', size=[8, 8, 8])

# Each kind of observation (an image or a spectral cube, along a flat or
# a grid line of sight): its observable and line of sight, and a
# component it sees
OBSERVATIONS = dict(
    image_flat=(IMAGE, FLAT, BAND_THIN),
    image_grid=(IMAGE, GRID, BAND_THICK),
    cube_flat=(CUBE, FLAT, LINES_THIN),
    cube_grid=(CUBE, GRID, LINES_THICK))

FLAT_KINDS = ['image_flat', 'cube_flat']
GRID_KINDS = ['image_grid', 'cube_grid']

# A depth axis of a grid line of sight, other than the default one
DEPTH = dict(extent=10, step=1)


def config(kind, components=None, line_of_sight=None):
    """
    The configuration of an observation of the given kind, of a galaxy
    of the given components (by default, the one the observation sees),
    with the given line of sight (by default, that of the kind).
    """
    observable, default_line_of_sight, component = OBSERVATIONS[kind]
    if components is None:
        components = [component]
    if line_of_sight is None:
        line_of_sight = default_line_of_sight
    return dict(
        galaxies=[dict(components=components)],
        observations=[dict(
            type='sky',
            observable=observable, line_of_sight=line_of_sight)])


def with_dust(kind):
    """
    The components of a galaxy with dust: the one that the given kind of
    observation sees, and the dust (with names: there are two).
    """
    _, _, component = OBSERVATIONS[kind]
    return [component | dict(name='stars'), DUST | dict(name='dust')]


def load_error(info):
    """The error message of loading a bad configuration."""
    with pytest.raises(parseutils.ConfigError) as error:
        config_group_plan(info)
    return str(error.value)


def galaxy_error(info):
    """The error message of loading a bad galaxy."""
    with pytest.raises(parseutils.ConfigError) as error:
        galaxy_parser.load(info)
    return str(error.value)


@pytest.mark.parametrize('kind', OBSERVATIONS)
def test_observation_loads(kind):
    config_group_plan(config(kind))


@pytest.mark.parametrize('info', [dict(), dict(components=[])])
def test_galaxies_without_components_need_mass(info):
    message = galaxy_error(info)
    assert "a galaxy needs at least one component, or mass" in message


@pytest.mark.parametrize('kind', FLAT_KINDS)
def test_flat_lines_of_sight_do_not_see_mcdisks(kind):
    # A Monte Carlo disk is thick
    _, _, component = OBSERVATIONS[kind]
    mcdisk = component | dict(
        type='disk', clouds=dict(cflux=1e-3), density=THICK)
    message = load_error(config(kind, [mcdisk]))
    assert "a thick component needs a line of sight with a depth axis" \
        in message


IMAGE_SEES_NONE = (
    "the observation is an image, which sees the components that emit in "
    "a band; it sees none")
CUBE_SEES_NONE = (
    "the observation is a spectral cube, which sees the components that "
    "emit lines (and stars, when its lines name 'stars'); it sees none")
FLAT_SEES_NO_THICK = (
    "a thick component needs a line of sight with a depth axis "
    "(grid or transfer)")


@pytest.mark.parametrize('kind, component, message', [
    ('image_flat', LINES_THIN, IMAGE_SEES_NONE),
    ('image_grid', LINES_THICK, IMAGE_SEES_NONE),
    ('cube_flat', BAND_THIN, CUBE_SEES_NONE),
    ('cube_grid', BAND_THICK, CUBE_SEES_NONE),
    ('image_grid', DUST, IMAGE_SEES_NONE),
    ('cube_grid', DUST, CUBE_SEES_NONE),
    ('cube_flat', LINES_THICK, FLAT_SEES_NO_THICK),
    ('image_flat', BAND_THICK, FLAT_SEES_NO_THICK)])
def test_observations_reject_components_of_other_kinds(
        kind, component, message):
    assert message in load_error(config(kind, [component]))


@pytest.mark.parametrize('kind', GRID_KINDS)
def test_opacity_cannot_be_selected(kind):
    # The opacity of every component applies: an observation selects the
    # components whose light it sees
    info = config(kind, with_dust(kind))
    info['observations'][0]['components'] = ['dust']
    message = load_error(info)
    assert "the component 'dust' is selected, but the observation is" \
        in message


@pytest.mark.parametrize('kind', GRID_KINDS)
def test_grids_have_depth_options(kind):
    config_group_plan(
        config(kind, with_dust(kind), GRID | dict(depth=DEPTH)))


@pytest.mark.parametrize('kind', FLAT_KINDS)
def test_flat_lines_of_sight_have_no_depth(kind):
    line_of_sight = FLAT | dict(depth=DEPTH)
    message = load_error(config(kind, line_of_sight=line_of_sight))
    assert message.startswith(
        "observations[0].line_of_sight [flat]: unknown options")
    assert "'depth'" in message


@pytest.mark.parametrize('kind', FLAT_KINDS)
def test_flat_lines_of_sight_see_no_opacity(kind):
    message = load_error(config(kind, with_dust(kind)))
    assert "component 'dust': " in message
    assert "needs a line of sight with a depth axis (grid or transfer)" \
        in message


@pytest.mark.parametrize('options, message', [
    (dict(extent=-1), "extent must be 'galaxy', 'legacy' or a positive"),
    (dict(extent='half'), "extent must be 'galaxy', 'legacy' or a positive"),
    (dict(step=0), "step must be positive")])
def test_invalid_depths_are_rejected(options, message):
    line_of_sight = GRID | dict(depth=options)
    error = load_error(config('image_grid', line_of_sight=line_of_sight))
    assert error.startswith("observations[0].line_of_sight.depth: ")
    assert message in error


@pytest.mark.parametrize('depth, axis', [
    # As many voxels as cover the extent, centred on 0
    (Depth(extent=21, step=1.5), (14, 1.5, -9.75)),
    (Depth(extent=20, step=1.5), (14, 1.5, -9.75)),
    # By default, the diameter of the disks (radius 5), in voxels of the
    # step of the longer of the x and y axes
    (Depth(), (20, 0.5, -4.75)),
    # Legacy: those of the longer of the x and y axes
    (Depth(extent='legacy'), (12, 0.5, -2.75)),
    (Depth(extent='legacy', step=0.25), (12, 0.25, -1.375))])
def test_depth_axis(depth, axis):
    grid = gridutils.Grid(
        (8, 12), gridutils.Coords((1, 0.5), (3.5, 5.5), (0, 0), 0), None)
    assert depth.axis(grid, 5) == axis


@pytest.mark.parametrize('name', COMPONENTS)
def test_unknown_trait_options_are_errors(name):
    component = COMPONENTS[name]
    density = component['density'] | dict(ptraits=dict(EXPONENTIAL, foo=1))
    message = galaxy_error(dict(components=[
        component | dict(density=density)]))
    assert message.startswith("components[0].density.ptraits")
    assert "'foo'" in message


@pytest.mark.parametrize('name', COMPONENTS)
def test_unknown_options_are_warnings_in_lenient_mode(name, caplog):
    info = dict(components=COMPONENTS[name] | dict(foo=1))
    with parseutils.strict_mode(False), caplog.at_level(logging.WARNING):
        galaxy_parser.load(info)
    assert "'foo'" in caplog.text


def test_trait_names_prefix_their_parameters():
    # (in place of the kind and position, which an inserted trait moves)
    inflow = dict(type='uniform', direction='rad', name='inflow')
    rotation = ARCTAN | dict(name='rot')
    info = dict(components=[LINES_THIN | dict(kinematics=dict(
        type='traits', vptraits=[inflow, rotation], dptraits=UNIFORM))])
    galaxy = galaxy_parser.load(info)
    assert {'inflow_vr', 'rot_rt', 'rot_vt', 'dpt_a'} <= set(galaxy.pdescs())
    dumped = galaxy_parser.dump(galaxy)
    vptraits = dumped['components'][0]['kinematics']['vptraits']
    assert [trait['name'] for trait in vptraits] == ['inflow', 'rot']


@pytest.mark.parametrize('vptraits, dptraits, message', [
    ([ARCTAN | dict(name='rot'), ARCTAN], UNIFORM,
     "either all or none of the traits must have a name"),
    ([ARCTAN | dict(name='rot')], UNIFORM | dict(name='rot'),
     r"cannot share a name; repeated: \['rot'\]")])
def test_trait_names_are_all_or_none_and_unique(vptraits, dptraits, message):
    info = dict(components=[LINES_THIN | dict(kinematics=dict(
        type='traits', vptraits=vptraits, dptraits=dptraits))])
    assert re.search(message, galaxy_error(info))


@pytest.mark.parametrize('rnmin, rnmax, rnsep, nodes', [
    # (0.2 * 11 rounds above 2.2: np.arange to 2.4 added a node)
    (0, 2.2, 0.2, 12), (0, 9.0, 0.3, 31), (1, 2, 0.25, 5),
    # Not a multiple: to the first node beyond rnmax
    (0, 10, 3, 5)])
def test_radial_nodes_from_a_range(rnmin, rnmax, rnsep, nodes):
    args = parse_component_rnode_args(
        rnmin, rnmax, rnsep, None, None, None, 'linear')
    assert len(args['rnodes']) == nodes
    assert args['rnodes'][-1] == pytest.approx(rnmin + (nodes - 1) * rnsep)


def test_radial_step_can_be_half_the_node_separation():
    # (half of 0.19999999999999996, the separation as it rounds)
    args = parse_component_rnode_args(
        None, None, None, None, [0, 0.2, 0.4, 0.6, 0.8, 1.0], 0.1, 'linear')
    assert args['rstep'] == 0.1


def test_discouraged_traits_are_warnings_with_the_path(caplog):
    component = BAND_THIN | dict(density=dict(ptraits=UNIFORM))
    with parseutils.config_path('galaxies'):
        galaxy_parser.load([dict(components=component)])
    assert "galaxies[0].components [disk]: the use of" in caplog.text
