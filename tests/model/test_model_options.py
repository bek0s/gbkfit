"""
Tests for the options of the galaxies and of the depth axes of their
observations: which components each kind of observation sees, how the
components decide its evaluation along the line of sight, and which
options they accept and reject.
"""

import logging
import re

import pytest
from gbkfit.config import ConfigError, config_path, strict_mode
from gbkfit.grid import Coords, Grid
from gbkfit.model import Nodes, SceneEvaluation, galaxy_parser
from gbkfit.model.components._nodal import parse_component_rnode_args
from gbkfit.observation import Depth
from modelutils import config_group, config_group_plan, located_message


EXPONENTIAL = dict(type='exponential')
SECH2 = dict(type='sech2', zmax=8)
ARCTAN = dict(type='arctan', direction='tan')
UNIFORM = dict(type='uniform')

DISK = dict(loose=False, tilted=False, rnodes=[0, 2, 4, 6, 8])

THIN = dict(rptraits=EXPONENTIAL)
THICK = dict(rptraits=EXPONENTIAL, rhtraits=SECH2)
KINEMATICS = dict(vptraits=ARCTAN, dptraits=UNIFORM)

# A component of each kind, with its required options
BAND_THIN = dict(
    DISK, type='thin_disk', **THIN, emission=dict(type='band'))
BAND_THICK = BAND_THIN | dict(type='thick_disk', **THICK)
LINES_THIN = BAND_THIN | dict(
    **KINEMATICS, emission=dict(type='lines'))
LINES_THICK = LINES_THIN | dict(type='thick_disk', **THICK)
DUST = dict(DISK, type='thick_disk', **THICK, opacity=dict(type='dust'))

COMPONENTS = dict(
    band_thin=BAND_THIN, band_thick=BAND_THICK, lines_thin=LINES_THIN,
    lines_thick=LINES_THICK, dust=DUST)

IMAGE = dict(type='pixel_brightness', size=[8, 8])
CUBE = dict(
    type='pixel_spectra', size=[8, 8],
    spectral=dict(size=8, convention='relativistic'))

# Each kind of observation (an image or a spectral cube, of thin or
# thick components): its observable, and a component it sees
OBSERVATIONS = dict(
    image_thin=(IMAGE, BAND_THIN),
    image_thick=(IMAGE, BAND_THICK),
    cube_thin=(CUBE, LINES_THIN),
    cube_thick=(CUBE, LINES_THICK))

THIN_KINDS = ['image_thin', 'cube_thin']

# A depth axis other than the default one
DEPTH = dict(extent=10, step=1)


def config(kind, components=None, depth=None):
    """
    The configuration of an observation of the given kind, of a galaxy
    of the given components (by default, the one the observation sees),
    with the given depth axis, if any.
    """
    kind_options, component = OBSERVATIONS[kind]
    if components is None:
        components = [component]
    return dict(
        galaxies=[dict(components=components)],
        observations=[
            kind_options | ({} if depth is None else dict(depth=depth))])


def evaluation(info):
    """
    How the observation of a configuration is evaluated along the line of
    sight (see Scene.evaluation), which the components it sees decide.
    """
    group = config_group(info)
    return group.observations()[0]._scene(group.seen_by(0)).evaluation()


def with_dust(kind):
    """
    The components of a galaxy with dust: the one that the given kind of
    observation sees, and the dust (with names: there are two).
    """
    _, component = OBSERVATIONS[kind]
    return [component | dict(name='stars'), DUST | dict(name='dust')]


def load_error(info):
    """
    The error message of loading a bad configuration: of a mistake
    that the parsing finds (ConfigError), or a constructor
    (ValueError, TypeError).
    """
    with pytest.raises((ConfigError, ValueError, TypeError)) as error:
        config_group_plan(info)
    return located_message(error.value)


def galaxy_error(info):
    """
    The error message of loading a bad galaxy: of a mistake that the
    parsing finds (ConfigError), or a constructor (ValueError, TypeError).
    """
    with pytest.raises((ConfigError, ValueError, TypeError)) as error:
        galaxy_parser.load(info)
    return located_message(error.value)


@pytest.mark.parametrize('kind', OBSERVATIONS)
def test_observation_loads(kind):
    config_group_plan(config(kind))


@pytest.mark.parametrize('info', [dict(), dict(components=[])])
def test_galaxies_without_components_need_mass(info):
    message = galaxy_error(info)
    assert "a galaxy needs at least one component, or mass" in message


@pytest.mark.parametrize('kind, expected', [
    ('image_thin', SceneEvaluation.PLANE),
    ('cube_thin', SceneEvaluation.PLANE),
    ('image_thick', SceneEvaluation.DEPTH),
    ('cube_thick', SceneEvaluation.DEPTH)])
def test_thick_components_need_a_depth_axis(kind, expected):
    # Thin components are seen on the plane of the sky, thick ones on a
    # depth axis
    assert evaluation(config(kind)) is expected


@pytest.mark.parametrize('kind', THIN_KINDS)
def test_monte_carlo_disks_need_a_depth_axis(kind):
    # A Monte Carlo disk is thick
    _, component = OBSERVATIONS[kind]
    mcdisk = component | dict(
        type='thick_disk', clouds=dict(cflux=1e-3), **THICK)
    assert evaluation(config(kind, [mcdisk])) is SceneEvaluation.DEPTH


@pytest.mark.parametrize('kind', OBSERVATIONS)
def test_dust_needs_a_depth_axis(kind):
    # Each galaxy is dimmed by its own dust on a depth axis, even when it
    # is seen by thin components only
    assert evaluation(config(kind, with_dust(kind))) \
        is SceneEvaluation.DEPTH


def test_line_opacity_needs_radiative_transfer():
    gas = LINES_THICK | dict(opacity=dict(
        type='line', coefficients=dict(type='uniform')))
    assert evaluation(config('cube_thick', [gas])) \
        is SceneEvaluation.TRANSFER


def two_galaxies(kind, depth_order):
    """
    The configuration of an observation of the given kind of two
    galaxies, a and b, of the component it sees, and b of dust too, in
    the given order along the line of sight, if any.
    """
    _, component = OBSERVATIONS[kind]
    info = config(kind)
    info['galaxies'] = [
        dict(name='a', components=[component]),
        dict(name='b', components=with_dust(kind))]
    info['observations'][0]['galaxies'] = ['a', 'b']
    if depth_order is not None:
        info['depth_order'] = depth_order
    return info


@pytest.mark.parametrize('kind', OBSERVATIONS)
def test_galaxies_that_dim_one_another_need_radiative_transfer(kind):
    # The dust of each dims the light of those behind it: radiative
    # transfer marches them front to back, in their order, of all of them
    assert evaluation(two_galaxies(kind, ['b', 'a'])) \
        is SceneEvaluation.TRANSFER
    message = load_error(two_galaxies(kind, None))
    assert "must place the galaxies ['a', 'b']" in message


IMAGE_SEES_NONE = (
    "the observation is an image, which sees the components that emit in "
    "a band; it sees none")
CUBE_SEES_NONE = (
    "the observation is a spectral cube, which sees the components that "
    "emit lines; it sees none")


@pytest.mark.parametrize('kind, component, message', [
    ('image_thin', LINES_THIN, IMAGE_SEES_NONE),
    ('image_thick', LINES_THICK, IMAGE_SEES_NONE),
    ('cube_thin', BAND_THIN, CUBE_SEES_NONE),
    ('cube_thick', BAND_THICK, CUBE_SEES_NONE),
    ('image_thick', DUST, IMAGE_SEES_NONE),
    ('cube_thick', DUST, CUBE_SEES_NONE)])
def test_observations_reject_components_of_other_kinds(
        kind, component, message):
    assert message in load_error(config(kind, [component]))


@pytest.mark.parametrize('kind', OBSERVATIONS)
def test_opacity_cannot_be_selected(kind):
    # The opacity of every component applies: an observation selects the
    # components whose light it sees
    info = config(kind, with_dust(kind))
    info['observations'][0]['components'] = ['dust']
    message = load_error(info)
    assert "the component 'dust' is selected, but the observation is" \
        in message


@pytest.mark.parametrize('kind', OBSERVATIONS)
def test_depth_options_are_taken(kind):
    # (those of an observation that needs no depth axis apply to nothing)
    config_group_plan(config(kind, depth=DEPTH | dict(slab=2)))
    config_group_plan(config(kind, with_dust(kind), DEPTH))


@pytest.mark.parametrize('options, message', [
    (dict(extent=-1), "'extent' must be 'galaxy', 'legacy' or of type float "
                      "greater than 0"),
    (dict(extent='half'), "'extent' must be 'galaxy', 'legacy' or"),
    (dict(step=0), "'step' must be greater than 0")])
def test_invalid_depths_are_rejected(options, message):
    error = load_error(config('image_thick', depth=options))
    assert error.startswith("observations[0].depth: ")
    assert message in error


@pytest.mark.parametrize('depth, axis', [
    # As many voxels as cover the extent, centred on 0
    (Depth(extent=21, step=1.5), (14, 1.5, -9.75)),
    (Depth(extent=20, step=1.5), (14, 1.5, -9.75)),
    # By default, the diagonal of the x and y axes (of 8 and 6 arcsec: 10),
    # or the diameter of the bounded components (radius 5: 10), if longer,
    # in voxels of the step of the longer of the axes (x, in arcsec; y has
    # more pixels)
    (Depth(), (10, 1, -4.5)),
    # Legacy: those of the longer of the x and y axes
    (Depth(extent='legacy'), (8, 1, -3.5)),
    (Depth(extent='legacy', step=0.25), (8, 0.25, -0.875))])
def test_depth_axis(depth, axis, caplog):
    grid = Grid(
        (8, 12), Coords((1, 0.5), (3.5, 5.5), (0, 0), 0), None)
    caplog.set_level(logging.WARNING)
    assert depth.axis(grid, 5) == axis
    # (legacy is shorter than the diagonal by choice: no warning)
    assert 'shorter than the galaxies need' not in caplog.text


@pytest.mark.parametrize('name', COMPONENTS)
def test_unknown_trait_options_are_errors(name):
    component = COMPONENTS[name]
    message = galaxy_error(dict(components=[
        component | dict(rptraits=dict(EXPONENTIAL, foo=1))]))
    assert message.startswith("components[0].rptraits")
    assert "'foo'" in message


@pytest.mark.parametrize('name', COMPONENTS)
def test_unknown_options_are_warnings_in_lenient_mode(name, caplog):
    info = dict(components=COMPONENTS[name] | dict(foo=1))
    with strict_mode(False), caplog.at_level(logging.WARNING):
        galaxy_parser.load(info)
    assert "'foo'" in caplog.text


def test_trait_names_prefix_their_parameters():
    # (after their kind, in place of their position, which an inserted
    # trait moves)
    inflow = dict(type='uniform', direction='rad', name='inflow')
    rotation = ARCTAN | dict(name='rot')
    info = dict(components=[
        LINES_THIN | dict(vptraits=[inflow, rotation], dptraits=UNIFORM)])
    galaxy = galaxy_parser.load(info)
    assert {'vpt_inflow_vr', 'vpt_rot_rt', 'vpt_rot_vt', 'dpt_a'} \
        <= set(galaxy.pdescs())
    dumped = galaxy_parser.dump(galaxy)
    vptraits = dumped['components'][0]['vptraits']
    assert [trait['name'] for trait in vptraits] == ['inflow', 'rot']


def test_unnamed_traits_are_numbered_from_the_second():
    # (a trait added at the end renames none of the others)
    info = dict(components=[LINES_THIN | dict(
        rptraits=[EXPONENTIAL, EXPONENTIAL, EXPONENTIAL],
        vptraits=[ARCTAN, dict(type='uniform', direction='rad')],
        dptraits=UNIFORM)])
    names = set(galaxy_parser.load(info).pdescs())
    assert {'rpt_a', 'rpt2_a', 'rpt3_a', 'vpt_vt', 'vpt2_vr', 'dpt_a'} \
        <= names
    assert not {'rpt1_a', 'vpt1_vr', 'pt_a'} & names


def test_traits_of_other_kinds_can_share_a_name():
    info = dict(components=[LINES_THIN | dict(
        rptraits=EXPONENTIAL | dict(name='ring'),
        vptraits=ARCTAN, dptraits=UNIFORM | dict(name='ring'))])
    assert {'rpt_ring_a', 'dpt_ring_a'} <= set(
        galaxy_parser.load(info).pdescs())


@pytest.mark.parametrize('vptraits, message', [
    ([ARCTAN | dict(name='rot'), ARCTAN],
     "either all or none of the traits must have a name"),
    ([ARCTAN | dict(name='rot'), ARCTAN | dict(name='rot')],
     r"cannot share a name; repeated: \['rot'\]")])
def test_trait_names_are_all_or_none_and_unique_in_a_list(vptraits, message):
    info = dict(components=[
        LINES_THIN | dict(vptraits=vptraits, dptraits=UNIFORM)])
    assert re.search(message, galaxy_error(info))


@pytest.mark.parametrize('rnmax, rnsep, nodes', [
    # (0.2 * 11 rounds above 2.2: np.arange to 2.4 added a node)
    (2.2, 0.2, 12), (9.0, 0.3, 31), (2, 0.25, 9),
    # Not a multiple: to the first node beyond max
    (10, 3, 5)])
def test_radial_nodes_from_a_range(rnmax, rnsep, nodes):
    args = parse_component_rnode_args(
        Nodes.every(rnmax, rnsep), None, 'linear')
    assert len(args['rnodes']) == nodes
    assert args['rnodes'][0] == 0
    assert args['rnodes'][-1] == pytest.approx((nodes - 1) * rnsep)


def test_radial_nodes_of_a_count():
    args = parse_component_rnode_args(Nodes.count(8, 5), None, 'linear')
    assert args['rnodes'] == (0, 2, 4, 6, 8)


def test_radial_nodes_start_at_the_centre():
    with pytest.raises(ValueError, match="must start at 0"):
        Nodes([1, 2, 3])


@pytest.mark.parametrize('rnodes, message', [
    (None, "rnodes must be given"),
    (dict(max=4), "it has none of them"),
    # (one form only: the other would be ignored silently)
    (dict(max=4, sep=1, len=3), "it has those of several"),
    (dict(max=4, sep=5), "sep must be greater than 0 and at most max"),
    (dict(max=4, len=1), "len must be at least 2")])
def test_invalid_radial_nodes(rnodes, message):
    disk = {k: v for k, v in LINES_THIN.items() if k != 'rnodes'}
    if rnodes is not None:
        disk['rnodes'] = rnodes
    assert message in galaxy_error(dict(components=[disk]))


@pytest.mark.parametrize('rnodes', [
    [0, 1, 2, 4], dict(max=4, sep=1), dict(max=4, len=3)])
def test_radial_nodes_keep_their_form(rnodes):
    info = dict(components=[LINES_THIN | dict(rnodes=rnodes)])
    dumped = galaxy_parser.dump(galaxy_parser.load(info))
    assert dumped['components'][0]['rnodes'] == rnodes


def test_radial_step_can_be_half_the_node_separation():
    # (half of 0.19999999999999996, the separation as it rounds)
    args = parse_component_rnode_args(
        Nodes([0, 0.2, 0.4, 0.6, 0.8, 1.0]), 0.1, 'linear')
    assert args['rstep'] == 0.1


def test_discouraged_traits_are_warnings_with_the_path(caplog):
    component = BAND_THIN | dict(rptraits=UNIFORM)
    with config_path('galaxies'):
        galaxy_parser.load([dict(components=component)])
    assert "galaxies[0].components [thin_disk]: the use of" in caplog.text


def test_the_azimuths_of_patterns_of_m_arms_repeat_every_360_over_m():
    # (an m = 2 harmonic declared 360: p and p + 180 are the same pattern,
    # audit 2026-10-31 #40)
    from gbkfit.model.components.traits.options import Harmonic, Spiral
    assert Harmonic(order=2).params(3, 'km/s')['p'].period() == 180
    assert Spiral(order=3).params_sm(False)['p'].period() == 120
