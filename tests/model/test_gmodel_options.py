"""
Tests for the options of the gmodels: which components and options each
gmodel accepts, and which it rejects.
"""

import logging

import numpy as np
import pytest
from gbkfit.model import gmodel_parser
from gbkfit.utils import parseutils


EXPONENTIAL = dict(type='exponential')
SECH2 = dict(type='sech2')
ARCTAN = dict(type='tan_arctan')
UNIFORM = dict(type='uniform')
RELATIVE = dict(type='relative1', origin=0)

DISK = dict(loose=False, tilted=False, rnodes=[0, 2, 4, 6, 8])

# A component of each kind, with its required options
BRIGHTNESS_2D = dict(DISK, type='smdisk', bptraits=EXPONENTIAL)
BRIGHTNESS_3D = BRIGHTNESS_2D | dict(bhtraits=SECH2)
SPECTRAL_2D = BRIGHTNESS_2D | dict(vptraits=ARCTAN, dptraits=UNIFORM)
SPECTRAL_3D = SPECTRAL_2D | dict(bhtraits=SECH2)
OPACITY_3D = dict(DISK, type='smdisk', optraits=EXPONENTIAL, ohtraits=SECH2)

# Each type of gmodel, with a component of the kind it accepts
GMODELS = dict(
    intensity_2d=BRIGHTNESS_2D,
    intensity_3d=BRIGHTNESS_3D,
    kinematics_2d=SPECTRAL_2D,
    kinematics_3d=SPECTRAL_3D)

GMODELS_2D = ['intensity_2d', 'kinematics_2d']
GMODELS_3D = ['intensity_3d', 'kinematics_3d']

# The options of the 3d gmodels only
OPTIONS_3D = dict(
    opacity_components=[OPACITY_3D], size_z=10, step_z=1, zero_z=-4.5)


def strict_load(info):
    """Load a gmodel, with unknown options as errors."""
    with parseutils.strict_mode():
        return gmodel_parser.load(info)


def load_error(info):
    """The error message of loading a bad gmodel."""
    with pytest.raises(parseutils.ConfigError) as error:
        strict_load(info)
    return str(error.value)


@pytest.mark.parametrize('gmodel_type', GMODELS)
def test_gmodel_loads(gmodel_type):
    strict_load(dict(type=gmodel_type, components=GMODELS[gmodel_type]))


@pytest.mark.parametrize('gmodel_type', GMODELS)
def test_components_cannot_be_missing(gmodel_type):
    message = load_error(dict(type=gmodel_type))
    assert "option 'components' is required but not provided" in message


@pytest.mark.parametrize('gmodel_type', GMODELS)
def test_components_cannot_be_null(gmodel_type):
    message = load_error(dict(type=gmodel_type, components=None))
    assert "option 'components' cannot be null" in message


@pytest.mark.parametrize('gmodel_type', GMODELS)
def test_components_cannot_be_empty(gmodel_type):
    message = load_error(dict(type=gmodel_type, components=[]))
    assert "at least one component must be configured" in message


@pytest.mark.parametrize('gmodel_type', GMODELS_2D)
def test_2d_gmodels_have_no_mcdisk_components(gmodel_type):
    component = GMODELS[gmodel_type] | dict(type='mcdisk', cflux=1e-3)
    message = load_error(dict(type=gmodel_type, components=component))
    assert "unknown" in message and "type 'mcdisk'" in message


@pytest.mark.parametrize('gmodel_type, component, message', [
    ('intensity_2d', SPECTRAL_2D, "unknown options"),
    ('intensity_3d', SPECTRAL_3D, "unknown options"),
    ('kinematics_2d', BRIGHTNESS_2D, "option 'vptraits' is required"),
    ('kinematics_3d', BRIGHTNESS_3D, "option 'vptraits' is required"),
    ('intensity_3d', OPACITY_3D, "option 'bptraits' is required"),
    ('kinematics_3d', OPACITY_3D, "option 'bptraits' is required"),
    ('kinematics_2d', SPECTRAL_3D, "unknown options"),
    ('intensity_2d', BRIGHTNESS_3D, "unknown options")])
def test_gmodels_reject_components_of_other_kinds(
        gmodel_type, component, message):
    assert message in load_error(dict(type=gmodel_type, components=component))


@pytest.mark.parametrize('gmodel_type', GMODELS_3D)
def test_opacity_components_are_not_components(gmodel_type):
    info = dict(
        type=gmodel_type, components=GMODELS[gmodel_type],
        opacity_components=BRIGHTNESS_3D)
    assert "option 'optraits' is required" in load_error(info)


@pytest.mark.parametrize('gmodel_type', GMODELS_3D)
def test_3d_gmodels_have_3d_options(gmodel_type):
    strict_load(
        dict(type=gmodel_type, components=GMODELS[gmodel_type]) | OPTIONS_3D)


@pytest.mark.parametrize('gmodel_type', GMODELS_2D)
@pytest.mark.parametrize('option', OPTIONS_3D)
def test_2d_gmodels_have_no_3d_options(gmodel_type, option):
    info = dict(
        type=gmodel_type, components=GMODELS[gmodel_type],
        **{option: OPTIONS_3D[option]})
    message = load_error(info)
    assert message.startswith(f"{gmodel_type}: unknown options")
    assert f"'{option}'" in message


@pytest.mark.parametrize('gmodel_type', GMODELS)
def test_spatial_nwmodes(gmodel_type):
    component = GMODELS[gmodel_type] | dict(
        loose=True, tilted=True,
        xpos_nwmode=RELATIVE, ypos_nwmode=RELATIVE,
        posa_nwmode=RELATIVE, incl_nwmode=RELATIVE)
    strict_load(dict(type=gmodel_type, components=component))


@pytest.mark.parametrize('gmodel_type', ['kinematics_2d', 'kinematics_3d'])
def test_spectral_components_have_a_vsys_nwmode(gmodel_type):
    component = GMODELS[gmodel_type] | dict(
        loose=True, vsys_nwmode=RELATIVE)
    strict_load(dict(type=gmodel_type, components=component))


@pytest.mark.parametrize('gmodel_type', ['intensity_2d', 'intensity_3d'])
def test_brightness_components_have_no_vsys_nwmode(gmodel_type):
    component = GMODELS[gmodel_type] | dict(
        loose=True, vsys_nwmode=RELATIVE)
    message = load_error(dict(type=gmodel_type, components=component))
    assert "'vsys_nwmode'" in message


@pytest.mark.parametrize('gmodel_type', GMODELS_3D)
def test_opacity_components_have_no_vsys_nwmode(gmodel_type):
    ocomponent = OPACITY_3D | dict(loose=True, vsys_nwmode=RELATIVE)
    info = dict(
        type=gmodel_type, components=GMODELS[gmodel_type],
        opacity_components=ocomponent)
    assert "'vsys_nwmode'" in load_error(info)


# Each node-wise mode of each type of gmodel, and the option it needs:
# the spectral components also have a vsys node-wise mode
NWMODE_SWITCHES = [
    (gmodel_type, nwmode, switch)
    for gmodel_type in GMODELS
    for nwmode, switch in [
        ('xpos_nwmode', 'loose'), ('ypos_nwmode', 'loose'),
        ('posa_nwmode', 'tilted'), ('incl_nwmode', 'tilted')]
] + [
    (gmodel_type, 'vsys_nwmode', 'loose')
    for gmodel_type in ['kinematics_2d', 'kinematics_3d']]


@pytest.mark.parametrize('gmodel_type, nwmode, switch', NWMODE_SWITCHES)
def test_nwmodes_are_ignored_without_their_switch(
        gmodel_type, nwmode, switch, caplog):
    component = GMODELS[gmodel_type] | {nwmode: RELATIVE}
    gmodel = strict_load(dict(type=gmodel_type, components=component))
    assert (
        f"{nwmode} is set to 'relative1', but it will be ignored "
        f"because {switch} is not set to True") in caplog.text
    # Without the switch the parameter is not node-wise
    assert gmodel.pdescs()[nwmode[:4]].type() == 'scalar'


@pytest.mark.parametrize('gmodel_type', GMODELS)
def test_unknown_trait_options_are_errors_in_strict_mode(gmodel_type):
    component = GMODELS[gmodel_type] | dict(
        bptraits=dict(EXPONENTIAL, foo=1))
    message = load_error(dict(type=gmodel_type, components=[component]))
    assert message.startswith("components[0].bptraits")
    assert "'foo'" in message


@pytest.mark.parametrize('gmodel_type', GMODELS)
def test_unknown_options_are_warnings(gmodel_type, caplog):
    component = GMODELS[gmodel_type] | dict(foo=1)
    with caplog.at_level(logging.WARNING):
        gmodel_parser.load(dict(type=gmodel_type, components=component))
    assert "'foo'" in caplog.text


@pytest.mark.parametrize('rnmin, rnmax, rnsep, nodes', [
    # (0.2 * 11 rounds above 2.2: np.arange to 2.4 added a node)
    (0, 2.2, 0.2, 12), (0, 9.0, 0.3, 31), (1, 2, 0.25, 5),
    # Not a multiple: to the first node beyond rnmax
    (0, 10, 3, 5)])
def test_radial_nodes_from_a_range(rnmin, rnmax, rnsep, nodes):
    from gbkfit.model.components.disks._detail import parse_component_rnode_args
    args = parse_component_rnode_args(
        rnmin, rnmax, rnsep, None, None, None, 'linear')
    assert len(args['rnodes']) == nodes
    assert args['rnodes'][-1] == pytest.approx(rnmin + (nodes - 1) * rnsep)


def test_radial_step_can_be_half_the_node_separation():
    # (half of 0.19999999999999996, the separation as it rounds)
    from gbkfit.model.components.disks._detail import parse_component_rnode_args
    args = parse_component_rnode_args(
        None, None, None, None, [0, 0.2, 0.4, 0.6, 0.8, 1.0], 0.1, 'linear')
    assert args['rstep'] == 0.1


@pytest.mark.parametrize('nwmode, values, expected', [
    # Each node relative to the origin
    (dict(type='relative1', origin=1), [1, 2, 3, 4], [3, 2, 5, 6]),
    # Each node relative to its neighbour towards the origin
    (dict(type='relative2', origin=1), [1, 2, 3, 4], [3, 2, 5, 9])])
def test_relative_nwmodes(nwmode, values, expected):
    from gbkfit.model.components.disks.nwmodes import nwmode_parser
    result = nwmode_parser.load(nwmode).transform(
        np.array(values, float), in_place=False)
    np.testing.assert_array_equal(result, expected)
