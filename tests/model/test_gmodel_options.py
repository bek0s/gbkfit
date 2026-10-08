"""
Tests for the options of the gmodels: which components and options each
gmodel accepts, and which it rejects.
"""

import logging

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
    assert "unknown options for gmodel" in message
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


@pytest.mark.parametrize('gmodel_type', GMODELS)
@pytest.mark.parametrize('nwmode, switch', [
    ('xpos_nwmode', 'loose'), ('ypos_nwmode', 'loose'),
    ('posa_nwmode', 'tilted'), ('incl_nwmode', 'tilted')])
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
