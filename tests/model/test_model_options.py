"""
Tests for the options of the models: which components and options each
model accepts, and which it rejects.
"""

import logging

import pytest
from gbkfit.model import model_parser
from gbkfit.utils import parseutils


EXPONENTIAL = dict(type='exponential')
SECH2 = dict(type='sech2')
ARCTAN = dict(type='tan_arctan')
UNIFORM = dict(type='uniform')

DISK = dict(loose=False, tilted=False, rnodes=[0, 2, 4, 6, 8])

# A component of each kind, with its required options
BRIGHTNESS_2D = dict(DISK, type='smdisk', bptraits=EXPONENTIAL)
BRIGHTNESS_3D = BRIGHTNESS_2D | dict(bhtraits=SECH2)
SPECTRAL_2D = BRIGHTNESS_2D | dict(vptraits=ARCTAN, dptraits=UNIFORM)
SPECTRAL_3D = SPECTRAL_2D | dict(bhtraits=SECH2)
OPACITY_3D = dict(DISK, type='smdisk', optraits=EXPONENTIAL, ohtraits=SECH2)

# Each type of model, with a component of the kind it accepts
MODELS = dict(
    intensity_2d=BRIGHTNESS_2D,
    intensity_3d=BRIGHTNESS_3D,
    kinematics_2d=SPECTRAL_2D,
    kinematics_3d=SPECTRAL_3D)

MODELS_2D = ['intensity_2d', 'kinematics_2d']
MODELS_3D = ['intensity_3d', 'kinematics_3d']

# The options of the 3d models only
OPTIONS_3D = dict(
    opacity_components=[OPACITY_3D], size_z=10, step_z=1, zero_z=-4.5)


def strict_load(info):
    """Load a model, with unknown options as errors."""
    with parseutils.strict_mode():
        return model_parser.load(info)


def load_error(info):
    """The error message of loading a bad model."""
    with pytest.raises(parseutils.ConfigError) as error:
        strict_load(info)
    return str(error.value)


@pytest.mark.parametrize('model_type', MODELS)
def test_model_loads(model_type):
    strict_load(dict(type=model_type, components=MODELS[model_type]))


@pytest.mark.parametrize('model_type', MODELS)
def test_components_cannot_be_missing(model_type):
    message = load_error(dict(type=model_type))
    assert "option 'components' is required but not provided" in message


@pytest.mark.parametrize('model_type', MODELS)
def test_components_cannot_be_null(model_type):
    message = load_error(dict(type=model_type, components=None))
    assert "option 'components' cannot be null" in message


@pytest.mark.parametrize('model_type', MODELS)
def test_components_cannot_be_empty(model_type):
    message = load_error(dict(type=model_type, components=[]))
    assert "at least one component must be configured" in message


@pytest.mark.parametrize('model_type', MODELS_2D)
def test_2d_models_have_no_mcdisk_components(model_type):
    component = MODELS[model_type] | dict(type='mcdisk', cflux=1e-3)
    message = load_error(dict(type=model_type, components=component))
    assert "unknown" in message and "type 'mcdisk'" in message


@pytest.mark.parametrize('model_type, component, message', [
    ('intensity_2d', SPECTRAL_2D, "unknown options"),
    ('intensity_3d', SPECTRAL_3D, "unknown options"),
    ('kinematics_2d', BRIGHTNESS_2D, "option 'vptraits' is required"),
    ('kinematics_3d', BRIGHTNESS_3D, "option 'vptraits' is required"),
    ('intensity_3d', OPACITY_3D, "option 'bptraits' is required"),
    ('kinematics_3d', OPACITY_3D, "option 'bptraits' is required"),
    ('kinematics_2d', SPECTRAL_3D, "unknown options"),
    ('intensity_2d', BRIGHTNESS_3D, "unknown options")])
def test_models_reject_components_of_other_kinds(
        model_type, component, message):
    assert message in load_error(dict(type=model_type, components=component))


@pytest.mark.parametrize('model_type', MODELS_3D)
def test_opacity_components_are_not_components(model_type):
    info = dict(
        type=model_type, components=MODELS[model_type],
        opacity_components=BRIGHTNESS_3D)
    assert "option 'optraits' is required" in load_error(info)


@pytest.mark.parametrize('model_type', MODELS_3D)
def test_3d_models_have_3d_options(model_type):
    strict_load(
        dict(type=model_type, components=MODELS[model_type]) | OPTIONS_3D)


@pytest.mark.parametrize('model_type', MODELS_2D)
@pytest.mark.parametrize('option', OPTIONS_3D)
def test_2d_models_have_no_3d_options(model_type, option):
    info = dict(
        type=model_type, components=MODELS[model_type],
        **{option: OPTIONS_3D[option]})
    message = load_error(info)
    assert message.startswith(f"{model_type}: unknown options")
    assert f"'{option}'" in message


@pytest.mark.parametrize('model_type', MODELS)
def test_unknown_trait_options_are_errors_in_strict_mode(model_type):
    component = MODELS[model_type] | dict(
        bptraits=dict(EXPONENTIAL, foo=1))
    message = load_error(dict(type=model_type, components=[component]))
    assert message.startswith("components[0].bptraits")
    assert "'foo'" in message


@pytest.mark.parametrize('model_type', MODELS)
def test_unknown_options_are_warnings(model_type, caplog):
    component = MODELS[model_type] | dict(foo=1)
    with caplog.at_level(logging.WARNING):
        model_parser.load(dict(type=model_type, components=component))
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
