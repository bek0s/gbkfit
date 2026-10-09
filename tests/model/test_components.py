"""
Tests for the options of the gmodel components, and their errors.
"""

import re

import numpy as np
import pytest
from gbkfit.model import gmodel_parser, gmodels
from gbkfit.utils import gridutils


EXPONENTIAL = dict(type='exponential')
SECH2 = dict(type='sech2')
ARCTAN = dict(type='tan_arctan')
UNIFORM = dict(type='uniform')

DISK = dict(loose=False, tilted=False, rnodes=[0, 2, 4, 6, 8])

# Each type of component: its class, the gmodel and the option of the
# gmodel it is in, and the component with its required options
COMPONENTS = dict(
    brightness_smdisk_2d=(
        gmodels.BrightnessSMDisk2D, 'intensity_2d', 'components',
        dict(type='smdisk', bptraits=EXPONENTIAL)),
    brightness_smdisk_3d=(
        gmodels.BrightnessSMDisk3D, 'intensity_3d', 'components',
        dict(type='smdisk', bptraits=EXPONENTIAL, bhtraits=SECH2)),
    brightness_mcdisk_3d=(
        gmodels.BrightnessMCDisk3D, 'intensity_3d', 'components',
        dict(type='mcdisk', cflux=1e-3, bptraits=EXPONENTIAL, bhtraits=SECH2)),
    spectral_smdisk_2d=(
        gmodels.SpectralSMDisk2D, 'kinematics_2d', 'components',
        dict(type='smdisk', bptraits=EXPONENTIAL, vptraits=ARCTAN,
             dptraits=UNIFORM)),
    spectral_smdisk_3d=(
        gmodels.SpectralSMDisk3D, 'kinematics_3d', 'components',
        dict(type='smdisk', bptraits=EXPONENTIAL, bhtraits=SECH2,
             vptraits=ARCTAN, dptraits=UNIFORM)),
    spectral_mcdisk_3d=(
        gmodels.SpectralMCDisk3D, 'kinematics_3d', 'components',
        dict(type='mcdisk', cflux=1e-3, bptraits=EXPONENTIAL, bhtraits=SECH2,
             vptraits=ARCTAN, dptraits=UNIFORM)),
    opacity_smdisk_3d=(
        gmodels.OpacitySMDisk3D, 'intensity_3d', 'opacity_components',
        dict(type='smdisk', optraits=EXPONENTIAL, ohtraits=SECH2)),
    opacity_mcdisk_3d=(
        gmodels.OpacityMCDisk3D, 'intensity_3d', 'opacity_components',
        dict(type='mcdisk', cflux=1e-3, optraits=EXPONENTIAL, ohtraits=SECH2)))

# Each required trait option of each type of component
REQUIRED_TRAITS = [
    (name, key)
    for name, (_, _, _, component) in COMPONENTS.items()
    for key in component if key.endswith('traits')]

# Each height trait option that must have as many traits as a polar one
HEIGHT_TRAITS = [
    (name, key) for name, key in REQUIRED_TRAITS if key[1] == 'h']


def gmodel_info(name, **options):
    """
    The configuration of a gmodel with a component of the given type, with
    its options updated. An option set to ... is left out.
    """
    _, gmodel_type, key, component = COMPONENTS[name]
    component = dict(DISK, **component) | options
    component = {k: v for k, v in component.items() if v is not ...}
    info = dict(type=gmodel_type, components=[
        dict(DISK, type='smdisk', bptraits=EXPONENTIAL, bhtraits=SECH2)])
    info[key] = [component]
    return info


def load_error(name, **options):
    """The error message of loading a gmodel with a bad component."""
    with pytest.raises(Exception) as error:
        gmodel_parser.load(gmodel_info(name, **options))
    message = str(error.value)
    # The message names the type of component
    cls = COMPONENTS[name][0]
    assert f"(class={cls.__qualname__})" in message
    return message


@pytest.mark.parametrize('name', COMPONENTS)
def test_component_loads(name):
    gmodel_parser.load(gmodel_info(name))


@pytest.mark.parametrize('name, key', REQUIRED_TRAITS)
def test_required_traits_cannot_be_missing(name, key):
    message = load_error(name, **{key: ...})
    assert f"option '{key}' is required but not provided" in message


@pytest.mark.parametrize('name, key', REQUIRED_TRAITS)
def test_required_traits_cannot_be_null(name, key):
    message = load_error(name, **{key: None})
    assert f"option '{key}' cannot be null" in message


@pytest.mark.parametrize('name, key', REQUIRED_TRAITS)
def test_required_traits_cannot_be_empty(name, key):
    message = load_error(name, **{key: []})
    assert f"at least one {key[:-1]} is required" in message


@pytest.mark.parametrize('name, key', HEIGHT_TRAITS)
def test_height_traits_must_match_polar_traits(name, key):
    polar_key = key.replace('h', 'p', 1)
    message = load_error(name, **{key: [SECH2, SECH2]})
    assert (
        f"the number of {key} must be equal to "
        f"the number of {polar_key} (2 != 1)") in message


@pytest.mark.parametrize('name', COMPONENTS)
def test_optional_traits_can_be_null(name):
    gmodel_parser.load(gmodel_info(name, sptraits=None, wptraits=None))


@pytest.mark.parametrize('name', COMPONENTS)
def test_weight_traits_are_parsed(name):
    # Weight traits are not implemented yet, but every component must
    # get as far as checking them
    weights = dict(type='axis_range', axis=0, angle=10, weight=2)
    message = load_error(name, wptraits=weights)
    assert re.search("weight polar trait .* is not implemented yet", message)


@pytest.mark.parametrize('name', ['spectral_smdisk_3d', 'spectral_mcdisk_3d'])
def test_spectral_3d_with_several_velocity_traits(name):
    # Rotation and radial motions, each with the default height trait.
    # The disk needs a height trait for each velocity trait to evaluate.
    from gbkfit.driver.drivers.host import DriverHost
    gmodel = gmodel_parser.load(gmodel_info(name, vptraits=[
        ARCTAN, dict(type='nw_rad_uniform')]))
    assert {'vpt_vt', 'vpt1_vr'} <= set(gmodel.pdescs())
    params = {
        name: np.ones(pdesc.size()) if pdesc.type() == 'vector' else 1.0
        for name, pdesc in gmodel.pdescs().items()}
    scube = np.zeros((11, 16, 16), np.float32)
    grid = gridutils.Grid((16, 16, 11), gridutils.Coords(
        (1, 1, 10), (7.5, 7.5, 5), (0, 0, 0), 0), 2)
    gmodel.plan(DriverHost(), grid, False, np.float32).evaluate(
        params, scube, None, None)
    assert scube.sum() > 0


def test_errors_have_paths():
    info = gmodel_info('spectral_smdisk_3d', vptraits=[
        ARCTAN, dict(type='nw_rad_uniform', nwmode=dict(type='relative9'))])
    with pytest.raises(Exception) as error:
        gmodel_parser.load(info)
    assert str(error.value).startswith(
        "components[0].vptraits[1].nwmode: unknown NWMode type 'relative9'")


def component_error(name, **options):
    """The error message of loading a gmodel with a bad component."""
    with pytest.raises(Exception) as error:
        gmodel_parser.load(gmodel_info(name, **options))
    return str(error.value)


def test_at_most_4_traits_of_a_kind():
    # The native kernels take at most 4 traits of each kind
    message = component_error(
        'brightness_smdisk_2d', bptraits=[EXPONENTIAL] * 5)
    assert "at most 4 bptraits" in message


@pytest.mark.parametrize('trait, message', [
    (dict(type='nw_harmonic', order=-1), "order must be at least 0"),
    (dict(type='mixture_gauss', nblobs=0), "nblobs must be at least 1")])
def test_orders_and_blobs_are_checked(trait, message):
    # The kernels read the phases of the orders above 0, and the
    # parameters of each blob
    assert message in component_error('brightness_smdisk_2d', bptraits=trait)

def test_trunc_cannot_be_negative():
    message = component_error(
        'brightness_smdisk_3d', bhtraits=dict(type='sech2', trunc=-1))
    assert "trunc must be at least 0" in message
