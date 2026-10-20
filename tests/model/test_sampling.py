"""
Tests for the sampling of the node-wise parameters of traits: at the
radial nodes of their disk (the default), which interpolates them to the
subnodes, or at the subnodes themselves ('subrings'), so that an
expression of the subnodes ('subrnodes') is not limited by the nodes.
"""

import numpy as np
import pytest

from gbkfit.math import interpolation
from gbkfit.model import Extras, galaxy_parser
from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.params.space import ParamSpace
from modelutils import observation_group


PROPERTIES = dict(
    vsys=0, xpos=0, ypos=0, posa=30, incl=45, pt_a=1, pt_s=4, dpt_a=10)


def disk(type_, rnodes, vptraits, thick=False, **options):
    """
    A disk of the given type that emits lines, with the given velocity
    polar traits: thick (of sech2 height traits) if thick, as a Monte
    Carlo disk always is.
    """
    density = dict(ptraits=dict(type='exponential'))
    if thick or type_ == 'mcdisk':
        density['htraits'] = dict(type='sech2', zmax=8)
    mcdisk_options = dict(cflux=1e-3) if type_ == 'mcdisk' else {}
    return dict(
        type=type_, loose=False, tilted=False, rnodes=rnodes,
        density=density,
        kinematics=dict(vptraits=vptraits, dptraits=dict(type='uniform')),
        emission=dict(type='lines'), **mcdisk_options, **options)


def case(component):
    """
    A spectral cube of a galaxy of the component: on a grid if it is
    thick, or else on the plane of the sky.
    """
    thick = 'htraits' in component['density']
    return dict(
        driver=dict(type='host'),
        observable=dict(
            type='pixel_spectra', size=[32, 32, 41], step=[1, 1, 10]),
        line_of_sight=dict(type='grid' if thick else 'flat'),
        galaxy=dict(components=[component]))


def group(component):
    return observation_group([case(component)])


@pytest.mark.parametrize('type_', ['smdisk', 'mcdisk'])
def test_values_at_the_subnodes_are_used_as_they_are(evaluate_cases, type_):
    # A curve given at the nodes gives the same model as its values
    # interpolated to the subnodes, given at the subnodes
    rnodes = [0, 4, 10, 20]
    curve = [0, 150, 180, 170]
    at_rnodes = disk(type_, rnodes, dict(type='nw_uniform', direction='tan'))
    at_subnodes = disk(
        type_, rnodes, dict(type='nw_uniform', direction='tan', sampling='subrings'))
    subrnodes = group(at_subnodes).constants()['subrnodes']
    interpolated = interpolation.InterpolatorLinear(rnodes, curve)(subrnodes)
    expected, _ = evaluate_cases(
        [case(at_rnodes)], PROPERTIES | dict(ht_s=1, vpt_vt=curve))
    actual, _ = evaluate_cases(
        [case(at_subnodes)],
        PROPERTIES | dict(ht_s=1, vpt_vt=interpolated.tolist()))
    # The host adds the clouds of a thick disk in any order (float32)
    rtol = 1e-4 if type_ == 'mcdisk' else 0
    np.testing.assert_allclose(
        actual[0]['spectra']['d'], expected[0]['spectra']['d'], rtol=rtol)


def test_node_wise_parameters_have_a_value_for_each_subnode():
    component = disk(
        'smdisk', [0, 20], dict(type='nw_harmonic', direction='tan', order=1,
                                sampling='subrings'),
        rstep=0.5)
    obs_group = group(component)
    nsubrnodes = len(obs_group.constants()['subrnodes'])
    # The edges of the disk and the centres of 40 rings
    assert nsubrnodes == 42
    assert obs_group.pdescs()['vpt_a'].size() == nsubrnodes
    assert obs_group.pdescs()['vpt_p'].size() == nsubrnodes


def test_traits_evaluated_apart_are_sampled_as_by_their_disk(driver):
    # Node-wise values given at the nodes are interpolated to the subnodes
    # with the interpolation of the disk, and linearly between them, up to
    # its last node (the disk is closed); off the disk, the traits are NaN
    from gbkfit.model import BandEmission, Density, SMDisk
    from gbkfit.model.components import traits
    rnodes, curve = (0.0, 4.0, 10.0, 20.0), [1.0, 3.0, 2.0, 0.5]
    component = SMDisk(
        Density(traits.RPTraitNWUniform()), emission=BandEmission(),
        loose=False, tilted=False, rnodes=rnodes, rstep=1.0,
        interp='akima')
    density_disk = component.density_disk()
    plan = density_disk.plan(driver, 0, np.dtype(np.float32))
    params = dict(xpos=0, ypos=0, posa=0, incl=45, pt_a=np.array(curve))
    subrnodes = np.array(density_disk.subrnodes())
    at = interpolation.InterpolatorAkima(rnodes, curve)(subrnodes)
    r = np.concatenate(
        (subrnodes, (subrnodes[1:] + subrnodes[:-1]) / 2, [20.5]))
    expected = np.concatenate((at, (at[1:] + at[:-1]) / 2, [np.nan]))
    # (r and theta broadcast together)
    values = plan.evaluate_polar_traits(params, 'rpt', r[:, None], [0, 90])
    assert values.shape == (1, r.size, 2)
    np.testing.assert_allclose(
        values[0], np.column_stack((expected, expected)), rtol=1e-6)


def test_flaring_at_the_subnodes():
    galaxy = galaxy_parser.load(dict(components=[dict(
        type='smdisk', loose=False, tilted=False, rnodes=[0, 20],
        rstep=0.5,
        density=dict(
            ptraits=dict(type='exponential'),
            htraits=dict(
                type='sech2', zmax=8, rnodes=True, sampling='subrings')),
        emission=dict(type='band'))]))
    assert galaxy.pdescs()['ht_s'].size() == 42


def evaluate_curve(component, free):
    # Evaluate the velocity field of the component with vmax and rt free,
    # as a fitter does
    obs_group = group(component)
    pdescs = obs_group.pdescs() | dict(
        vmax=ParamScalarDesc(), rt=ParamScalarDesc())
    space = ParamSpace(pdescs, PROPERTIES | dict(
        vmax=dict(val=100), rt=dict(val=1),
        vpt_vt='vmax * 2 / np.pi * np.arctan(subrnodes / rt)'),
        constants=obs_group.constants())
    extra = Extras()
    obs_group.model_h(space.evaluate(free), extra)
    return extra.get('vdata', component='component0').data


@pytest.mark.parametrize('vmax, rt', [(150, 2), (220, 4)])
def test_an_expression_of_the_subnodes_follows_the_curve(
        evaluate_cases, vmax, rt):
    # With two nodes, a curve given at the subnodes is the analytical one,
    # up to its linear interpolation between the subnodes
    rnodes = [0, 20]
    analytical = disk(
        'smdisk', rnodes, dict(type='arctan', direction='tan'), rstep=0.1)
    at_subnodes = disk(
        'smdisk', rnodes, dict(type='nw_uniform', direction='tan', sampling='subrings'),
        rstep=0.1)
    _, extra = evaluate_cases(
        [case(analytical)], PROPERTIES | dict(vpt_vt=vmax, vpt_rt=rt))
    expected = extra.get('vdata', component='component0').data
    actual = evaluate_curve(at_subnodes, dict(vmax=vmax, rt=rt))
    np.testing.assert_allclose(actual, expected, atol=0.1, equal_nan=True)


def test_sampling_is_dumped():
    galaxy = galaxy_parser.load(dict(components=[disk(
        'smdisk', [0, 20],
        dict(type='nw_uniform', direction='tan', sampling='subrings'))]))
    info = galaxy_parser.dump(galaxy)
    assert info['components'][0]['kinematics']['vptraits'] == [dict(
        type='nw_uniform', direction='tan', sampling='subrings')]
    galaxy = galaxy_parser.load(info)
    assert galaxy_parser.dump(galaxy) == info
    # The default is not dumped
    galaxy = galaxy_parser.load(dict(components=[disk(
        'smdisk', [0, 20], dict(type='nw_uniform', direction='tan'))]))
    info = galaxy_parser.dump(galaxy)
    assert info['components'][0]['kinematics']['vptraits'] == [
        dict(type='nw_uniform', direction='tan')]


@pytest.mark.parametrize('part, traits, message', [
    ('kinematics',
     dict(vptraits=dict(type='nw_uniform', direction='tan', sampling='nodes')),
     r"sampling must be one of \['rnodes', 'subrings'\]; it is 'nodes'"),
    ('density',
     dict(htraits=dict(type='sech2', zmax=8, sampling='subrings')),
     "sampling is 'subrings', but rnodes is False: the trait has no "
     "node-wise parameters")])
def test_invalid_sampling_is_rejected(part, traits, message):
    component = disk('smdisk', [0, 20], dict(type='arctan', direction='tan'), thick=True)
    component[part] |= traits
    with pytest.raises(Exception, match=message):
        galaxy_parser.load(dict(components=[component]))
