"""
Tests for the names of components: a name, given to every component of a
list or to none, prefixes the parameters of the component instead of its
position.
"""

import pytest

from gbkfit.model import gmodel_parser


def disk(**options):
    return dict(
        type='smdisk', loose=False, tilted=False, rnodes=[0, 1, 2],
        bptraits=dict(type='exponential'), bhtraits=dict(type='sech2'),
        **options)


def opacity(**options):
    return dict(
        type='smdisk', loose=False, tilted=False, rnodes=[0, 1, 2],
        optraits=dict(type='exponential'), ohtraits=dict(type='sech2'),
        **options)


def intensity_3d(components, opacity_components=None):
    return gmodel_parser.load(dict(
        type='intensity_3d', components=components,
        opacity_components=opacity_components))


def test_without_names_the_prefixes_are_positions():
    gmodel = intensity_3d([disk(), disk()], [opacity()])
    assert {'xpos', 'cmp1_xpos', 'ocmp_xpos'} <= set(gmodel.pdescs())
    assert set(gmodel.constants()) == {
        'rnodes', 'subrnodes', 'cmp1_rnodes', 'cmp1_subrnodes',
        'ocmp_rnodes', 'ocmp_subrnodes'}


def test_names_prefix_the_parameters_and_constants():
    gmodel = intensity_3d(
        [disk(name='disk'), disk(name='bulge')], [opacity(name='dust')])
    names = set(gmodel.pdescs())
    assert {'disk_xpos', 'bulge_bpt_s', 'dust_opt_a'} <= names
    assert not {'xpos', 'cmp1_xpos', 'ocmp_xpos'} & names
    assert set(gmodel.constants()) == {
        'disk_rnodes', 'disk_subrnodes', 'bulge_rnodes', 'bulge_subrnodes',
        'dust_rnodes', 'dust_subrnodes'}


def test_names_do_not_depend_on_the_order():
    first = intensity_3d([disk(name='disk'), disk(name='bulge')])
    second = intensity_3d([disk(name='bulge'), disk(name='disk')])
    assert set(first.pdescs()) == set(second.pdescs())


def test_names_are_dumped():
    info = gmodel_parser.dump(intensity_3d([disk(name='disk')]))
    assert info['components'][0]['name'] == 'disk'
    info = gmodel_parser.dump(intensity_3d([disk()]))
    assert 'name' not in info['components'][0]


@pytest.mark.parametrize('components, opacity_components, message', [
    ([disk(name='disk'), disk()], None,
     "either all or none of the components must have a name"),
    ([disk(name='disk'), disk(name='disk')], None,
     r"the names of the components must be unique; repeated: \['disk'\]"),
    ([disk(name='disk')], [opacity(name='disk')],
     "the components and the opacity components must have different "
     "names"),
    ([disk(name='my disk')], None, "invalid name 'my disk'")])
def test_invalid_names_are_rejected(components, opacity_components, message):
    with pytest.raises(Exception, match=message):
        intensity_3d(components, opacity_components)


def gmodel(name=None):
    component = disk() | dict(bhtraits=None)
    return dict(type='intensity_2d', components=[component]) \
        | (dict(name=name) if name is not None else {})


def observation(gmodel_=None, name=None):
    return dict(
        driver=dict(type='host'), observable=dict(type='pixel_brightness', size=[8, 8])) \
        | (dict(gmodel=gmodel_) if gmodel_ is not None else {}) \
        | (dict(name=name) if name is not None else {})


def group(gmodels, observations):
    from gbkfit.observation import ObservationGroup, observation_parser
    return ObservationGroup(
        gmodel_parser.load(gmodels), observation_parser.load(observations))


def test_gmodel_names_prefix_the_parameters():
    group_ = group(
        [gmodel('hi'), gmodel('halpha')],
        [observation('hi'), observation('halpha')])
    assert {'hi_xpos', 'halpha_xpos'} <= set(group_.pdescs())
    assert set(group_.constants()) == {
        'hi_rnodes', 'hi_subrnodes', 'halpha_rnodes', 'halpha_subrnodes'}
    assert gmodel_parser.dump(group_.gmodels()[0])['name'] == 'hi'


def test_observations_of_one_gmodel_share_its_parameters():
    group_ = group([gmodel()], [observation(), observation()])
    assert 'xpos' in group_.pdescs()
    assert not any(name.startswith('gmodel') for name in group_.pdescs())


@pytest.mark.parametrize('gmodels, observations, message', [
    ([gmodel('hi'), gmodel()], [observation('hi')],
     "either all or none of the gmodels must have a name"),
    ([gmodel('hi'), gmodel('hi')], [observation('hi')],
     "the names of the gmodels must be unique"),
    ([gmodel('hi'), gmodel('ha')], [observation()],
     "observation 0 must name its gmodel"),
    ([gmodel('hi')], [observation('ha')], "unknown gmodel 'ha'"),
    ([gmodel('hi')], [observation('hi', name='hi')],
     "the gmodels and the observations must have different names")])
def test_invalid_gmodel_and_observation_names_are_rejected(
        gmodels, observations, message):
    with pytest.raises(Exception, match=message):
        group(gmodels, observations)


def test_names_name_the_extra_outputs():
    import gbkfit.params
    component = disk(name='disk') | dict(bhtraits=None)
    group_ = group(
        [dict(type='intensity_2d', components=[component])],
        [observation(name='hi')])
    params = gbkfit.params.EvaluationParams(group_.pdescs(), dict(
        disk_xpos=0, disk_ypos=0, disk_posa=0, disk_incl=45,
        disk_bpt_a=1, disk_bpt_s=2), constants=group_.constants())
    extra = {}
    group_.model_h(params.evaluate(), extra)
    assert {'hi_gmodel_disk_bdata', 'hi_dcube_lo'} <= set(extra)
