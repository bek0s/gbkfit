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


def models(*names):
    from gbkfit.model import ModelGroup, model_parser
    component = disk() | dict(bhtraits=None)
    return ModelGroup(model_parser.load([
        dict(driver=dict(type='host'), dmodel=dict(type='image', size=[8, 8]),
             gmodel=dict(type='intensity_2d', components=[component]))
        | (dict(name=name) if name is not None else {})
        for name in names]))


def test_model_names_prefix_the_parameters():
    assert {'xpos', 'model1_xpos'} <= set(models(None, None).pdescs())
    group = models('hi', 'halpha')
    assert {'hi_xpos', 'halpha_xpos'} <= set(group.pdescs())
    assert set(group.constants()) == {
        'hi_rnodes', 'hi_subrnodes', 'halpha_rnodes', 'halpha_subrnodes'}
    assert group.models()[0].dump()['name'] == 'hi'


@pytest.mark.parametrize('names, message', [
    (('hi', None), "either all or none of the models must have a name"),
    (('hi', 'hi'), r"the names of the models must be unique")])
def test_invalid_model_names_are_rejected(names, message):
    with pytest.raises(Exception, match=message):
        models(*names)


def test_names_name_the_extra_outputs(evaluate_models):
    component = disk(name='disk') | dict(bhtraits=None)
    model = dict(
        name='hi', driver=dict(type='host'),
        dmodel=dict(type='image', size=[8, 8]),
        gmodel=dict(type='intensity_2d', components=[component]))
    _, extra = evaluate_models([model], dict(
        hi_disk_xpos=0, hi_disk_ypos=0, hi_disk_posa=0, hi_disk_incl=45,
        hi_disk_bpt_a=1, hi_disk_bpt_s=2))
    assert {'hi_gmodel_disk_bdata', 'hi_dmodel_dcube_lo'} <= set(extra)
