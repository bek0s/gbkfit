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
        'rnodes', 'cmp1_rnodes', 'ocmp_rnodes'}


def test_names_prefix_the_parameters_and_constants():
    gmodel = intensity_3d(
        [disk(name='disk'), disk(name='bulge')], [opacity(name='dust')])
    names = set(gmodel.pdescs())
    assert {'disk_xpos', 'bulge_bpt_s', 'dust_opt_a'} <= names
    assert not {'xpos', 'cmp1_xpos', 'ocmp_xpos'} & names
    assert set(gmodel.constants()) == {
        'disk_rnodes', 'bulge_rnodes', 'dust_rnodes'}


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
