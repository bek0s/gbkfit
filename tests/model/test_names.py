"""
Tests for the names of components and galaxies: a component needs a name
when its galaxy has several, and a galaxy when there are several; the
name prefixes the parameters and constants, and is the id of the extra
outputs.
"""

import pytest

from gbkfit.model import Extras, galaxy_parser


THIN = dict(ptraits=dict(type='exponential'))
THICK = THIN | dict(htraits=dict(type='sech2', zmax=8))


def disk(density=THICK, **options):
    return dict(
        type='smdisk', loose=False, tilted=False, rnodes=[0, 1, 2],
        density=density, emission=dict(type='band'), **options)


def dust(**options):
    return dict(
        type='smdisk', loose=False, tilted=False, rnodes=[0, 1, 2],
        density=THICK, opacity=dict(type='dust'), **options)


def load_galaxy(components):
    return galaxy_parser.load(dict(components=components))


def test_one_component_needs_no_name():
    galaxy = load_galaxy([disk()])
    assert {'xpos', 'pt_a', 'ht_s'} <= set(galaxy.pdescs())
    assert set(galaxy.constants()) == {'rnodes', 'subrnodes'}


def test_names_prefix_the_parameters_and_constants():
    galaxy = load_galaxy(
        [disk(name='disk'), disk(name='bulge'), dust(name='dust')])
    names = set(galaxy.pdescs())
    assert {'disk_xpos', 'bulge_pt_s', 'dust_pt_a'} <= names
    assert not {'xpos', 'pt_a'} & names
    assert set(galaxy.constants()) == {
        'disk_rnodes', 'disk_subrnodes', 'bulge_rnodes', 'bulge_subrnodes',
        'dust_rnodes', 'dust_subrnodes'}


def test_names_do_not_depend_on_the_order():
    first = load_galaxy([disk(name='disk'), disk(name='bulge')])
    second = load_galaxy([disk(name='bulge'), disk(name='disk')])
    assert set(first.pdescs()) == set(second.pdescs())


def test_names_are_dumped():
    info = galaxy_parser.dump(load_galaxy([disk(name='disk')]))
    assert info['components'][0]['name'] == 'disk'
    info = galaxy_parser.dump(load_galaxy([disk()]))
    assert 'name' not in info['components'][0]


@pytest.mark.parametrize('components, message', [
    ([disk(), disk()],
     r"the components of a galaxy need names when there are several; the "
     r"components \[0, 1\] have none"),
    ([disk(name='disk'), dust()],
     r"the components of a galaxy need names when there are several; the "
     r"components \[1\] have none"),
    ([disk(name='disk'), disk(name='disk')],
     r"the names of the components must be unique; repeated: \['disk'\]"),
    ([disk(name='disk'), dust(name='disk')],
     r"the names of the components must be unique; repeated: \['disk'\]"),
    ([disk(name='my disk')], "invalid name 'my disk'")])
def test_invalid_names_are_rejected(components, message):
    with pytest.raises(Exception, match=message):
        load_galaxy(components)


def galaxy(name=None, components=None):
    if components is None:
        components = [disk(THIN)]
    return dict(components=components) \
        | (dict(name=name) if name is not None else {})


def observation(galaxies=None, name=None):
    return dict(
        type='sky',
        observable=dict(type='pixel_brightness', size=[8, 8]),
        line_of_sight=dict(type='flat')) \
        | (dict(galaxies=galaxies) if galaxies is not None else {}) \
        | (dict(name=name) if name is not None else {})


def group(galaxies, observations):
    from gbkfit.observation import ObservationGroup, observation_parser
    return ObservationGroup(
        galaxy_parser.load(galaxies), observation_parser.load(observations))


def test_galaxy_names_prefix_the_parameters():
    group_ = group(
        [galaxy('hi'), galaxy('halpha')],
        [observation(['hi']), observation(['halpha'])])
    assert {'hi_xpos', 'halpha_xpos'} <= set(group_.pdescs())
    assert set(group_.constants()) == {
        'hi_rnodes', 'hi_subrnodes', 'halpha_rnodes', 'halpha_subrnodes'}
    assert galaxy_parser.dump(group_.galaxies()[0])['name'] == 'hi'


def test_observations_of_one_galaxy_share_its_parameters():
    group_ = group([galaxy()], [observation(), observation()])
    assert 'xpos' in group_.pdescs()
    assert not any(name.startswith('galaxy') for name in group_.pdescs())


@pytest.mark.parametrize('galaxies, observations, message', [
    ([galaxy('hi'), galaxy()], [observation(['hi'])],
     "the galaxies need names when there are several"),
    ([galaxy('hi'), galaxy('hi')], [observation(['hi'])],
     "the names of the galaxies must be unique"),
    ([galaxy('hi'), galaxy('ha')], [observation()],
     "observation 0 must name the galaxies it sees"),
    ([galaxy('hi')], [observation(['ha'])],
     r"observation 0 refers to unknown galaxies \['ha'\]"),
    ([galaxy('hi')], [observation(['hi'], name='hi')],
     "the galaxies and the observations must have different names")])
def test_invalid_galaxy_and_observation_names_are_rejected(
        galaxies, observations, message):
    with pytest.raises(Exception, match=message):
        group(galaxies, observations)


# The properties of a thin disk
DISK_PROPERTIES = dict(xpos=0, ypos=0, posa=0, incl=45, pt_a=1, pt_s=2)


def extra_outputs(group_, prefixes):
    """
    The names of the extra outputs of an evaluation of a group of thin
    disks, whose parameters have the given prefixes.
    """
    import gbkfit.params
    properties = {
        prefix + name: value
        for prefix in prefixes for name, value in DISK_PROPERTIES.items()}
    params = gbkfit.params.ParamSpace(
        group_.pdescs(), properties, constants=group_.constants())
    extra = Extras()
    group_.model_h(params.evaluate(), extra)
    return {(x.name, *x.ids.values()) for x in extra}


def test_names_name_the_extra_outputs():
    group_ = group(
        [galaxy(components=[disk(THIN, name='disk')])],
        [observation(name='hi')])
    extra = extra_outputs(group_, ['disk_'])
    assert {('bdata', 'hi', 'galaxy0', 'disk'), ('dcube_lo', 'hi')} <= extra


def test_galaxy_names_prefix_the_extra_outputs():
    components = [disk(THIN, name='disk')]
    group_ = group(
        [galaxy('north', components), galaxy('south', components)],
        [observation(['north', 'south'], name='hi')])
    extra = extra_outputs(group_, ['north_disk_', 'south_disk_'])
    assert {('bdata', 'hi', 'north', 'disk'),
            ('bdata', 'hi', 'south', 'disk')} <= extra
