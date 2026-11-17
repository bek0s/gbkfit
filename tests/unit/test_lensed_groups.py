"""
Tests for groups of galaxies and observations with gravitational lenses
(see ObservationGroup): the parameters of the lenses, the planes that
each observation sees, the refusals of what cannot be lensed, and the
section 'lenses' of configurations.
"""

import logging

import astropy.units as u
import pytest

import gbkfit
from gbkfit.config import FORMAT_VERSION
from gbkfit.lensing import Lens, LensShear, LensSIE
from gbkfit.model import galaxy_parser
from gbkfit.observation import (
    DepthGroup, ObservationGroup, ObservationPixelBrightness,
    ObservationProfile, Weights, WeightAzimuthPower)


def disk(name='disk', **options):
    """A thin disk that emits in a band."""
    return dict(
        type='thin_disk', name=name, loose=False, tilted=False,
        rnodes=[0, 1, 2], rptraits=dict(type='exponential'),
        emission=dict(type='band')) | options


def galaxy(name, redshift=None, components=None):
    """A galaxy of a thin disk (or of the given components)."""
    info = dict(name=name, components=components or [disk()])
    if redshift is not None:
        info['redshift'] = redshift
    return galaxy_parser.load(info)


def image(galaxies=None, **options):
    """An image of the galaxies of the given names."""
    return ObservationPixelBrightness(
        size=(16, 16), step=0.1, galaxies=galaxies, **options)


def lens(name='lens', redshift=0.3):
    """A lens of an SIE and a shear, quoted for a source at z = 2."""
    return Lens(
        name, redshift, [LensSIE(name='main'), LensShear(name='shear')],
        source_redshift=2.0)


def lensed_group(observations=None, **options):
    """
    A group of a galaxy at the redshift of a lens in front of a galaxy at
    z = 2, and an image of both (or the given observations).
    """
    return ObservationGroup(
        [galaxy('deflector', 0.3), galaxy('arc', 2.0)],
        observations or [image(['deflector', 'arc'], name='hst')],
        [lens()], **options)


def test_the_parameters_of_a_lens_are_prefixed_by_its_name():
    group = lensed_group()
    names = set(group.pdescs())
    assert {'lens_main_theta_e', 'lens_main_q', 'lens_shear_gamma',
            'lens_shear_posa'} <= names
    assert group.lens_mappings() == ({
        name: f'lens_{name}' for name in lens().pdescs()},)
    # (its model reads them: they bend the light of the arc)
    assert 'lens_main_theta_e' in group.model_params(0)


@pytest.mark.parametrize('names, message', [
    (('deflector', 'hst', 'deflector'), "galaxies and the lenses"),
    (('deflector', 'hst', 'hst'), "observations and the lenses"),
    (('deflector', 'hst', 'arc'), "galaxies and the lenses")])
def test_names_are_unique_among_galaxies_observations_and_lenses(
        names, message):
    galaxy_name, observation_name, lens_name = names
    with pytest.raises(ValueError, match=message):
        ObservationGroup(
            [galaxy(galaxy_name, 0.3), galaxy('arc', 2.0)],
            image([galaxy_name, 'arc'], name=observation_name),
            [lens(lens_name)])


def test_lenses_need_the_redshifts_of_the_galaxies():
    with pytest.raises(ValueError, match="needs a redshift above 0"):
        ObservationGroup(
            [galaxy('deflector'), galaxy('arc', 2.0)],
            image(['deflector', 'arc']), [lens()])


def test_an_observation_sees_the_image_plane_and_those_behind_lenses():
    # The galaxy of the plane of the lens, and one in front of it, are
    # drawn on the image (by redshift); the arc on a plane of its own
    group = ObservationGroup(
        [galaxy('arc', 2.0), galaxy('deflector', 0.3),
         galaxy('near', 0.1)],
        image(['arc', 'deflector', 'near']), [lens()])
    seen = group.seen_by(0)
    tracer = group.tracer()
    assert [plane.redshift for plane in tracer.planes()] == [0.1, 0.3, 2.0]
    # (the galaxies by their positions among those it sees: arc,
    # deflector, near; those of the image plane near to far)
    assert [(plane.galaxies, plane.traced) for plane in seen.planes] == [
        ((1, 2), None), ((0,), 2)]
    assert [[unit.galaxies for unit in plane.order]
            for plane in seen.planes] == [[(1,), (0,)], [(0,)]]
    assert seen.lensed()
    assert seen.tracer is tracer


def test_without_lenses_an_observation_sees_one_plane():
    group = ObservationGroup(
        [galaxy('a'), galaxy('b')], image(['a', 'b']),
        depth_order=['b', 'a'])
    seen = group.seen_by(0)
    assert group.tracer() is None
    assert len(seen.planes) == 1
    assert seen.planes[0].traced is None
    assert [unit.galaxies for unit in seen.planes[0].order] == [(1,), (0,)]
    assert not seen.lensed()


def test_the_depth_order_orders_the_galaxies_of_a_plane():
    # Two galaxies at the redshift of the lens that pass through each other
    group = ObservationGroup(
        [galaxy('a', 0.3), galaxy('b', 0.3001), galaxy('arc', 2.0)],
        image(['a', 'b', 'arc']), [lens()],
        depth_order=[DepthGroup('pair', ('b', 'a'))])
    image_plane = group.seen_by(0).planes[0]
    assert image_plane.galaxies == (0, 1)
    [unit] = image_plane.order
    assert unit.galaxies == (1, 0) and unit.name == 'pair'


def test_profiles_are_never_lensed():
    # (a profile is of the galaxy itself, not of its light on the sky)
    arc = galaxy_parser.load(dict(
        name='arc', redshift=2.0, components=[disk()],
        mass=[dict(type='halo', name='halo', profile=dict(type='nfw'))],
        distance=dict(type='parameter')))
    group = ObservationGroup(
        [galaxy('deflector', 0.3), arc],
        [image(['deflector', 'arc'], name='hst'),
         ObservationProfile(
             ['vcirc'], [1, 2] * u.arcsec, galaxies=['arc'], name='rc')],
        [lens()])
    seen = group.seen_by(1)
    assert [plane.traced for plane in seen.planes] == [None]
    assert not any(name.startswith('lens_')
                   for name in group.model_params(1))


def test_a_lens_that_bends_nothing_seen_is_warned_of(caplog):
    with caplog.at_level(logging.WARNING):
        ObservationGroup(
            [galaxy('near', 0.1), galaxy('deflector', 0.3)],
            image(['near', 'deflector']), [lens()])
    assert "bends the light of no galaxy that an observation sees" \
        in caplog.text


def test_an_observation_of_the_deflector_alone_reads_no_lens():
    group = lensed_group([
        image(['deflector', 'arc'], name='hst'),
        image(['deflector'], name='ground')])
    assert not any(name.startswith('lens_')
                   for name in group.model_params(1))
    assert not group.seen_by(1).lensed()


def test_pixel_weights_of_a_lensed_disk_are_refused():
    weights = Weights('arc.disk', WeightAzimuthPower(2))
    with pytest.raises(ValueError, match="whose light no lens bends"):
        lensed_group([image(
            ['deflector', 'arc'], name='hst', pixel_weights=weights)])
    # (those of the disk of the deflector are in the plane of the image)
    lensed_group([image(
        ['deflector', 'arc'], name='hst',
        pixel_weights=Weights('deflector.disk', WeightAzimuthPower(2)))])


def test_galaxies_behind_a_lens_whose_light_is_not_seen_are_refused():
    # (an observation that sees none of the light behind the lens draws
    # the image plane alone, of the galaxies there)
    with pytest.raises(ValueError, match="sees none of the light of"):
        lensed_group([image(
            ['deflector', 'arc'], name='hst', components=['deflector.disk'])])


def lensed_config(**observation):
    """A configuration of an image of a lensed arc, without data."""
    return dict(
        version=FORMAT_VERSION,
        galaxies=[
            dict(name='deflector', redshift=0.3, components=[disk()]),
            dict(name='arc', redshift=2.0, components=[disk()])],
        lenses=[dict(
            name='lens', redshift=0.3, source_redshift=2.0,
            pieces=dict(type='sie'))],
        observations=[dict(
            type='pixel_brightness', name='hst', size=[16, 16], step=0.1,
            galaxies=['deflector', 'arc']) | observation],
        params=dict(properties=dict(
            lens_theta_e=dict(value=0.5, min=0.1, max=1), lens_q=1,
            lens_posa=0, lens_xpos=0, lens_ypos=0,
            **{f'{g}_disk_{p}': v for g in ('deflector', 'arc')
               for p, v in dict(xpos=0, ypos=0, posa=0, incl=30, rpt_a=1,
                                rpt_s=0.3).items()})))


def test_configurations_have_a_section_of_lenses():
    description = gbkfit.describe(lensed_config(
        source_grid=dict(step=0.01, extent=[-1, 1, -1, 1])))
    assert not [p for p in description.problems if p.severity == 'error']
    group = description.group
    assert [lens.name() for lens in group.lenses()] == ['lens']
    assert group.observations()[0].source_grid().step() == 0.01
    schema = gbkfit.schema()
    assert 'lenses' in schema['properties']
    assert 'lens_piece' in gbkfit.catalogue()


def test_a_lens_of_a_configuration_without_redshift_is_refused_at_its_path():
    config = lensed_config()
    del config['lenses'][0]['redshift']
    description = gbkfit.describe(config)
    [error] = [p for p in description.problems if p.severity == 'error']
    assert error.path[:2] == ['lenses', 0]
