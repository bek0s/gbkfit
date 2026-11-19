"""
Tests for groups of galaxies and observations with gravitational lenses
(see ObservationGroup): the parameters of the lenses, the planes that
each observation sees, the refusals of what cannot be lensed, the
pieces that are the masses of galaxies (see LensGalaxyMass) and their
checks, and the section 'lenses' of configurations.
"""

import logging

import astropy.units as u
import pytest

import gbkfit
from gbkfit.config import FORMAT_VERSION
from gbkfit.lensing import (
    Lens, LensGalaxyMass, LensShear, LensSIE, lens_parser)
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
            lens_sie_theta_e=dict(value=0.5, min=0.1, max=1), lens_sie_q=1,
            lens_sie_posa=0, lens_sie_xpos=0, lens_sie_ypos=0,
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


# The galaxy whose mass is a lens: a gas disk that its mass moves (which
# places its mass on the sky), an NFW halo, and a cosmological distance
GAS = dict(
    type='thin_disk', name='gas', loose=False, tilted=False,
    rnodes=[0, 1, 2], rptraits=dict(type='exponential'),
    vptraits=dict(type='mass'), dptraits=dict(type='uniform'),
    emission=dict(type='lines'))

HALO = dict(type='halo', name='halo', profile=dict(type='nfw'))


def deflector(redshift=0.3, **options):
    """A galaxy of mass whose mass is that of a lens (see GAS and HALO)."""
    return galaxy_parser.load(dict(
        name='deflector', redshift=redshift, components=[GAS],
        mass=[HALO], distance=dict(type='cosmological')) | options)


def mass_lens(*pieces, redshift=0.3, galaxy='deflector'):
    """
    A lens of the mass of a galaxy, and other pieces (quoted for a source
    at z = 2).
    """
    return Lens(
        'lens', redshift, [LensGalaxyMass(galaxy), *pieces],
        source_redshift=2.0 if pieces else None)


def mass_group(galaxy=None, lens=None, observations=None):
    """
    A group of the deflector (or the given galaxy) in front of an arc at
    z = 2, and an image of the arc alone.
    """
    return ObservationGroup(
        [galaxy or deflector(), galaxy_parser.load(dict(
            name='arc', redshift=2.0, components=[disk()]))],
        observations or [image(['arc'], name='alma')],
        [lens or mass_lens()])


def test_pieces_are_named_after_their_types():
    # [mass, shear] needs no names: the shear's parameters are 'shear_*'
    lens = lens_parser.load(dict(
        name='lens', redshift=0.3, source_redshift=2.0,
        pieces=[dict(type='mass', galaxy='deflector'), dict(type='shear')]))
    assert [piece.name() for piece in lens.pieces()] == ['mass', 'shear']
    assert list(lens.pdescs()) == ['shear_gamma', 'shear_posa']
    assert lens.galaxies() == ('deflector',)
    assert lens_parser.dump(lens)['pieces'] == [
        dict(type='mass', galaxy='deflector'), dict(type='shear')]


def test_the_mass_of_a_galaxy_bends_what_does_not_see_it():
    # The image of the arc alone is bent by the mass of the deflector: its
    # model takes the mass on the sky of the deflector, not its parameters
    galaxy = deflector(components=[GAS, disk('stars')])
    group = mass_group(galaxy, mass_lens(LensShear()), [
        image(['arc'], name='alma'), image(['deflector'], name='ground')])
    assert group.lensing_galaxies() == (0,)
    assert group.bending_masses(0) == (0,)
    assert group.bending_masses(1) == ()
    params = group.model_params(0)
    assert 'lens_shear_gamma' in params
    assert not any(name.startswith('deflector_') for name in params)


def test_a_galaxy_whose_mass_only_lenses_is_not_warned_of(caplog):
    # (a galaxy of no 'mass' traits whose mass is that of a lens: its mass
    # is used; its stars place it, followed by its mass)
    stars = disk('stars')
    galaxy = galaxy_parser.load(dict(
        name='deflector', redshift=0.3, components=[stars],
        mass=[dict(type='traced_disk', name='stars', component='stars'),
              HALO], distance=dict(type='cosmological')))
    with caplog.at_level(logging.WARNING):
        mass_group(galaxy)
    assert "only derived values can use it" not in caplog.text


@pytest.mark.parametrize('lens, message', [
    (mass_lens(galaxy='other'), "which is not one of the galaxies"),
    (Lens('lens', 0.3, [LensGalaxyMass('deflector', name='a'),
                        LensGalaxyMass('deflector', name='b')]),
     "would bend light twice"),
    (mass_lens(redshift=0.5), "within 3000 km/s of it")])
def test_a_piece_of_the_mass_of_a_galaxy_is_checked(lens, message):
    with pytest.raises(ValueError, match=message):
        mass_group(lens=lens)


def test_two_lenses_of_the_mass_of_one_galaxy_are_refused():
    with pytest.raises(ValueError, match="would bend light twice"):
        ObservationGroup(
            [deflector(), galaxy('arc', 2.0)], image(['arc']),
            [mass_lens(), Lens('other', 0.3, LensGalaxyMass('deflector'))])


@pytest.mark.parametrize('options, message', [
    (dict(components=[disk()], mass=None, distance=None), "which has none"),
    (dict(distance=dict(type='parameter'), redshift=0.3), None),
    (dict(components=[], mass=[HALO]), None),
    (dict(gravity=dict(type='mond', interpolation='simple')),
     "relativistic theory"),
    (dict(mass=[HALO, dict(
        type='table', name='table', radii=dict(value=[1, 2], unit='kpc'),
        velocities=[100, 120])]), "do not say where their mass is"),
    (dict(mass=[dict(type='disk', name='thin', surface=dict(
        type='exponential'), vertical=dict(type='thin'))],
        components=[GAS | dict(type='thin_disk')]), None)])
def test_what_the_galaxy_of_a_mass_lens_must_give(options, message):
    galaxy = deflector(**options)
    if message is None:
        mass_group(galaxy)
        return
    with pytest.raises(ValueError, match=message):
        mass_group(galaxy)


def test_the_mass_of_a_lens_has_a_place_on_the_sky():
    # A galaxy whose mass a lens bends light by has the parameters of where
    # its mass is on the sky, in the group: its centre (a halo lies at it);
    # one that no lens takes has none (e.g. of its rotation curve alone)
    group = mass_group()
    assert {'deflector_mass_xpos', 'deflector_mass_ypos'} \
        <= group.pdescs().keys()
    assert 'deflector_mass_posa' not in group.pdescs()
    alone = deflector()
    assert not alone.sky_pdescs().keys() & alone.pdescs().keys()
    seen = ObservationGroup(
        [alone, galaxy_parser.load(dict(
            name='arc', redshift=2.0, components=[disk()]))],
        [image(['arc'], name='alma')])
    assert not any('mass_xpos' in name for name in seen.pdescs())


def test_a_mass_in_its_plane_has_a_plane_of_its_own():
    # A disk of mass lies in the plane of the mass, its own (mass_posa and
    # mass_incl), whatever the components that its mass follows (a
    # sphere here)
    spheroid = dict(
        type='spheroid', name='stars', profile=dict(type='sersic'),
        rmax=5, emission=dict(type='band'))
    traced = dict(type='traced_spheroid', name='stars', component='stars')
    galaxy = deflector(components=[spheroid], mass=[traced, dict(
        type='disk', name='disk', surface=dict(type='exponential'),
        vertical=dict(type='sech2'))])
    mass_group(galaxy)
    assert {'mass_posa', 'mass_incl'} <= galaxy.sky_pdescs().keys()
    # (a halo has none)
    galaxy = deflector(components=[spheroid], mass=[traced, HALO])
    mass_group(galaxy)
    assert 'mass_posa' not in galaxy.sky_pdescs()


@pytest.mark.parametrize('gravity', ['hu_sawicki', 'palatini'])
def test_f_r_gravity_bends_light_as_its_mass(gravity):
    # (Hu-Sawicki: Zhao, Li & Koyama 2011; Palatini: Toniato et al. 2020)
    galaxy = deflector(gravity=dict(type=gravity))
    assert mass_group(galaxy).lensing_galaxies() == (0,)


def test_rings_that_are_not_ellipses_place_no_mass_and_describe_says_so():
    # A disk of a vertical distortion of order 2, which a mass follows,
    # has rings that are not ellipses on the sky: its galaxy cannot lens,
    # as gbkfit.describe says at the piece; a disk that the mass only
    # moves is not refused (its rings place no mass)
    bowl = dict(
        type='thick_disk', name='gas', loose=False, tilted=False,
        rnodes=[0, 1, 2], rptraits=dict(type='exponential'),
        rhtraits=dict(type='sech2', zmax=1),
        zptraits=dict(type='nw_harmonic', order=2, zmax=1),
        emission=dict(type='band'))
    config = lensed_config()
    config['galaxies'][0] |= dict(
        components=[bowl], distance=dict(type='cosmological'),
        mass=[dict(type='traced_disk', name='gas', component='gas')])
    config['lenses'][0] = dict(
        name='lens', redshift=0.3,
        pieces=[dict(type='mass', galaxy='deflector')])
    description = gbkfit.describe(config)
    [error] = [p for p in description.problems if p.severity == 'error']
    assert error.path[:4] == ['lenses', 0, 'pieces', 0]
    assert "not ellipses on the sky" in error.message
    moved = bowl | dict(
        vptraits=dict(type='mass'), dptraits=dict(type='uniform'),
        emission=dict(type='lines'))
    mass_group(deflector(components=[moved]))
