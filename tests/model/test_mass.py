"""
Tests for mass models: the circular velocity of their components against
independent formulas, and the velocity traits 'mass', which take it.
"""

import copy

import numpy as np
import pytest
import scipy.integrate
import scipy.special

from gbkfit.model import model_parser
from gbkfit.model.mass import (
    MassExponentialDisk, MassModel, MassPseudoIsothermal, mass_model_parser)
from gbkfit.utils.parseutils import ConfigError


# The gravitational constant (kpc (km/s)^2 / Msun)
G = 4.30091727e-6


def test_exponential_disk_is_its_potential():
    # The circular velocity of a thin disk from its potential, by the
    # Hankel transform of its surface density (Binney & Tremaine 2.165):
    # v^2 = 2 pi G r int_0^inf S(k) k J1(k r) dk, with S(k) = sigma0 rd^2
    # (1 + k^2 rd^2)^(-3/2) for an exponential disk
    logm, rd = 10.5, 3.0
    sigma0 = 10 ** logm / (2 * np.pi * rd ** 2)

    def vcirc2(r):
        # (integrated between the zeros of J1, up to where the rest of
        # the integral is negligible)
        if r == 0:
            return 0
        edges = np.concatenate(([0], scipy.special.jn_zeros(1, 500) / r))
        integral = sum(
            scipy.integrate.quad(
                lambda k: k * scipy.special.j1(k * r)
                * (1 + (k * rd) ** 2) ** -1.5, a, b)[0]
            for a, b in zip(edges[:-1], edges[1:]))
        return 2 * np.pi * G * sigma0 * rd ** 2 * r * integral

    radius = np.array([0, 0.1, 1, 3, 6.6, 10])
    result = MassExponentialDisk('stars').vcirc2(
        radius, dict(logm=logm, rd=rd))
    np.testing.assert_allclose(
        result, [vcirc2(r) for r in radius], rtol=1e-4)


def test_pseudo_isothermal_is_its_enclosed_mass():
    # Spherical: v^2 = G M(<r) / r, with M(<r) the integral of the density
    logrho0, rc = 7.5, 2.0

    def vcirc2(r):
        mass, _ = scipy.integrate.quad(
            lambda x: 4 * np.pi * x ** 2 * 10 ** logrho0 / (1 + (x / rc) ** 2),
            0, r)
        return G * mass / r

    # (including radii where the series near the centre is used)
    radius = np.array([1e-4, 1.9e-3, 2.1e-3, 0.5, 2, 10, 100])
    result = MassPseudoIsothermal('halo').vcirc2(
        radius, dict(logrho0=logrho0, rc=rc))
    np.testing.assert_allclose(
        result, [vcirc2(r) for r in radius], rtol=1e-6)


def test_mass_model_adds_its_components_in_quadrature():
    # At a distance of 10 Mpc, an arcsec is 48.48 pc
    model = MassModel([
        MassExponentialDisk('stars'), MassPseudoIsothermal('halo')])
    assert list(model.pdescs()) == [
        'distance', 'mass_stars_logm', 'mass_stars_rd', 'mass_halo_logrho0',
        'mass_halo_rc']
    params = dict(
        distance=10, mass_stars_logm=10.5, mass_stars_rd=3,
        mass_halo_logrho0=7.5, mass_halo_rc=5)
    radius = np.array([0, 10, 50, 200])
    kpc = radius * 0.0484813681
    expected = np.sqrt(
        MassExponentialDisk('s').vcirc2(kpc, dict(logm=10.5, rd=3))
        + MassPseudoIsothermal('h').vcirc2(kpc, dict(logrho0=7.5, rc=5)))
    np.testing.assert_allclose(model.vcirc(radius, params), expected, rtol=1e-8)


MASS_MODEL = dict(components=[
    dict(type='exponential_disk', name='stars'),
    dict(type='pseudo_isothermal', name='halo')])

MASS_PROPERTIES = dict(
    distance=10, mass_stars_logm=10.5, mass_stars_rd=0.5,
    mass_halo_logrho0=8, mass_halo_rc=1)

DISK = dict(
    loose=False, tilted=False, rnodes=list(range(0, 21, 2)), rstep=0.5,
    bptraits=dict(type='exponential'), dptraits=dict(type='uniform'))

DISK_PROPERTIES = dict(
    xpos=0, ypos=0, posa=30, incl=60, vsys=0, bpt_a=1, bpt_s=4, dpt_a=15)


def kinematics(model_type, disk_type, vptraits, mass_model=None):
    disk = DISK | dict(type=disk_type, vptraits=vptraits)
    if model_type == 'kinematics_3d':
        disk |= dict(bhtraits=dict(type='sech2'))
    if disk_type == 'mcdisk':
        disk |= dict(cflux=1e-3)
    info = dict(type=model_type, components=[disk])
    if mass_model is not None:
        info |= dict(mass_model=copy.deepcopy(mass_model))
    return model_parser.load(info)


def evaluate(driver, model, properties):
    from gbkfit.observation import Observation, ObservationGroup, PixelSpectra
    from gbkfit.params import EvaluationParams
    group = ObservationGroup([model], [Observation(
        PixelSpectra(size=(32, 32, 51), step=(1, 1, 10)), driver)])
    params = EvaluationParams(group.pdescs(), properties)
    return group.model_h(params.evaluate())[0]['spectra']['d'].copy()


@pytest.mark.parametrize('model_type, disk_type', [
    ('kinematics_2d', 'smdisk'),
    ('kinematics_3d', 'smdisk'),
    ('kinematics_3d', 'mcdisk')])
def test_mass_trait_is_the_circular_velocity_at_the_subnodes(
        driver, model_type, disk_type):
    # A disk with the trait 'mass' is the disk whose node-wise tangential
    # velocity at its subnodes is the circular velocity of the mass model
    massive = kinematics(
        model_type, disk_type, dict(type='mass'), MASS_MODEL)
    given = kinematics(
        model_type, disk_type,
        dict(type='nw_tan_uniform', sampling='subrings'))
    subrnodes = np.array(given.constants()['subrnodes'])
    vcirc = mass_model_parser.load(copy.deepcopy(MASS_MODEL)).vcirc(
        subrnodes, MASS_PROPERTIES)
    assert vcirc.max() > 100
    properties = DISK_PROPERTIES
    if model_type == 'kinematics_3d':
        properties |= dict(bht_s=1)
    np.testing.assert_allclose(
        evaluate(driver, massive, properties | MASS_PROPERTIES),
        evaluate(driver, given, properties | dict(vpt_vt=vcirc)),
        rtol=1e-6, atol=1e-7)


def test_mass_trait_parameters_and_round_trip():
    model = kinematics('kinematics_3d', 'smdisk', dict(type='mass'), MASS_MODEL)
    # The trait has no parameters: those of the mass model are the
    # model's
    assert not any(name.startswith('vpt') for name in model.pdescs())
    assert set(MASS_PROPERTIES) <= set(model.pdescs())
    dumped = model_parser.dump(model)
    assert dumped['mass_model'] == MASS_MODEL
    assert model_parser.dump(model_parser.load(copy.deepcopy(dumped))) \
        == dumped


def test_mass_trait_and_mass_model_need_each_other():
    with pytest.raises(ConfigError, match="need the mass_model"):
        kinematics('kinematics_2d', 'smdisk', dict(type='mass'))
    with pytest.raises(ConfigError, match="no component uses"):
        kinematics(
            'kinematics_2d', 'smdisk', dict(type='tan_arctan'), MASS_MODEL)
