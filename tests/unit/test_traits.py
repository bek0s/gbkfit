"""
Tests for the validation of trait options.
"""

import inspect

import numpy as np
import pytest
import scipy.integrate

from gbkfit.model.components import traits
from gbkfit.utils.parseutils import ConfigError


def kernel_consts_cases():
    """
    Each type of trait, options for it, and the constants the kernels
    read from it (traits.hpp): the truncation and the node-wise switch of
    the density heights, the node-wise switch of the other
    heights, the largest displacement of the vertical distortions (and
    the order of a harmonic one), the number of blobs of mixtures, the
    order of harmonics, and the grid of a map. The other traits have
    none. The velocity traits (but mass) are given a direction, which the
    kernels read apart from the constants (vpt_directions).
    """
    parsers = dict(
        rpt=traits.rpt_parser, rht=traits.rht_parser,
        vpt=traits.vpt_parser, vht=traits.vht_parser,
        dpt=traits.dpt_parser, dht=traits.dht_parser,
        zpt=traits.zpt_parser, spt=traits.spt_parser)
    for kind, parser in parsers.items():
        for type_, cls in parser.registered_classes().items():
            # (an image is a file; its constants are those of the map, see
            # test_image_light)
            if cls is traits.RPTraitImage:
                continue
            if kind == 'rht':
                options, consts = dict(zmax=5.0, rnodes=True), (5.0, True)
            elif kind == 'vht' and type_ == 'mass':
                options, consts = dict(zmax=5.0), (5.0, 33)
            elif kind in ('vht', 'dht'):
                options, consts = dict(), (False,)
            elif kind == 'zpt' and type_.endswith('harmonic'):
                options, consts = dict(zmax=5.0, order=2), (5.0, 2)
            elif kind == 'zpt':
                options, consts = dict(zmax=5.0), (5.0,)
            elif type_.startswith('mixture'):
                options, consts = dict(nblobs=3), (3,)
            elif type_.endswith('harmonic'):
                options, consts = dict(order=2), (2,)
            elif type_ == 'spiral':
                options, consts = dict(order=2), (2, 4)
            elif type_ == 'map':
                options, consts = dict(size=[3, 2], step=2.0), (3, 2, 2.0)
            else:
                options, consts = dict(), ()
            if issubclass(cls, traits.VPTrait) and \
                    'direction' in inspect.signature(cls).parameters:
                options['direction'] = 'tan'
            yield pytest.param(cls, options, consts, id=f"{kind}-{type_}")


@pytest.mark.parametrize('cls, options, consts', list(kernel_consts_cases()))
def test_traits_give_the_constants_the_kernels_read(cls, options, consts):
    assert cls(**options).consts() == consts


def test_velocity_traits_name_their_speed_after_their_direction():
    assert list(traits.VPTraitArctan('rad').params_sm()) == ['rt', 'vr']
    assert list(traits.VPTraitNWUniform('ver').params_rnw(3)) == ['vv']
    assert list(traits.VPTraitUniform('out').params_sm()) == ['vo']
    with pytest.raises(ConfigError, match="direction must be one of"):
        traits.VPTraitArctan('up')


def test_harmonic_orders_are_integers():
    # (a configuration is checked against the annotation; the Python API
    # by the trait)
    with pytest.raises(ConfigError, match="order must be an integer"):
        traits.RPTraitNWHarmonic(order=1.5)


# The subnodes of a disk of clouds: its edges 0 and 6, and the middles of
# its three rings, of half width 0.75 (see density._rings)
SUBRNODES = np.array([0, 0.75, 2.25, 3.75, 5.25, 6])


def ring_integrals(azimuthal, knots=()):
    """
    The integral of a density over each ring of SUBRNODES, of the
    integral of its |value| over the azimuths at each radius
    (azimuthal(r)), split at the radii of its kinks (knots).
    """
    middles = SUBRNODES[1:-1]
    return np.array([scipy.integrate.quad(
        lambda r: r * azimuthal(r), middle - 0.75, middle + 0.75,
        points=[k for k in (middle, *knots)
                if middle - 0.75 < k < middle + 0.75],
        epsabs=1e-13, epsrel=1e-12)[0] for middle in middles])


@pytest.mark.parametrize('trait, params, profile', [
    (traits.RPTraitExponential(), dict(a=2.0, s=1.5),
     lambda r: 2 * np.exp(-r / 1.5)),
    (traits.RPTraitGauss(), dict(a=2.0, s=1.5),
     lambda r: 2 * np.exp(-r ** 2 / 4.5)),
    (traits.RPTraitGGauss(), dict(a=2.0, s=1.5, b=0.7),
     lambda r: 2 * np.exp(-(r / 1.5) ** 0.7)),
    (traits.RPTraitLorentz(), dict(a=2.0, s=1.5),
     lambda r: 2 * 2.25 / (r ** 2 + 2.25)),
    (traits.RPTraitMoffat(), dict(a=2.0, s=1.5, b=2.5),
     lambda r: 2 * (1 + (r / 1.5) ** 2) ** -2.5),
    (traits.RPTraitSech2(), dict(a=2.0, s=1.5),
     lambda r: 2 / np.cosh(r / 1.5) ** 2)])
def test_the_rings_of_clouds_have_the_flux_of_the_profile_over_them(
        trait, params, profile):
    np.testing.assert_allclose(
        trait.cloud_flux(params, SUBRNODES),
        ring_integrals(lambda r: 2 * np.pi * profile(r)), rtol=1e-8)


def test_the_rings_of_clouds_of_node_wise_traits_have_their_flux():
    # (of |the trait|: the clouds take its sign, which a changes in a ring;
    # |cos(2 (t - p))| integrates to 4 over a turn, whatever p)
    a = np.array([1, 0.5, -0.5, 0.2, 1, 0.3])
    p = np.array([0, 10, 20, 30, 40, 50])
    s = np.array([0.5, 1, 1.5, 0.5, 2, 1])
    zeros = (1.5, 2.25 + 1.5 * 0.5 / 0.7)

    def at(values, r):
        return np.interp(r, SUBRNODES, values)
    harmonic = traits.RPTraitNWHarmonic(2)
    np.testing.assert_allclose(
        harmonic.cloud_flux(dict(a=a, p=p), SUBRNODES),
        ring_integrals(lambda r: 4 * np.abs(at(a, r)), zeros), rtol=1e-8)
    distortion = traits.RPTraitNWDistortion()
    np.testing.assert_allclose(
        distortion.cloud_flux(dict(a=a, p=p, s=s), SUBRNODES),
        ring_integrals(lambda r: np.abs(at(a, r)) * scipy.integrate.quad(
            lambda t: np.exp(-(t * r) ** 2 / (2 * at(s, r) ** 2)),
            -np.pi, np.pi, epsabs=1e-14, epsrel=1e-13)[0], zeros),
        rtol=1e-8)
