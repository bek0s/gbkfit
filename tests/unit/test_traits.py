"""
Tests for the validation of trait options.
"""

import inspect

import numpy as np
import pytest
import scipy.integrate

from gbkfit.model.components import traits
from gbkfit.model.components.traits import shapes


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
                options, consts = dict(zmax=5.0, nodewise=True), (5.0, True)
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


PARSERS = dict(
    rpt=traits.rpt_parser, rht=traits.rht_parser,
    vpt=traits.vpt_parser, vht=traits.vht_parser,
    dpt=traits.dpt_parser, dht=traits.dht_parser,
    zpt=traits.zpt_parser, spt=traits.spt_parser)

# The families whose kernels take the shapes (polar) or the factors
# (height) that they share in the default case of their switch (see
# traits.hpp), by the suffix of the classes of those in shapes
SHARING = dict(rpt='Shape', vpt='Shape', dpt='Shape', zpt='Shape',
               vht='Factor', dht='Factor')


@pytest.mark.parametrize('kind', PARSERS)
def test_each_type_of_a_family_has_a_uid_of_its_own(kind):
    # (the kernels take an image as a map, of its uid)
    uids = [cls.uid() for cls in PARSERS[kind].registered_classes().values()
            if cls is not traits.RPTraitImage]
    assert len(set(uids)) == len(uids), sorted(uids)


@pytest.mark.parametrize('kind', SHARING)
def test_a_shape_has_one_uid_in_every_family(kind):
    shared = {
        cls.type(): cls.uid() for cls in (
            getattr(shapes, name) for name in shapes.__all__
            if name.endswith(SHARING[kind]))}
    for type_, cls in PARSERS[kind].registered_classes().items():
        if type_ in shared:
            assert cls.uid() == shared[type_], type_
        else:
            assert cls.uid() not in shared.values(), type_


def test_velocity_traits_name_their_speed_after_their_direction():
    assert list(traits.VPTraitArctan('rad').params_sm()) == ['rt', 'vr']
    assert list(traits.VPTraitNWUniform('ver').params_rnw(3)) == ['vv']
    assert list(traits.VPTraitUniform('out').params_sm()) == ['vo']
    with pytest.raises(ValueError, match="'direction' must be one of"):
        traits.VPTraitArctan('up')


def test_harmonic_orders_are_integers():
    # (a configuration is checked against the annotation; the Python API
    # by the trait)
    with pytest.raises(TypeError, match="'order' must be of type int"):
        traits.RPTraitNWHarmonic(order=1.5)


# The subnodes of a disk of clouds: its edges 0 and 6, and the middles of
# its three rings, of half width 0.75 (see density._rings)
SUBRNODES = np.array([0, 0.75, 2.25, 3.75, 5.25, 6])


def ring_integrals(azimuthal, knots=(), tail=None):
    """
    The integral of a density over each ring of SUBRNODES, of the
    integral of its |value| over the azimuths at each radius
    (azimuthal(r)), split at the radii of its kinks (knots), and over
    its tail beyond the last subnode, of tail(r) (see tail_scale).
    """
    middles = SUBRNODES[1:-1]
    rings = [scipy.integrate.quad(
        lambda r: r * azimuthal(r), middle - 0.75, middle + 0.75,
        points=[k for k in (middle, *knots)
                if middle - 0.75 < k < middle + 0.75],
        epsabs=1e-13, epsrel=1e-12)[0] for middle in middles]
    beyond = scipy.integrate.quad(
        lambda r: r * tail(r), SUBRNODES[-1], np.inf,
        epsabs=1e-13, epsrel=1e-12)[0]
    return np.array(rings + [beyond])


def tail_scale(log_slope):
    """
    The scale of the tail of a density of the given log-slope at the last
    subnode rl: -1 / log_slope where it falls, at most rl, else rl.
    """
    rl = SUBRNODES[-1]
    return min(-1 / log_slope, rl) if log_slope < 0 else rl


def with_tail(profile):
    """
    A radial profile up to the last subnode rl, and beyond, its value at
    rl times exp(-(r - rl) / h), of its log-slope there (numerically).
    """
    rl, eps = SUBRNODES[-1], 1e-6
    scale = tail_scale(
        (np.log(profile(rl + eps)) - np.log(profile(rl - eps))) / (2 * eps))
    return lambda r: profile(min(r, rl)) * np.exp(-max(r - rl, 0) / scale)


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
    tailed = with_tail(profile)
    np.testing.assert_allclose(
        trait.cloud_flux(params, SUBRNODES),
        ring_integrals(
            lambda r: 2 * np.pi * profile(r),
            tail=lambda r: 2 * np.pi * tailed(r)),
        rtol=1e-8)


def test_the_rings_of_clouds_of_node_wise_traits_have_their_flux():
    # (of |the trait|: the clouds take its sign, which a changes in a ring;
    # |cos(2 (t - p))| integrates to 4 over a turn, whatever p)
    a = np.array([1, 0.5, -0.5, 0.2, 1, 0.3])
    p = np.array([0, 10, 20, 30, 40, 50])
    s = np.array([0.5, 1, 1.5, 0.5, 2, 1])
    zeros = (1.5, 2.25 + 1.5 * 0.5 / 0.7)

    def at(values, r):
        return np.interp(r, SUBRNODES, values)

    # (beyond the last subnode: a times its tail, of the log-slope of its
    # last interval)
    scale = tail_scale(
        (a[-1] - a[-2]) / ((SUBRNODES[-1] - SUBRNODES[-2]) * a[-1]))

    def amplitude(r):
        return np.abs(at(a, r)) * np.exp(-max(r - SUBRNODES[-1], 0) / scale)

    def distortion_mean(r):
        return amplitude(r) * scipy.integrate.quad(
            lambda t: np.exp(-(t * r) ** 2 / (2 * at(s, r) ** 2)),
            -np.pi, np.pi, epsabs=1e-14, epsrel=1e-13)[0]
    harmonic = traits.RPTraitNWHarmonic(2)
    np.testing.assert_allclose(
        harmonic.cloud_flux(dict(a=a, p=p), SUBRNODES),
        ring_integrals(
            lambda r: 4 * amplitude(r), zeros, lambda r: 4 * amplitude(r)),
        rtol=1e-8)
    distortion = traits.RPTraitNWDistortion()
    np.testing.assert_allclose(
        distortion.cloud_flux(dict(a=a, p=p, s=s), SUBRNODES),
        ring_integrals(distortion_mean, zeros, distortion_mean), rtol=1e-8)
