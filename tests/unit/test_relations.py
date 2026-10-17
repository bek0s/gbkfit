"""
Tests for the scaling relations that expressions can call: each at a
point where its value follows from its formula, and from expressions.
"""

import numpy as np
import pytest

from gbkfit import relations
from gbkfit.params import ParamScalarDesc, ParamSpace


def test_dekel_zhao_parameters_convert_both_ways():
    # s1 is the slope (a + 3.5 x^(1/2)) / (1 + x^(1/2)) at x = 0.01 c, and
    # c2 = c (1.5 / (2 - a))^2 (see mass.ProfileDekelZhao)
    a, c = np.array([0.0, 0.5, 1.2]), np.array([8.0, 15.0, 25.0])
    s1 = (a + 3.5 * np.sqrt(0.01 * c)) / (1 + np.sqrt(0.01 * c))
    c2 = c * (1.5 / (2 - a)) ** 2
    np.testing.assert_allclose(relations.dz_a(s1, c2), a, atol=1e-12)
    np.testing.assert_allclose(relations.dz_c(s1, c2), c, rtol=1e-12)


def test_dutton14_is_its_intercept_at_its_pivot():
    # At M = 10^12 h^-1 Msun, log10 c = a(z)
    for z in (0, 1.5):
        log_m = 12 - np.log10(0.671)
        assert np.log10(relations.cvir_dutton14(log_m, z)) == pytest.approx(
            0.537 + (1.025 - 0.537) * np.exp(-0.718 * z ** 1.08))
        assert np.log10(relations.c200_dutton14(log_m, z)) == pytest.approx(
            0.520 + (0.905 - 0.520) * np.exp(-0.617 * z ** 1.21))


def test_dekel_zhao_halos_of_few_stars_are_dark_matter_only():
    # At M_star / M_vir = 10^-6, s1 and c2 are those of the NFW halo of
    # the dark-matter-only concentration (to the precision of the fits)
    c_dmo = relations.cvir_dutton14(11, 0)
    s_dmo = (1 + 0.03 * c_dmo) / (1 + 0.01 * c_dmo)
    assert relations.dz_s1(5, 11) == pytest.approx(s_dmo, rel=1e-4)
    assert relations.dz_c2(5, 11) == pytest.approx(c_dmo, rel=1e-2)


def test_halo_masses_invert_their_stellar_mass_relations():
    # At M = M1: m / M = N (Moster et al. 2013), and log10 m = log10(eps
    # M1) (Behroozi et al. 2013); and the inverse of each relation
    from gbkfit.relations import _log_mstar_behroozi13, _log_mstar_moster13
    z = np.array([0.0, 1.5])
    w = z / (z + 1)
    log_m1 = 11.590 + 1.195 * w
    log_mstar = log_m1 + np.log10(0.0351 - 0.0247 * w)
    np.testing.assert_allclose(
        relations.log_mhalo_moster13(log_mstar, z), log_m1, atol=1e-10)
    log_mstar = _log_mstar_behroozi13(np.array([11.5, 11.5]), z)
    np.testing.assert_allclose(
        relations.log_mhalo_behroozi13(log_mstar, z), 11.5, atol=1e-10)
    log_mhalo = np.linspace(10, 14, 9)
    np.testing.assert_allclose(relations.log_mhalo_moster13(
        _log_mstar_moster13(log_mhalo, np.full(9, 2.0)), 2.0), log_mhalo,
        atol=1e-10)


def test_expressions_call_the_relations():
    params = ParamSpace(
        dict(s1=ParamScalarDesc(), c2=ParamScalarDesc(),
             c=ParamScalarDesc()),
        dict(s1=1.0, c2=12.0, c='relations.dz_c(s1, c2)'))
    assert params.evaluate()['c'] == pytest.approx(
        float(relations.dz_c(1.0, 12.0)))
    with pytest.raises(Exception, match="cannot be called|not allowed"):
        ParamSpace(
            dict(c=ParamScalarDesc()), dict(c='relations._dutton14(1)'))
