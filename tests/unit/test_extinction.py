"""
Tests for the extinction laws of dust.
"""

import pytest

from gbkfit.model import (
    ExtinctionCCM, ExtinctionCalzetti, ExtinctionTable,
    extinction_law_parser)
from gbkfit.utils.parseutils import ConfigError


@pytest.mark.parametrize('wavelength, ratio', [
    # Those of the package dust_extinction (CCM89, Rv = 3.1)
    (0.15, 2.663879), (0.2175, 3.185101), (0.4862, 1.163952),
    (0.55, 0.998850), (0.6564, 0.817592), (1.25, 0.282070),
    (2.1661, 0.116396)])
def test_ccm(wavelength, ratio):
    assert ExtinctionCCM(3.1).ratio(wavelength) == pytest.approx(
        ratio, abs=1e-6)


def test_calzetti():
    # k(Halpha) = 3.33 and k(Hbeta) = 4.60 for R_V = 4.05
    law = ExtinctionCalzetti()
    assert law.ratio(0.6563) * 4.05 == pytest.approx(3.33, abs=0.005)
    assert law.ratio(0.4861) * 4.05 == pytest.approx(4.60, abs=0.005)


def test_table_interpolates_linearly():
    law = ExtinctionTable([0.5, 0.6, 0.8], [1.2, 1.0, 0.6])
    assert law.ratio(0.55) == pytest.approx(1.1)
    assert law.ratio(0.7) == pytest.approx(0.8)


@pytest.mark.parametrize('law, wavelength', [
    (ExtinctionCCM(), 0.05), (ExtinctionCCM(), 5),
    (ExtinctionCalzetti(), 0.1), (ExtinctionCalzetti(), 2.5),
    (ExtinctionTable([0.5, 0.6], [1, 1]), 0.7)])
def test_laws_raise_outside_their_range(law, wavelength):
    with pytest.raises(ConfigError, match="is defined from"):
        law.ratio(wavelength)


def test_laws_of_the_same_options_are_equal():
    assert ExtinctionCCM(3.1) == extinction_law_parser.load(
        dict(type='ccm', rv=3.1))
    assert ExtinctionCCM(3.1) != ExtinctionCCM(5)
    assert len({ExtinctionCCM(), ExtinctionCCM(), ExtinctionCalzetti()}) == 2


def test_laws_of_equal_options_share_a_cube():
    # (3 and 3.0 are one option: the laws are one key of a dict)
    assert len(dict.fromkeys([ExtinctionCCM(3), ExtinctionCCM(3.0)])) == 1
