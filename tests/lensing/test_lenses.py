"""
Tests of lenses (gbkfit.lensing.Lens): their pieces, parameters and
source redshifts, and their deflection at D_ls / D_s = 1.
"""

import copy

import numpy as np
import pytest
from astropy.cosmology import Planck18

from gbkfit.cosmology import DEFAULT
from gbkfit.driver.drivers.host import DriverHost
from gbkfit.lensing import (
    Lens, LensDeflectionMap, LensSIE, LensShear, SkyPoints, lens_parser)
from gbkfit.params import ModelEvaluationError


POINTS = SkyPoints(
    np.array([[0.5, -1.2], [2.0, 0.3]]), np.array([[1.0, 0.4], [-0.7, 0.0]]),
    (150.0, 2.0))

SIE = dict(theta_e=1.3, q=0.7, posa=20, xpos=0.1, ypos=-0.2)

# (the driver of the deflections: the host)
HOST = DriverHost()


def test_the_parameters_of_a_lens():
    # Several pieces: prefixed by their names; one: not
    lens = Lens('lens', 0.35, [LensSIE(name='main'), LensShear(name='shear')],
                source_redshift=2.1)
    assert list(lens.pdescs()) == [
        'main_theta_e', 'main_q', 'main_posa', 'main_xpos', 'main_ypos',
        'shear_gamma', 'shear_posa']
    alone = Lens('lens', 0.35, LensSIE(name='main'), source_redshift=2.1)
    assert list(alone.pdescs()) == ['theta_e', 'q', 'posa', 'xpos', 'ypos']
    # (named after their types by default)
    unnamed = Lens('lens', 0.35, [LensSIE(), LensShear()], source_redshift=2.1)
    assert list(unnamed.pdescs()) == [
        'sie_theta_e', 'sie_q', 'sie_posa', 'sie_xpos', 'sie_ypos',
        'shear_gamma', 'shear_posa']
    assert 'name' not in lens_parser.dump(unnamed)['pieces'][0]
    with pytest.raises(ValueError, match="repeat names"):
        Lens('lens', 0.35, [LensSIE(name='a'), LensShear(name='a')],
             source_redshift=2.1)
    with pytest.raises(ValueError, match="named after its type"):
        Lens('lens', 0.35, [LensSIE(), LensSIE()], source_redshift=2.1)
    with pytest.raises(ValueError, match="at least one piece"):
        Lens('lens', 0.35, [])
    with pytest.raises(ValueError):
        Lens('a lens', 0.35, LensSIE(), source_redshift=2.1)


def test_the_source_redshift_is_that_of_quoted_pieces():
    with pytest.raises(ValueError, match="source_redshift"):
        Lens('lens', 0.35, LensSIE())
    deflection = LensDeflectionMap(np.zeros((3, 3)), np.zeros((3, 3)))
    Lens('cluster', 0.35, deflection)
    with pytest.raises(ValueError, match="says nothing"):
        Lens('cluster', 0.35, deflection, source_redshift=2.1)
    # Maps made for one source, as an SIE is quoted
    for_one = LensDeflectionMap(
        np.zeros((3, 3)), np.zeros((3, 3)), quoted=True)
    Lens('cluster', 0.35, for_one, source_redshift=2.1)
    with pytest.raises(ValueError, match="deflection_map"):
        Lens('cluster', 0.35, for_one)
    # (behind the plane of the lens, not within it)
    with pytest.raises(ValueError, match="behind its plane"):
        Lens('lens', 0.35, LensSIE(), source_redshift=0.355)


def test_the_quoted_factor_is_d_s_over_d_ls_of_the_source():
    lens = Lens('lens', 0.35, LensSIE(), source_redshift=2.1)
    d_s = Planck18.angular_diameter_distance(2.1).value
    d_ls = Planck18.angular_diameter_distance(0.35, 2.1).value
    assert lens.quoted_factor(0.35, DEFAULT) == pytest.approx(d_s / d_ls)
    # (of the distance of its plane)
    d_ls = Planck18.angular_diameter_distance(0.36, 2.1).value
    assert lens.quoted_factor(0.36, DEFAULT) == pytest.approx(d_s / d_ls)
    with pytest.raises(ValueError, match="not behind"):
        lens.quoted_factor(2.2, DEFAULT)
    deflection = LensDeflectionMap(np.zeros((3, 3)), np.zeros((3, 3)))
    assert Lens('map', 0.35, deflection).quoted_factor(0.35, DEFAULT) == 1


def test_a_lens_converts_its_quoted_pieces_only():
    # alpha-hat: the quoted pieces times the factor, the maps as they are
    deflection = LensDeflectionMap(
        np.full((5, 6), 0.4), np.full((5, 6), -0.1), step=1,
        rval=POINTS.origin, name='map')
    lens = Lens('lens', 0.35, [LensSIE(name='main'), deflection],
                source_redshift=2.1)
    params = {f'main_{key}': value for key, value in SIE.items()}
    sie = np.array(LensSIE().deflection(POINTS, SIE, HOST))
    alpha = np.array(lens.deflection(POINTS, params, 1.7, HOST))
    np.testing.assert_allclose(
        alpha, 1.7 * sie + np.array([0.4, -0.1])[:, None, None])


def test_a_lens_deflects_nowhere_off_its_maps():
    deflection = LensDeflectionMap(
        np.zeros((3, 3)), np.zeros((3, 3)), step=0.1, rval=POINTS.origin)
    with pytest.raises(ModelEvaluationError, match="do not cover"):
        Lens('map', 0.35, deflection).deflection(POINTS, {}, 1, HOST)


def test_lenses_from_configurations(tmp_path):
    info = dict(
        name='a370', redshift=0.375, source_redshift=1.06, pieces=[
            dict(type='epl', name='main', multipoles=[4]),
            dict(type='nfw', name='halo'),
            dict(type='sheet', name='sheet'),
            dict(type='shear', name='shear')])
    lens = lens_parser.load(copy.deepcopy(info))
    assert [piece.type() for piece in lens.pieces()] == [
        'epl', 'nfw', 'sheet', 'shear']
    assert lens_parser.dump(lens) == info
    assert 'main_a4' in lens.pdescs()
    # Maps are dumped to files named after the lens (and the piece, of
    # several)
    maps = Lens('cluster', 0.4, LensDeflectionMap(
        np.ones((4, 4)), np.ones((4, 4)), step=0.5, rval=(150.0, 2.0),
        name='map'))
    dumped = lens_parser.dump(maps, prefix=str(tmp_path / 'd_'))
    assert dumped['pieces'][0]['alpha_x'] == str(
        tmp_path / 'd_cluster_alpha_x.fits')
    again = lens_parser.load(copy.deepcopy(dumped))
    centre = SkyPoints(np.zeros(1), np.zeros(1), (150.0, 2.0))
    np.testing.assert_allclose(
        again.deflection(centre, {}, 1, HOST), np.ones((2, 1)))
