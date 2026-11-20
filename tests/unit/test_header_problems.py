"""
Tests for the problems of headers of files of data: the steps and
references they leave out (see reference_errors), every independent
problem found together (see describe_fits), short messages of the
failures of wcslib and astropy, and the unit of the values, stated or of
the header (see data_unit).
"""

import glob
import os
import re

import numpy as np
import pytest
from astropy.io import fits

from gbkfit.config import ConfigProblem, InputError
from gbkfit.dataset import Data, dataset_parser
from gbkfit.io import describe_fits, read_arrays, read_oriented
from gbkfit.prep.arrays import FitsArrays


CRAFTED = os.path.expanduser(
    '~/work/dev/gbkfit/scratch/overhaul/census/crafted')

SKY = dict(
    CTYPE1='RA---SIN', CRVAL1=150.0, CDELT1=-1 / 3600, CRPIX1=4.0,
    CUNIT1='deg', CTYPE2='DEC--SIN', CRVAL2=2.0, CDELT2=1 / 3600,
    CRPIX2=3.0, CUNIT2='deg')
SPECTRAL = dict(
    CTYPE3='VRAD', CRVAL3=1000.0, CDELT3=10.0, CRPIX3=1.0, CUNIT3='km/s')


def write(path, header, shape=(5, 6, 7)):
    fits.writeto(path, np.ones(shape, np.float32), fits.Header(header))
    return str(path)


def problems(path, **options):
    """The problems of the first HDU of a file, as (path, message)."""
    return [(p.path, p.message) for p in describe_fits(path, **options)[0]
            .problems if p.severity == 'error']


def without(header, *keys):
    return {key: value for key, value in header.items() if key not in keys}


def raised(path, call):
    """The errors that a call raises, alone or together, as problems."""
    with pytest.raises(Exception) as e:
        call()
    errors = getattr(e.value, 'exceptions', [e.value])
    assert all(isinstance(error, InputError) for error in errors)
    return [(error.config_path, str(error).removeprefix(f"{path}: "))
            for error in errors]


@pytest.mark.parametrize('left_out, expected', [
    (('CDELT1',), [(['step'], 'CDELT1 missing')]),
    (('CRPIX2',), [(['rpix'], 'CRPIX2 missing')]),
    (('CRVAL1', 'CRVAL2'), [
        (['rval'], 'CRVAL1 missing'), (['rval'], 'CRVAL2 missing')]),
    (('CDELT3',), [(['spectral', 'step'], 'CDELT3 missing')]),
    (('CRPIX3',), [(['spectral', 'rpix'], 'CRPIX3 missing')]),
    (('CRVAL3',), [(['spectral', 'rval'], 'CRVAL3 missing')])])
def test_a_step_or_reference_left_out_is_an_error_at_its_option(
        tmp_path, left_out, expected):
    path = write(tmp_path / 'cube.fits', without(SKY | SPECTRAL, *left_out))
    assert problems(path) == expected
    assert describe_fits(path)[0].coords is None
    # (read as describe says it)
    assert raised(path, lambda: Data.from_files(path, spectral_axis=2)) \
        == expected


def test_a_step_stated_is_no_error(tmp_path):
    path = write(tmp_path / 'cube.fits', without(SKY | SPECTRAL, 'CDELT1'))
    assert problems(path, stated={'step'}) == []
    path = write(tmp_path / 'other.fits', without(SKY | SPECTRAL, 'CDELT3'))
    assert problems(path, stated={'step'}) == [
        (['spectral', 'step'], 'CDELT3 missing')]
    assert problems(path, stated={'spectral.step'}) == []


def test_a_reference_stated_alone_needs_the_other_of_a_header_without_one(
        tmp_path):
    # (the one stated alone takes the other from the header, which cannot
    # without both)
    path = write(tmp_path / 'cube.fits', without(SKY | SPECTRAL, 'CRPIX1'))
    assert problems(path, stated={'rpix'}) == [(['rval'], 'CRPIX1 missing')]
    assert problems(path, stated={'rval'}) == [(['rpix'], 'CRPIX1 missing')]
    assert problems(path, stated={'rpix', 'rval'}) == []
    assert describe_fits(path, stated={'rpix', 'rval'})[0].coords is not None


def test_a_dataset_states_its_steps_and_references(tmp_path):
    path = write(tmp_path / 'cube.fits', without(
        SKY | SPECTRAL, 'CRPIX1', 'CRPIX2', 'CDELT3'))
    info = dict(type='pixel_spectra', data=path)
    assert raised(path, lambda: dataset_parser.load(info)) == [
        (['rpix'], 'CRPIX1 missing'), (['rpix'], 'CRPIX2 missing'),
        (['spectral', 'step'], 'CDELT3 missing')]
    dataset = dataset_parser.load(info | dict(
        rpix=[3, 2], rval=[150.0, 2.0], spectral=dict(step=10.0)))
    assert dataset.grid().coords.rpix[:2] == (3, 2)
    assert dataset.grid().coords.step[2] == 10.0


def test_a_cd_matrix_with_cdelt_is_an_error_unless_they_agree(tmp_path):
    # (CDELTn of 1, or of CDn_n, agree with the matrix, as CALIFA writes
    # them; another is read by some tools, and left out by FITS)
    cd = dict(CD1_1=-1 / 3600, CD2_2=1 / 3600, CD3_3=10.0)
    agreed = without(SKY | SPECTRAL, 'CDELT1', 'CDELT2', 'CDELT3') | cd \
        | dict(CDELT1=1.0, CDELT2=1.0, CDELT3=10.0)
    assert problems(write(tmp_path / 'califa.fits', agreed)) == []
    mixed = agreed | dict(CDELT3=4000.0)
    assert problems(write(tmp_path / 'mixed.fits', mixed)) == [
        (['spectral', 'step'], 'CDELT3 4000 with CD3_3 10')]
    missing = without(agreed, 'CD3_3', 'CDELT3')
    assert problems(write(tmp_path / 'missing.fits', missing)) == [
        (['spectral', 'step'], 'CD3_3 missing')]


def test_axes_of_length_1_are_not_looked_at(tmp_path):
    # (e.g. the spectral axis of one pixel left in a moment map)
    path = write(
        tmp_path / 'map.fits',
        without(SKY | SPECTRAL, 'CDELT3', 'CRVAL3', 'CRPIX3'), (1, 6, 7))
    assert problems(path) == []


def test_every_independent_problem_is_found_together(tmp_path):
    # Velocities of no convention (VELO-HEL without VELREF), a rest that
    # is not one, a skewed grid on the sky, and a BUNIT that is not a unit:
    # the convention does not hide the others
    header = SKY | SPECTRAL | dict(
        CTYPE3='VELO-HEL', PC1_2=0.5, BUNIT='JY/BEAM')
    path = write(tmp_path / 'cube.fits', header)
    found = problems(path, rest='6562.8')
    assert (['spectral', 'convention'], 'VELREF missing') in found
    assert (['spectral', 'rest'], "rest '6562.8' invalid") in found
    assert (['unit'], "BUNIT 'JY/BEAM' invalid") in found
    assert any('skewed' in message for _, message in found)


@pytest.mark.skipif(not os.path.isdir(CRAFTED), reason="no crafted files")
@pytest.mark.parametrize('name, expected', [
    ('lv-ctype1-badproj.fits', "CTYPE1 'RA---XXX' invalid"),
    ('lv-ctype-mixed-pair.fits', "CTYPE2 'GLAT-NCP' mismatched"),
    ('lv-ctype1-missing.fits', "CTYPE2 'DEC--NCP' unpaired"),
    ('lv-cunit1-degrees.fits', "CUNIT1 'DEGREES' invalid"),
    ('lv-cdelt1-zero.fits', 'CDELT/PC/CD matrix singular')])
def test_the_convention_does_not_hide_the_sky_of_crafted_files(
        name, expected):
    found = problems(os.path.join(CRAFTED, name))
    assert (['spectral', 'convention'], 'VELREF missing') in found
    assert any(message.startswith(expected) for _, message in found)


# What no message says: where wcslib failed, or an absolute path
_RAW = re.compile(r'\.c:|line \d|ERROR \d|wcsset|/home')


@pytest.mark.skipif(not os.path.isdir(CRAFTED), reason="no crafted files")
def test_no_message_is_raw_text_of_wcslib_or_astropy():
    messages = []
    for path in sorted(glob.glob(os.path.join(CRAFTED, '*'))):
        for options in ({}, dict(velocity='radio')):
            try:
                messages += [
                    problem.message for hdu in describe_fits(path, **options)
                    for problem in hdu.problems]
            except OSError as e:
                messages.append(str(e).removeprefix(f"{path}: "))
    assert messages and not [m for m in messages if _RAW.search(m)]
    assert 'not FITS' in messages and 'empty or truncated' in messages


@pytest.mark.skipif(not os.path.isdir(CRAFTED), reason="no crafted files")
def test_the_data_of_a_truncated_file_are_an_error_of_the_file():
    with pytest.raises(OSError, match="truncated$"):
        read_oriented(os.path.join(CRAFTED, 'lv-truncated.fits'))


def test_a_unit_that_is_not_of_fits_is_refused_alike_everywhere(tmp_path):
    # (no legacy spelling is read: 'JY/BEAM' is not Jy/beam)
    header = SKY | SPECTRAL | dict(BUNIT='JY/BEAM')
    path = write(tmp_path / 'cube.fits', header)
    message = "BUNIT 'JY/BEAM' invalid"
    assert problems(path) == [(['unit'], message)]
    with pytest.raises(InputError, match=f"{message}$") as e:
        read_arrays(path)
    assert e.value.config_path == ['unit']


def test_a_stated_unit_replaces_bunit(tmp_path):
    header = SKY | SPECTRAL | dict(
        BUNIT='JY/BEAM', BMAJ=3 / 3600, BMIN=2 / 3600, BPA=0.0)
    path = write(tmp_path / 'cube.fits', header)
    assert problems(path, unit='Jy/beam') == []
    assert problems(path, unit='Jy/bm') == [(['unit'], "unit 'Jy/bm' invalid")]
    assert str(read_arrays(path, unit='Jy/beam').unit) == 'Jy / beam'
    with pytest.raises(InputError, match="^unit 'Jy/bm' invalid$"):
        read_arrays(path, unit='Jy/bm')
    with pytest.raises(InputError, match="^unit 'Jy/bm' invalid$"):
        read_oriented(path, unit='Jy/bm')
    assert read_oriented(path, unit='Jy/beam')[0].shape == (5, 6, 7)
    # (the beam of a unit per beam stated)
    path = write(tmp_path / 'nobeam.fits', SKY | SPECTRAL)
    assert describe_fits(path, unit='Jy/beam')[0].problems == (ConfigProblem(
        'error', ['beam'], 'BMAJ, BMIN missing (unit Jy/beam)'),)


def test_prep_reads_at_its_options(tmp_path):
    # (the options of a dataset are those of prep: velocity, not
    # spectral.convention)
    header = SKY | SPECTRAL | dict(CTYPE3='VELO-HEL', BUNIT='JY/BEAM')
    path = write(tmp_path / 'cube.fits', header)
    with pytest.raises(ExceptionGroup) as e:
        FitsArrays.read(path)
    assert [error.config_path for error in e.value.exceptions] == [
        ['velocity'], ['unit']]
    arrays = FitsArrays.read(path, velocity='radio', unit='Jy/beam')
    assert str(arrays.unit) == 'Jy / beam'
