"""
Tests for the making of observations: their observables from data, and
the checks of their options.
"""

import astropy.units as u
import numpy as np
import pytest

from gbkfit.config import ConfigError, config_path
from gbkfit.dataset import Data, DatasetPixelMoments
from gbkfit.grid import SpectralAxis
from gbkfit.instrument import Instrument, LSFGauss
from gbkfit.observation import (
    LikelihoodGauss, LineOfSightFlat, ObservationSky, PixelBrightness,
    PixelMoments, UpperLimit, observable_parser, observation_parser)


def test_observable_from_data():
    # The data give the observable its grid and the orders of its
    # moments, and its spectral axis covers their velocities
    data = DatasetPixelMoments(
        {0: Data(np.ones((8, 20))), 1: Data(np.full((8, 20), 1500.0))},
        step=2, rota=30)
    observable = PixelMoments.from_data(
        data, spectral=SpectralAxis(step=5, convention='relativistic'))
    assert observable.grid() == data.grid()
    assert observable.orders() == (0, 1)
    assert observable.spectral().rval() == 1500
    ObservationSky(
        observable, LineOfSightFlat(), data=data,
        likelihood=LikelihoodGauss(missing_residual=3))


@pytest.mark.parametrize('dataset_class, observable_class, shape', [
    ('DatasetPixelSpectra', 'PixelSpectra', (10, 4, 4)),
    ('DatasetSlitSpectra', 'SlitSpectra', (10, 4))])
def test_spectra_keep_the_convention_of_their_data(
        dataset_class, observable_class, shape):
    # Spectra on an axis of log velocities (WAVE-LOG) make an observable of
    # their grid, its convention included
    import gbkfit.dataset
    import gbkfit.observation
    data = getattr(gbkfit.dataset, dataset_class)(
        Data(np.zeros(shape)),
        spectral=SpectralAxis(convention='log', rest='6564.61 Angstrom'))
    observable = getattr(gbkfit.observation, observable_class).from_data(data)
    assert observable.grid() == data.grid()
    assert observable.grid().coords.convention == 'log'


def test_a_model_that_can_have_missing_values_needs_their_residual():
    # A velocity map: the model has no velocity where its flux is below
    # the mask cutoff, and the residual there is the user's choice
    data = DatasetPixelMoments(
        {0: Data(np.ones((8, 20))), 1: Data(np.full((8, 20), 1500.0))})
    observable = PixelMoments.from_data(
        data, spectral=SpectralAxis(step=5, convention='relativistic'))
    with pytest.raises(ValueError, match="option missing_residual"):
        ObservationSky(observable, LineOfSightFlat(), data=data)


def test_observable_options_are_checked():
    with pytest.raises(ValueError, match="'flux_floor' must be at least 0 and less than 1"):
        PixelMoments(
            size=(8, 8), flux_floor=1,
            spectral=SpectralAxis(convention='relativistic'))
    with pytest.raises(ValueError, match="no spectral axis for an LSF"):
        ObservationSky(
            PixelBrightness(size=(8, 8)), LineOfSightFlat(),
            instrument=Instrument(lsf=LSFGauss(10 * u.km / u.s)))


def test_grid_options_take_one_value_or_one_for_each_axis():
    # (audit 2026-10-31 #52) In Python as in configurations: a value for
    # both axes on the sky, or one for each; more (e.g. for the channels
    # of a cube) are an error that names the option
    observable = PixelBrightness(size=8, step=0.5)
    assert observable.size() == (8, 8)
    assert observable.step() == (0.5, 0.5)
    with pytest.raises(ValueError, match="option 'size' must have one"):
        PixelBrightness(size=(8, 8, 10))
    with pytest.raises(ValueError, match="option 'size' must have one"):
        observable_parser.load(dict(type='pixel_brightness', size=[8, 8, 10]))


def test_observations_need_their_type():
    info = dict(
        line_of_sight=dict(type='flat'),
        observable=dict(type='pixel_brightness', size=[4, 4]))
    with pytest.raises(ConfigError, match="missing required options: 'type'.*'sky'"):
        observation_parser.load(info)
    observation = observation_parser.load(info | dict(type='sky'))
    assert observation_parser.dump(observation)['type'] == 'sky'


def test_instrument_warnings_have_the_path_of_the_configuration(caplog):
    # The instrument is checked when the observation is made, so its
    # warnings have the path of the observation
    with config_path('observations'):
        observation_parser.load([dict(
            type='sky',
            line_of_sight=dict(type='flat'),
            observable=dict(
                type='pixel_spectra', size=[4, 4],
                spectral=dict(size=4, convention='relativistic'),
                smooth_weights=True))])
    assert "observations[0] [sky]: smooth_weights is true" in caplog.text


@pytest.mark.parametrize('dtype', ['float16', 'int32', 'complex'])
def test_dtypes_without_kernels_are_errors(dtype):
    # (they failed later, in the native code)
    dataset = Data(np.ones((8, 8)))
    from gbkfit.dataset import DatasetPixelBrightness
    observable = PixelBrightness.from_data(DatasetPixelBrightness(dataset))
    with pytest.raises(ValueError, match="'dtype' must be one of 'float32', 'float64'"):
        ObservationSky(observable, LineOfSightFlat(), dtype=dtype)


def test_ffts_are_float64_by_default_and_not_less_precise_than_models():
    observable = PixelBrightness(size=(8, 8))
    assert ObservationSky(
        observable, LineOfSightFlat()).fft_dtype() == np.float64
    with pytest.raises(ValueError, match=r"fft_dtype \(float32\) is less"):
        ObservationSky(
            observable, LineOfSightFlat(
                ), dtype='float64', fft_dtype='float32')


def brightness(unit=None, spatial_correlation=None):
    """An image of 8 x 8 pixels of 1 and errors 1, in the given unit."""
    from gbkfit.dataset import DatasetPixelBrightness
    return DatasetPixelBrightness(Data(
        np.ones((8, 8)), error=np.ones((8, 8)), unit=unit,
        spatial_correlation=spatial_correlation))


def test_the_data_choose_the_dtype_unless_it_is_given():
    # float64 for data of correlated noise, float32 otherwise: a copy of
    # other data chooses again, and dumps keep only a given dtype (the
    # copy kept the first choice, and dumps fixed it: audit 2026-11-11)
    from gbkfit.instrument import PSFGauss
    correlated = brightness(spatial_correlation=PSFGauss(1.0))
    observable = PixelBrightness.from_data(correlated)
    chosen = ObservationSky(observable, LineOfSightFlat())
    assert chosen.dtype() == np.float32
    assert 'dtype' not in chosen.dump()
    assert chosen.with_data(correlated).dtype() == np.float64
    given = ObservationSky(observable, LineOfSightFlat(), dtype='float32')
    assert given.with_data(correlated).dtype() == np.float32
    assert given.dump()['dtype'] == 'float32'


def test_a_group_with_other_data_has_the_units_they_give():
    # The parameters of a fitted noise are in the units of the data (see
    # Likelihood.param_units): a group with data of another unit has its
    # parameters in it (it kept the old: audit 2026-11-11)
    from gbkfit.model import galaxy_parser
    from gbkfit.observation import FittedNoise, ObservationGroup
    galaxy = galaxy_parser.load(dict(components=[dict(
        type='thin_disk', rnodes=[0, 1, 2], rptraits=dict(type='exponential'),
        emission=dict(type='band'))]))
    data = brightness('Jy')
    observation = ObservationSky(
        PixelBrightness.from_data(data), LineOfSightFlat(), data=data,
        likelihood=LikelihoodGauss(fitted_noise=dict(
            brightness=FittedNoise(jitter=True))), name='obs')
    group = ObservationGroup([galaxy], [observation])
    jitter = 'obs_noise_jitter_brightness'
    assert group.pdescs()[jitter].unit() == u.Jy
    assert group.with_data([brightness('mJy')]).pdescs()[jitter].unit() \
        == u.mJy


def test_a_group_with_other_data_is_a_copy_of_the_same_parameters():
    # (the group and its plan ask one question, see same_params: a copy
    # shares what the group made of each observation; a group of other
    # parameters is made again)
    from gbkfit.model import galaxy_parser
    from gbkfit.observation import FittedNoise, ObservationGroup
    galaxy = galaxy_parser.load(dict(components=[dict(
        type='thin_disk', rnodes=[0, 1, 2], rptraits=dict(type='exponential'),
        emission=dict(type='band'))]))
    data = brightness('Jy')
    plain = ObservationSky(
        PixelBrightness.from_data(data), LineOfSightFlat(), data=data,
        name='obs')
    fitted = plain.with_data(data, LikelihoodGauss(fitted_noise=dict(
        brightness=FittedNoise(jitter=True))))
    group = ObservationGroup([galaxy], [plain])
    copied = group.with_data([brightness('Jy')])
    assert copied.same_params(group)
    assert copied.seen_by(0) is group.seen_by(0)
    assert not ObservationGroup([galaxy], [fitted]).same_params(group)


@pytest.mark.parametrize('options, message', [
    (dict(weights=dict(moment2=2)), r"does not have: \['moment2'\]"),
    (dict(upper_limits=dict(momnet0=UpperLimit(limit=1, error=0.1))),
     r"does not have: \['momnet0'\]")])
def test_likelihoods_name_items_of_their_observation(options, message):
    # A typo in the name of an item of the options of a likelihood would
    # leave the option out without a word
    data = DatasetPixelMoments(
        {0: Data(np.ones((8, 20))), 1: Data(np.full((8, 20), 1500.0))})
    with pytest.raises(ValueError, match=message):
        ObservationSky(
            PixelMoments.from_data(
                data, spectral=SpectralAxis(step=5, convention='relativistic')),
            LineOfSightFlat(), data=data,
            likelihood=LikelihoodGauss(missing_residual=3, **options))


def test_residuals_of_upper_limits_are_not_linear_in_the_model():
    # (so that no elements of the model are solved from them)
    from gbkfit.dataset import DatasetPixelSpectra
    from gbkfit.driver.drivers.host import DriverHost
    from gbkfit.objective import comparison_of
    from gbkfit.observation import PixelSpectra
    data = DatasetPixelSpectra(
        Data(np.ones((10, 4, 4))),
        spectral=SpectralAxis(convention='relativistic'))
    observation = ObservationSky(
        PixelSpectra.from_data(data), LineOfSightFlat(), data=data,
        likelihood=LikelihoodGauss(
            upper_limits=dict(spectra=UpperLimit(limit=0.1, error=0.05))))
    comparison_ = comparison_of(observation, DriverHost())
    for form in (comparison_.linear_form, comparison_.pointwise_form):
        assert "has an upper limit" in form(['spectra'])


def test_the_psf_must_be_the_beam_of_data_of_a_beam(caplog):
    # Data of a beam (of a radio interferometer) are smoothed by it, a
    # Gaussian: a PSF of another shape or size is warned of; a beam for
    # each channel of a cube is that of the PSF at each, or one of them
    from gbkfit.dataset import DatasetPixelSpectra
    from gbkfit.instrument import PSFGauss, PSFMoffat, SpectralTable
    from gbkfit.io import Beam
    from gbkfit.observation import PixelSpectra
    hi = 1420.405752 * u.MHz
    spectral = SpectralAxis(size=3, step=10, rval=0, rest=hi)

    def warnings(psf, beam):
        caplog.clear()
        data = DatasetPixelSpectra(
            Data(np.zeros((3, 4, 4)), unit='Jy/beam', beam=beam),
            spectral=SpectralAxis(step=10, rval=0, rest=hi))
        ObservationSky(
            PixelSpectra(size=(4, 4), spectral=spectral), LineOfSightFlat(),
            instrument=Instrument(psf=psf), data=data)
        return caplog.text
    beam = Beam(6.0, 3.0, 30.0)
    assert warnings(PSFGauss.from_beam(6.0, 3.0, 30.0), beam) == ""
    assert warnings(PSFGauss.from_fwhm(6.0, 0.5, 210.0), beam) == ""
    assert "the PSF of the instrument is the beam 6 x 3 arcsec at 40" \
        in warnings(PSFGauss.from_beam(6.0, 3.0, 40.0), beam)
    assert "of type 'moffat', not a Gaussian" in warnings(
        PSFMoffat.from_fwhm(6.0, 2.5, 0.5, 30.0), beam)
    assert "is none (a point)" in warnings(None, beam)
    # (of a cube of CASA: a beam at the frequency of each channel)
    frequencies = [1420.4, 1420.36, 1420.31] * u.MHz
    beams = Beam(
        np.array([6.0, 6.1, 6.2]), np.full(3, 3.0), np.full(3, 30.0),
        frequencies)
    varying = PSFGauss.varying(
        bmaj=SpectralTable(frequencies, [6.0, 6.1, 6.2]), bmin=3.0,
        bpa=30.0)
    assert warnings(varying, beams) == ""
    text = warnings(PSFGauss.from_beam(6.0, 3.0, 30.0), beams)
    assert "another at 2 of their 3 channels: e.g. at 1420.36 MHz" in text
    assert "have one beam, 6 x 3 arcsec at 30 deg, but the PSF of the " \
        "instrument varies" in warnings(varying, beam)
