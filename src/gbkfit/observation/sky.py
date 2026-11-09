"""
The observation on the sky: galaxies seen as an observable, along a
line of sight, through a foreground and an instrument.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, Self, TypeAlias

import astropy.units
import numpy as np

import gbkfit.units
from gbkfit.config import (
    Choices, check_name, check_options, config_path,
    load_option_and_update_info, make_typed_desc, warn)
from gbkfit.dataset import Data, Dataset, dataset_parser
from gbkfit.driver import DeviceArray, Driver
from gbkfit.grid import GridData, SpectraData, VelocityConvention
from gbkfit.instrument import PSF, Instrument, PSFGauss, instrument_parser
from gbkfit.io import Beam
from gbkfit.model import (
    Band, BandEmission, DataUnit, Extras, GalaxyState, LightUnit,
    Scene, band_parser)
from gbkfit.model.components.lines import catalogue_in_use
from gbkfit.params import (
    ParamDesc, ParamScalarDesc, ParamValues, ParamVectorDesc)
from ._scene import SeenGalaxies, check_component_names, make_scene
from .foreground import Foreground, foreground_parser
from .likelihood import Likelihood, likelihood_parser
from .line_of_sight import LineOfSight, line_of_sight_parser
from .noise import Noise
from .observables import (
    FluxScale, HostModelData, ModelData, Observable, ObservablePlan,
    observable_parser)
from .observation import Observation
from .weights import Weights, weights_geometry


__all__ = [
    'ObservationSky'
]


_log = logging.getLogger(__name__)


# The floating types of models: those the native modules have kernels for
_DType: TypeAlias = Annotated[
    Literal[tuple(dt.name for dt in Driver.dtypes())], Choices(
        float32='Single precision (float32)',
        float64='Double precision (float64)')]


class ObservationSky(Observation):
    """
    An observation on the sky: galaxies seen as an observable through an
    instrument, optionally compared with its data. They are seen along a
    line of sight, through a foreground (e.g. a gravitational lens) and
    the instrument, and the model of the galaxies is compared with the
    data under a likelihood (see Observation).

    In its configuration (type 'sky'), the data have the form of the
    observable (no type), and give it the options of options_from_data
    (see Observable.from_data).

    Its model is in the unit of its data (see data_unit): the light of
    the components whose emission has a unit is turned into it (see
    Component.plan), and that of the others is in it. Without data, the
    model is in the unit of the components it sees, which must then have
    one (or none).

    Its model can have a flux scale: its items of light (see
    Observable.light_items) are times the values of the parameter
    '{name}_flux_scale', one value ('one'; e.g. the calibration between
    the instruments of a joint fit), one for each spectrum of its data
    ('per_spectrum'), or one for each line of the image of slitless
    spectra along the dispersion ('per_row', the row amplitudes of geko).
    Its values can be fixed, tied, free or solved (see ParamSpace): e.g.
    {solve: {type: gaussian, mean: 1, sigma: 0.5, min: 0}} (see
    SolveGaussian). Scales per spectrum make a fit take the shape of each
    spectrum from the model (where its lines are, how wide; with beam
    smearing, also how the light of the model weighs its velocities), and
    its brightness mostly from the data, which suits a model whose light
    misses structure of the data (e.g. the clumps of a clumpy disk);
    solved, each is worked out at each evaluation with exact formulas,
    never putting the flux of the data into the model.

    Why that prior for scales per spectrum (the user chose it,
    2026-10-15; revise here): the log-likelihood of samplers needs one
    (they integrate the solved values out). A flat prior on a scale makes
    the integral grow as the model gets fainter (half the light, twice
    the scales, twice the integral): samplers would prefer galaxies of no
    light. A flat prior on the flux of each spectrum (the scale times the
    sum of the model there) removes that, but favours model lines whose
    flux the data pin down loosely (broad lines) where the spectra are
    faint: in 4000 random spectra of 61 channels of 10 km/s (noise 1),
    the log-evidence of a 60 km/s model line over a 30 km/s one was
    +0.35 on noise alone and +0.22 on a faint line truly 30 km/s wide
    (flux 3); with this prior (sigma 0.5), +0.03 and -0.09. Over
    thousands of faint spectra (e.g. of clumpy disks at z ~ 1.5) the flat
    prior would pull the dispersion up. This prior is proper (the evidence
    is a normal one), and keeps the brightness of the model meaningful:
    it is the mean brightness, about which the spectra scatter. Its cost
    is sigma, a statement of how much the spectra may differ from the
    model: a very large sigma is a flat prior on the scales again, with
    its bias.

    Parameters
    ----------
    observable : Observable
        What the data measure.
    line_of_sight : LineOfSight
        How it treats the line of sight: LineOfSightFlat() (the plane of
        the sky) or LineOfSightGrid() (a depth axis).
    foreground : Foreground, optional
        What the light meets before the telescope; nothing by default.
    instrument : Instrument, optional
        The telescope and the instrument; a perfect one by default.
    galaxies : Sequence of str, optional
        The names of the galaxies it sees; required only when there are
        several (see ObservationGroup).
    components : Sequence of str, optional
        The names of the components whose light it sees ('galaxy.component'
        when it sees several galaxies), which emit what it sees: in a band
        for an image, lines for a spectral cube; all of those by default.
        The opacity of every component applies.
    lines : Sequence of str, optional
        The names of the emission lines it sees (e.g. the line of the
        data); all by default.
    band : str or Band, optional
        The band of an image, or the name of a band of the catalogue in
        use (that of GBKFIT, or of a configuration, see lines_from): it
        sees the components that emit in every band or in it (see
        BandEmission), and dust of an extinction law dims them at its
        rest wavelength (see Galaxy).
    pixel_weights : Weights, optional
        The weights of the pixels of its data (e.g. less weight near the
        minor axis of a disk); none by default.
    flux_scale : str, optional
        The flux scale of its model (see above): 'one', or of an observable
        that has them (see Observable.flux_scales), 'per_spectrum' or
        'per_row'; its name is then required. None by default. Mock data
        take scales per spectrum or per row of 1 only.
    offsets : Mapping, optional
        The offsets of its frame that are fitted, for data whose positions
        or velocities are off from those of the other observations (e.g.
        by the error of their astrometry, or of the zero point of their
        velocities): 'position' (true or false) adds the parameters
        '{name}_dx' and '{name}_dy' (arcsec, +x to the west, +y to the
        north), and 'velocity' adds '{name}_dv' (km/s); its name is then
        required. They move what it sees: the centres (xpos, ypos) and
        systemic velocities (vsys) of all the components of the galaxies
        it sees. None by default; a galaxy needs an observation without
        them (see ObservationGroup).
    data : Dataset, optional
        The data, of the dataset class of the observable.
    likelihood : Likelihood, optional
        How the model is compared with the data; Gaussian by default when
        there are data, and none without data.
    oversampling : Sequence of int, optional
        How many times the model is oversampled along each axis of the
        data (an accuracy setting); 1 by default. The smooth components
        are sampled at the centre of each (sub)pixel, which differs from
        the mean of the pixel where the light or the velocity changes
        within it: at the centre of a disk a few pixels across, at edges
        inside the data, and in cubes, whose lines then miss the
        broadening by the velocity gradient within the pixel. On a mock
        disk of scale length 2.5 pixels at peak S/N 20, 1 biased the
        dispersion by +2 error bars and 2 by +0.5; on LVHIS data, of many
        pixels per beam, it changed nothing (see
        notes/adaptive-pixel-sampling.md).
    dtype : str, optional
        The floating type of the model: 'float32' or 'float64'. By default
        float64 if the noise of an item of its data is correlated (see
        Data): the whitening of its residuals (see LikelihoodGaussian)
        magnifies the rounding of a float32 model enough to stop least
        squares up to about 1 sigma short of the best fit (NGC 1313 of
        LVHIS, 2026-10-20); float32 otherwise.
    fft_dtype : str, optional
        The floating type of the FFT convolutions of the model by the PSF
        and the LSF: 'float64' (by default) or 'float32' (faster, for a
        float32 model only). The rounding of an FFT in float32 adds about
        1e-7 to 1e-6 of the brightest values of the cube to every value,
        which spoils the faint ones (e.g. their moments, and the
        finite-difference derivatives of fits).
    name : str, optional
        Its name, which is the id of its extra outputs instead of its
        position (see ObservationGroup.observation_label).
    unit : str, optional
        The unit of the light of its model without data (e.g. 'Jy/beam',
        see gbkfit.units.parse_unit): data give theirs. A unit per beam
        takes the beam of the PSF of the instrument (see PSF.beam_area),
        of each point of the spectral axis for a PSF that varies.

    Raises
    ------
    ValueError
        If a name is invalid, there are no galaxies or they repeat, a band
        is given to an observable that is not an image, it has a flux scale
        or offsets but no name, its flux scale is not one of its
        observable, the offsets are not position and velocity, true or
        false, the oversampling is not one integer of at least 1 for each
        axis, there is a likelihood without data, the data are not those of
        the observable, its model can lack values but its likelihood has no
        residual for them (see LikelihoodGaussian), its likelihood fits a
        noise but it has no name, the noise of an item is correlated where
        its kind of data cannot have it, its likelihood does not suit it
        (see Likelihood.check), the observable cannot be seen through the
        instrument (see Observable.check_observation), the dtype or the
        fft_dtype is not one of the native modules (see Driver.dtypes), or
        the fft_dtype is less precise than the dtype, it has pixel weights
        and a lens (not supported yet), or the unit is given with data, is
        not a unit, or is per beam without a beam (see _light_unit).
    """

    @staticmethod
    def type() -> str:
        return 'sky'

    @staticmethod
    def type_title() -> str:
        return 'Sky (through an instrument)'

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'ObservationSky':
        """
        Load an observation from its configuration.

        Its data are a dataset of the dataset class of its observable
        (see e.g. DatasetPixelSpectra.from_files, whose options outliers
        and mask_whole_spectra clean them), which gives the observable
        some of its options (see Observable.from_data).

        Parameters
        ----------
        info : dict
            The configuration.

        Returns
        -------
        Observation
            The observation.

        Raises
        ------
        ConfigError
            If the configuration has an error.
        ValueError, TypeError
            If an object it makes refuses its options.
        """
        info = dict(info)
        load_option_and_update_info(dataset_parser, info, 'data')
        load_option_and_update_info(
            observable_parser, info, 'observable', dataset=info.get('data'))
        return super().load(info)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        """
        Dump the observation to its configuration.

        Parameters
        ----------
        prefix : str, optional
            The start of the names of the files of its data and of the
            data of its parts (e.g. images; see Dataset.dump).
        dump_path : bool, optional
            Whether the configuration has the paths of the files, or only
            their names.
        overwrite : bool, optional
            Whether to overwrite existing files.

        Returns
        -------
        dict
            The options.
        """
        kwargs = dict(prefix=prefix, dump_path=dump_path, overwrite=overwrite)
        name = dict(name=self._name) if self._name is not None else {}
        galaxies = {} if self._galaxies is None else dict(
            galaxies=list(self._galaxies))
        components = {} if self._components is None else dict(
            components=list(self._components))
        lines = {} if self._lines is None else dict(lines=list(self._lines))
        band = {} if self._band is None else dict(
            band=band_parser.dump(self._band))
        pixel_weights = {} if self._pixel_weights is None else dict(
            pixel_weights=self._pixel_weights.dump())
        flux_scale = {} if self._flux_scale is None else dict(
            flux_scale=self._flux_scale)
        offsets = {} if not self._offsets else dict(offsets=dict(
            self._offsets))
        unit = {} if self._light is None else dict(
            unit=gbkfit.units.format_unit(self._light.unit))
        data = {} if self._data is None else dict(
            data=dataset_parser.dump(self._data, **kwargs),
            likelihood=likelihood_parser.dump(self._likelihood))
        return name | galaxies | components | lines | band | pixel_weights \
            | flux_scale | offsets | unit | data | dict(
            line_of_sight=line_of_sight_parser.dump(self._line_of_sight),
            foreground=foreground_parser.dump(self._foreground, **kwargs),
            instrument=instrument_parser.dump(self._instrument, **kwargs),
            observable=observable_parser.dump(
                self._observable, data=self._data, **kwargs),
            oversampling=self._oversampling,
            dtype=self._dtype.name,
            fft_dtype=self._fft_dtype.name)

    def __init__(
            self,
            observable: Observable,
            line_of_sight: LineOfSight,
            foreground: Foreground | None = None,
            instrument: Instrument | None = None,
            galaxies: Sequence[str] | None = None,
            components: Sequence[str] | None = None,
            lines: Sequence[str] | None = None,
            band: str | Band | None = None,
            pixel_weights: Weights | None = None,
            flux_scale: str | None = None,
            offsets: Mapping[str, bool] | None = None,
            data: Dataset | None = None,
            likelihood: Likelihood | None = None,
            oversampling: Sequence[int] | None = None,
            dtype: _DType | None = None,
            fft_dtype: _DType = 'float64',
            name: str | None = None,
            unit: str | None = None
    ):
        check_options(ObservationSky.__init__, locals())
        if isinstance(band, str):
            with config_path('band'):
                band = catalogue_in_use().band(band)
        if components is not None:
            components = tuple(components)
            check_component_names(components)
        if lines is not None:
            lines = tuple(lines)
            for line in lines:
                check_name(line)
        _check_flux_scale(flux_scale, observable, name)
        offsets = _check_offsets(offsets, name)
        if band is not None and observable.emission is not BandEmission:
            raise ValueError(
                "only images have a band; the lines of spectral cubes are "
                "their option lines")
        if foreground is None:
            foreground = Foreground()
        if name is None and foreground.pdescs():
            raise ValueError(
                "an observation with a lens of parameters needs a name, "
                "which names them (e.g. '{name}_lens_theta_e')")
        foreground = foreground.with_prefix(
            '' if name is None else f'{name}_')
        if instrument is None:
            instrument = Instrument()
        ndim = len(observable.size())
        oversampling = tuple(oversampling) if oversampling is not None \
            else (1,) * ndim
        if len(oversampling) != ndim or any(s < 1 for s in oversampling):
            raise ValueError(
                f"oversampling must have {ndim} values, each at least 1; it "
                f"is {oversampling}")
        if data is not None:
            observable.require_matching(data)
        self._observable = observable
        self._foreground = foreground
        self._instrument = instrument
        self._line_of_sight = line_of_sight
        self._components = components
        self._lines = lines
        self._band = band
        self._pixel_weights = pixel_weights
        self._flux_scale = flux_scale
        self._offsets = offsets
        self._oversampling = oversampling
        super().__init__(
            observable.keys(), galaxies, data, likelihood, name,
            observable.model_can_lack_values())
        observable.check_observation(
            instrument, self._oversampling, pixel_weights)
        if pixel_weights is not None and foreground.lens() is not None:
            raise ValueError("a lens does not support pixel_weights yet")
        self._light = None if unit is None \
            else _light_unit(unit, data, observable, instrument)
        if data is not None:
            _check_beams(data, instrument, observable)
        if dtype is None:
            dtype = 'float64' if data is not None \
                and _has_correlated_noise(data) else 'float32'
        self._dtype = np.dtype(dtype)
        self._fft_dtype = np.dtype(fft_dtype)
        if self._fft_dtype.itemsize < self._dtype.itemsize:
            raise ValueError(
                f"the fft_dtype ({self._fft_dtype.name}) is less precise "
                f"than the dtype ({self._dtype.name})")

    def line_of_sight(self) -> LineOfSight:
        """Return how it treats the line of sight."""
        return self._line_of_sight

    def components(self) -> tuple[str, ...] | None:
        """Return the names of the components it sees, if given."""
        return self._components

    def lines(self) -> tuple[str, ...] | None:
        """Return the names of the emission lines it sees, if given."""
        return self._lines

    def band(self) -> Band | None:
        """Return its band (of an image), if given."""
        return self._band

    def pixel_weights(self) -> Weights | None:
        """Return the weights of the pixels of its data, if any."""
        return self._pixel_weights

    def pdescs(self) -> dict[str, ParamDesc]:
        """
        Return its parameters: that of its flux scale and those of the
        offsets of its frame, if any (see the class), those of its lens,
        '{name}_lens_{parameter}' (see Foreground.pdescs), and those of the
        fitted noise of its likelihood, e.g. '{name}_noise_scale_{item}'
        (see LikelihoodGaussian).
        """
        flux_scale = {}
        if (scale := self.flux_scale()) is not None:
            name, kind = scale
            desc = f"the flux scale of observation {self._name}"
            flux_scale[name] = ParamScalarDesc(
                title='Flux scale', desc=desc, min=0, unit='', default=1) \
                if kind.axis is None else ParamVectorDesc(
                    kind.size, title='Flux scale', desc=desc, min=0,
                    unit='', default=1)
        descs = (
            ('Offset to the west', "the offset of its frame to the west",
             'arcsec'),
            ('Offset to the north', "the offset of its frame to the north",
             'arcsec'),
            ('Velocity offset', "the offset of its velocities", 'km/s'))
        offsets = {
            param: ParamScalarDesc(
                title=title, desc=f"{desc}, of observation {self._name}",
                unit=unit, default=0)
            for param, (title, desc, unit) in zip(
                self.offset_params(), descs)
            if param is not None}
        return flux_scale | offsets | self._foreground.pdescs() \
            | super().pdescs()

    def centre_params(self) -> tuple[tuple[str, str], ...]:
        return self._foreground.centre_params()

    def offset_params(self) -> tuple[str | None, str | None, str | None]:
        """
        Return the names of the parameters of the offsets of its frame
        (see the class): of x, y and the velocity, None for those it does
        not fit.
        """
        position = self._offsets.get('position', False)
        velocity = self._offsets.get('velocity', False)
        return (
            f'{self._name}_dx' if position else None,
            f'{self._name}_dy' if position else None,
            f'{self._name}_dv' if velocity else None)

    def sky_origin(self) -> tuple[float, float]:
        """
        Return the RA and Dec (degrees) of the origin of the frame of its
        model (see ObservationGroup): those of the reference pixel of the
        grid of its observable (see Observable.sky_rval).
        """
        return self._observable.sky_rval()

    def flux_scale(self) -> tuple[str, FluxScale] | None:
        """
        Return the parameter of its flux scale and how it scales its model
        (see the class), if it has one.
        """
        if self._flux_scale is None:
            return None
        return (f'{self._name}_flux_scale',
                self._observable.flux_scales()[self._flux_scale])

    def with_data(
            self, data: Dataset, likelihood: Likelihood | None = None
    ) -> Self:
        return type(self)(
            self._observable, self._line_of_sight, self._foreground,
            self._instrument, self._galaxies,
            self._components, self._lines, self._band, self._pixel_weights,
            self._flux_scale, self._offsets, data,
            likelihood if likelihood is not None else self._likelihood,
            self._oversampling, self._dtype.name, self._fft_dtype.name,
            self._name)

    def observable(self) -> Observable:
        """Return the observable."""
        return self._observable

    def velocity_frame(self) -> str | None:
        """
        Return the frame of rest of the velocities of its spectral axis
        (see SpectralAxis), if it has one that gives it.
        """
        spectral = self._observable.spectral()
        return None if spectral is None else spectral.frame()

    def foreground(self) -> Foreground:
        """Return the foreground."""
        return self._foreground

    def instrument(self) -> Instrument:
        """Return the instrument."""
        return self._instrument

    def make_dataset(self, items: Mapping[str, Data]) -> Dataset:
        """
        Make the dataset of data items measured as its observable (see
        Observable.make_dataset).
        """
        return self._observable.make_dataset(items)

    def data_unit(self) -> DataUnit | None:
        """
        Return the unit of the light of its data, which its model takes
        (see Observable.data_unit); None without data (its model is then
        in the unit of the components it sees), or without data of light.
        """
        return self._observable.data_unit(self._data, self._light)

    def output(
            self,
            data: np.ndarray,
            unit: astropy.units.UnitBase | None = None
    ) -> GridData | SpectraData | np.ndarray:
        """
        Return an array of the form of its data as an output, as its
        observable gives it (see Observable.output), with the unit, if
        given and the output takes it (GridData, SpectraData).
        """
        output = self._observable.output(data)
        if unit is None or not isinstance(
                output, (GridData, SpectraData)):
            return output
        return output._replace(unit=unit)

    def images(
            self,
            driver: Driver,
            values: Mapping[str, np.ndarray],
            weights: Mapping[str, np.ndarray] | None,
            units: Mapping[str, astropy.units.UnitBase | None] | None
    ) -> dict[str, GridData]:
        """
        Return the images of values of its data items that its observable
        makes (see Observable.images), on a driver.
        """
        return self._observable.images(driver, values, weights, units)

    def psf_images(
            self,
            driver: Driver,
            weights: Mapping[str, np.ndarray] | None
    ) -> dict[str, GridData]:
        """
        Return the images of the PSF of its data that its observable makes
        (see Observable.psf_images), on a driver.
        """
        return self._observable.psf_images(driver, weights)

    def oversampling(self) -> tuple[int, ...]:
        """Return the oversampling of the model along each axis."""
        return self._oversampling

    def dtype(self) -> np.dtype:
        """Return the floating type of the model."""
        return self._dtype

    def fft_dtype(self) -> np.dtype:
        """Return the floating type of the FFT convolutions of the model."""
        return self._fft_dtype

    def light_amplitudes(self, seen: SeenGalaxies) -> dict[str, str | None]:
        linear = self._observable.linear_in_light()
        amplitudes = {}
        for galaxy, mapping, emitters in zip(
                seen.galaxies, seen.mappings, self._scene(seen).emitters()):
            reasons = galaxy.light_amplitudes()
            for i in emitters:
                names = galaxy.component_params(i)
                for param in galaxy.components()[i].light_amplitudes():
                    reason = reasons[names[param]]
                    if reason is None and not linear:
                        reason = (
                            f"the observable {self._observable.type()!r} of "
                            f"the observation is not linear in the light "
                            f"(e.g. moments, ratios of sums of the light)")
                    amplitudes[mapping[names[param]]] = reason
        return amplitudes

    def orbit_components(
            self, seen: SeenGalaxies
    ) -> tuple[tuple[int, int], ...]:
        return self._scene(seen).orbit_emitters()

    def _scene(self, seen: SeenGalaxies) -> Scene:
        """
        Return the scene of the galaxies it sees, along its line of sight
        (see make_scene).
        """
        return make_scene(
            seen, self._components, self._lines, self._band,
            self._line_of_sight, self._observable.emission, self.data_unit())

    def check(self, seen: SeenGalaxies) -> None:
        scene = self._scene(seen)
        # (the light of an image is blurred by its PSF and again by ours)
        for trait in scene.image_traits():
            trait.check_psf(self._instrument.psf())
        if self._pixel_weights is not None:
            weights_geometry(self._pixel_weights, scene)
        self._line_of_sight.check_scene(scene)
        # (a flux scale of one value scales the flux of the model where the
        # observable records it; one of several values, its items of light)
        scale = self.flux_scale()
        one = scale is not None and scale[1].axis is None
        self._observable.check_scene(
            scene, self._foreground, self._instrument,
            scale[0] if one else None)

    def plan(self, driver: Driver, seen: SeenGalaxies) -> ObservablePlan:
        """
        Plan the evaluation of what it sees (see ObservationGroup.plan):
        the scene of the galaxies it sees, along its line of sight (see
        make_scene), as its observable.

        Parameters
        ----------
        driver : Driver
            The driver it is evaluated on.
        seen : SeenGalaxies
            The galaxies it sees.

        Returns
        -------
        ObservablePlan
            The plan.

        Raises
        ------
        ValueError
            If what it sees does not go with it (see make_scene), or with
            its observable (see Observable.plan).
        """
        scene = self._scene(seen)
        scale = self.flux_scale()
        one = scale is not None and scale[1].axis is None
        plan = self._observable.plan(
            driver, scene, self._line_of_sight, self._foreground,
            self._instrument, self._oversampling, self._dtype,
            self._fft_dtype, self._pixel_weights, scale[0] if one else None)
        if scale is None or one:
            return plan
        return _FluxScaledPlan(plan, driver, *scale, self._dtype)


def _light_unit(
        unit: str,
        data: Dataset | None,
        observable: Observable,
        instrument: Instrument
) -> LightUnit:
    """
    Return the unit of the light of the model of an observation without
    data, as given (see ObservationSky: unit): with the beam of the PSF of
    its instrument for a unit per beam, of each channel of the spectral
    axis of its observable for a PSF that varies (see PSF.beam_area).

    Raises
    ------
    ValueError
        If it has data, which give the unit, the unit is not a unit, or a
        unit per beam has no PSF, or one without a beam.
    """
    if data is not None:
        raise ValueError(
            "the data of an observation give the unit of its model; remove "
            "its unit")
    with config_path('unit'):
        unit = gbkfit.units.parse_unit(unit)
    if gbkfit.units.spatial_unit(unit) != astropy.units.beam:
        return LightUnit(unit)
    psf = instrument.psf()
    if psf is None:
        raise ValueError(
            f"a unit per beam ({gbkfit.units.format_unit(unit)}) takes the "
            f"beam of the PSF of the instrument, which has none")
    if not psf.varies():
        return LightUnit(unit, psf.beam_area())
    spectral = observable.spectral()
    if spectral is None:
        raise ValueError(
            "a PSF that varies along the spectral axis needs an observable "
            "with one")
    velocities = (
        spectral.rval() + spectral.step()
        * (np.arange(spectral.size()) - spectral.rpix())) \
        * astropy.units.km / astropy.units.s
    convention = VelocityConvention(
        spectral.rest(), spectral.medium(), spectral.convention())
    areas = [beam.beam_area() for beam in psf.at(velocities, convention)]
    return LightUnit(unit, np.array(areas), velocities)


def _check_offsets(
        offsets: Mapping[str, bool] | None, name: str | None
) -> dict[str, bool]:
    """
    Return the offsets of the frame of an observation (see Observation),
    those it fits, or raise ValueError if they are not position and
    velocity, true or false, or it has some but no name.
    """
    if offsets is None:
        return {}
    if not isinstance(offsets, Mapping) \
            or not set(offsets) <= {'position', 'velocity'} \
            or not all(isinstance(v, bool) for v in offsets.values()):
        raise ValueError(
            f"offsets are 'position' and 'velocity', true or false; they "
            f"are {offsets!r}")
    fitted = {key: True for key, value in offsets.items() if value}
    if fitted and name is None:
        raise ValueError(
            "an observation with offsets needs a name, which names their "
            "parameters (e.g. '{name}_dx')")
    return fitted


def _check_flux_scale(
        flux_scale: str | None,
        observable: Observable,
        name: str | None
) -> None:
    """
    Raise ValueError unless a flux scale is one of an observable (see
    Observable.flux_scales), and the observation has a name.
    """
    if flux_scale is None:
        return
    kinds = observable.flux_scales()
    if flux_scale not in kinds:
        desc = make_typed_desc(observable.__class__, 'observable')
        raise ValueError(
            f"{desc} has the flux scales {list(kinds)}; not {flux_scale!r}")
    if name is None:
        raise ValueError(
            "an observation with a flux scale needs a name, which names its "
            "parameter ('{name}_flux_scale')")


class _FluxScaledPlan(ObservablePlan):
    """
    The plan of an observable times a flux scale of several values (see
    ObservationSky): its items of light are times the values, each along
    the axis of the scale (one value at each point of the other axes).
    """

    def __init__(
            self,
            plan: ObservablePlan,
            driver: Driver,
            param: str,
            scale: FluxScale,
            dtype: np.dtype
    ):
        self._plan = plan
        self._driver = driver
        self._param = param
        self._scale = scale
        self._dtype = dtype
        # The values, of the shape of the items but 1 along the axis, on
        # the host (h) and the device (d), allocated on first use
        self._h_values: np.ndarray | None = None
        self._d_values: DeviceArray | None = None

    def evaluate(
            self,
            params: ParamValues,
            galaxies: Sequence[GalaxyState],
            out_extra: Extras | None
    ) -> ModelData:
        model = self._plan.evaluate(params, galaxies, out_extra)
        driver = self._driver
        if self._h_values is None:
            shape = model[self._scale.items[0]]['d'].shape
            (self._h_values, self._d_values) = driver.mem_alloc_s(
                tuple(1 if k == self._scale.axis else n
                      for k, n in enumerate(shape)), self._dtype)
        self._h_values[...] = np.reshape(
            params[self._param], self._h_values.shape)
        driver.mem_copy_h2d(self._h_values, self._d_values)
        for key in self._scale.items:
            values = model[key]['d']
            driver.math_mul(values, self._d_values, out=values)
        return model

    def mock(
            self,
            params: ParamValues,
            galaxies: Sequence[GalaxyState],
            noise: Noise,
            rng: np.random.Generator,
            ndraws: int
    ) -> tuple[list[HostModelData], dict[str, np.ndarray] | None]:
        """
        Draw mock data (see ObservablePlan.mock), of scales of 1 only: the
        noise is drawn where the observable records the light, which a
        scale of each spectrum does not reach.

        Raises
        ------
        ValueError
            If a value of the scale is not 1.
        """
        if np.any(params[self._param] != 1):
            raise ValueError(
                f"mock data take a flux scale per spectrum or per row of 1 "
                f"only; '{self._param}' is not")
        return self._plan.mock(params, galaxies, noise, rng, ndraws)

    def has_exact_errors(self) -> bool:
        return self._plan.has_exact_errors()


def _has_correlated_noise(data: Dataset) -> bool:
    """Return whether the noise of an item of the data is correlated."""
    return any(
        item.spatial_correlation() is not None
        or item.spectral_correlation() is not None
        for item in data.values())


def _check_beams(
        data: Dataset, instrument: Instrument, observable: Observable
) -> None:
    """
    Warn (see warn) if an item of the data has a beam, and the PSF of the
    instrument is not that beam: the data, of a radio interferometer, are
    smoothed by their beam, a Gaussian, and their light per beam is per
    it, but the PSF smooths the model otherwise. A cube of a beam for each
    channel is checked against the PSF at the point of each (see PSF.at).
    """
    psf = instrument.psf()
    for key in data.keys():
        beam = data[key].beam()
        if beam is None:
            continue
        if beam.spectral is None:
            if psf is not None and psf.varies():
                warn(
                    f"the data ({key}) have one beam, {_beam_text(beam)}, "
                    f"but the PSF of the instrument varies along the "
                    f"spectral axis: the model is smoothed by the PSF, and "
                    f"its light per beam is per beam of the data")
            elif not _is_beam(psf, beam):
                warn(
                    f"the data ({key}) have the beam {_beam_text(beam)}, "
                    f"but the PSF of the instrument is {_psf_text(psf)}: the "
                    f"model is smoothed by the PSF, and its light per beam is "
                    f"per beam of the data")
            continue
        # A beam for each channel, against the PSF at the point of each
        beams = [
            Beam(*values) for values in zip(beam.bmaj, beam.bmin, beam.bpa)]
        psfs = [None] * len(beams)
        if psf is not None:
            try:
                psfs = psf.at(
                    beam.spectral,
                    VelocityConvention.of(observable.grid().coords))
            except ValueError:
                # (points of the PSF that the spectral axis cannot place;
                # the model says so)
                continue
        differ = [
            i for i, (x, y) in enumerate(zip(psfs, beams))
            if not _is_beam(x, y)]
        if differ:
            i = differ[0]
            warn(
                f"the data ({key}) have a beam for each channel, but the PSF "
                f"of the instrument is another at {len(differ)} of their "
                f"{len(beams)} channels: e.g. at "
                f"{gbkfit.units.format_quantity(beam.spectral[i])}, the beam "
                f"{_beam_text(beams[i])} and the PSF {_psf_text(psfs[i])}; "
                f"the model is smoothed by the PSF, and its light per beam "
                f"is per beam of the data")


def _is_beam(psf: PSF | None, beam: Beam) -> bool:
    """
    Return whether a PSF is a beam (of one channel): a Gaussian of its
    axes, and of its angle unless it is round.
    """
    if not isinstance(psf, PSFGauss):
        return False
    own = psf.beam()
    return bool(np.allclose(
        (beam.bmaj, beam.bmin), (own.bmaj, own.bmin), rtol=1e-3, atol=0)
        and (np.isclose(beam.bmaj, beam.bmin, rtol=1e-3)
             or abs((beam.bpa - own.bpa + 90) % 180 - 90) < 0.1))


def _beam_text(beam: Beam) -> str:
    """A beam (of one channel), in messages."""
    return f"{beam.bmaj:g} x {beam.bmin:g} arcsec at {beam.bpa:g} deg"


def _psf_text(psf: PSF | None) -> str:
    """A PSF, in messages: its beam, if it is a Gaussian."""
    if psf is None:
        return "none (a point)"
    if isinstance(psf, PSFGauss):
        return f"the beam {_beam_text(psf.beam())}"
    return f"of type {psf.type()!r}, not a Gaussian"
