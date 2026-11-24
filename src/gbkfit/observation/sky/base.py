"""
The base of the observations on the sky (see ObservationSky): galaxies
seen through an instrument as one kind of data (images, moment maps,
spectral cubes and others), along a line of sight, through the lenses in
front of them.
"""

import abc
from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING, Annotated, Any, ClassVar, Literal, NamedTuple, TypeAlias)

import astropy.units
import numpy as np

import gbkfit.units
from gbkfit.config import Choices, check_name, config_path, make_typed_desc
from gbkfit.dataset import Dataset, DatasetGrid
from gbkfit.driver import Driver
from gbkfit.grid import (
    Grid, GridData, SpectraData, SpectralAxis, check_convention,
    make_cube_grid, make_grid, per_axis)
from gbkfit.instrument import Instrument, instrument_parser
from gbkfit.io import Beam
from gbkfit.lensing import SourceGrid
from gbkfit.model import (
    Band, DataUnit, LightUnit, Reference, View, band_parser)
from gbkfit.model.components.lines import catalogue_in_use
from gbkfit.params import ParamDesc, ParamScalarDesc, ParamVectorDesc
from .._scene import SeenScenes, check_component_names, make_scenes
from ..depth import Depth, check_scene
from ..noise import Noise, noise_parser
from ..observation import Observation, SeenGalaxies, SkyFrame
from ..plan import FluxScale, ObservationPlan
from ..weights import Weights, weights_geometry
from ._checks import (
    check_beams, check_flux_scale, check_offsets, has_correlated_noise,
    light_unit)
from ._cube_plan import SkyDrawing
from ._flux_scale import FluxScaledPlan
from ._options import (
    GRID_OPTIONS, config_value, dump_grid, in_files,
    without_options_from_data)


if TYPE_CHECKING:
    from gbkfit.dataset import Data


__all__ = [
    'ObservationSky'
]


def unmasked_pixels(
        dataset: Dataset | None, shape: tuple[int, int]
) -> np.ndarray:
    """
    Return the pixels of a grid of the given shape (ny, nx) where an item
    of the data has a value that counts (see Data.mask), in any channel;
    all of them without data. The items are of the grid, or of its
    channels before it (e.g. a spectral cube).
    """
    if dataset is None:
        return np.ones(shape, dtype=bool)
    measured = np.zeros(shape, dtype=bool)
    for key in dataset.keys():
        counts = dataset[key].mask() != 0
        measured |= counts.reshape((-1,) + shape).any(axis=0)
    return measured


# The floating types of models: those the native modules have kernels for
DType: TypeAlias = Annotated[
    Literal[tuple(dt.name for dt in Driver.dtypes())], Choices(
        float32='Single precision (float32)',
        float64='Double precision (float64)')]


class SkyOptions(NamedTuple):
    """
    The options of an observation on the sky (see ObservationSky) that
    its checks and plans take, made once by it.

    Attributes
    ----------
    depth : Depth
        The depth axis along the line of sight, for the scenes that need
        one, on which its plan plans its scene (see plan_scene of
        gbkfit.observation.depth).
    source_grid : SourceGrid
        How the grids of the planes behind lenses are laid out, which its
        plan draws them on (see plan_scenes of gbkfit.observation.depth).
    instrument : Instrument
        The telescope and the instrument.
    oversampling : tuple of int
        How many times the model is oversampled along each axis of the
        data.
    dtype : np.dtype
        The floating type of the model.
    fft_dtype : np.dtype
        The floating type of the FFT convolutions of the model (at least
        as precise as dtype).
    pixel_weights : Weights or None
        The pixel weights of the data, if any.
    flux_scale_param : str or None
        The name of the parameter of the flux scale of the observation, if
        it is one value (the flux scale 'one'), which scales the flux of
        the model where the observation records it. A flux scale of
        several values multiplies the items of light of the model instead
        (see ObservationSky), and is not here.
    """
    depth: Depth
    source_grid: SourceGrid
    instrument: Instrument
    oversampling: tuple[int, ...]
    dtype: np.dtype
    fft_dtype: np.dtype
    pixel_weights: Weights | None
    flux_scale_param: str | None


class ObservationSky(Observation, abc.ABC):
    """
    An observation on the sky: galaxies seen through an instrument as one
    kind of data (its type: e.g. an image, moment maps or a spectral
    cube). They are seen along a line of sight, through the lenses in
    front of them (see ObservationGroup) and the instrument; paired with
    data, an objective compares the model of the galaxies with them under
    the likelihood of its setup (see Observation).

    Each kind of data is a subclass, which gives the form of the data
    (their items, see keys) and where they are measured (e.g. their grid),
    declares the index of the spectral axis of its data (spectral_axis,
    None if none) and the class of its datasets (dataset_class). Its
    options are those of its kind, then those of every observation on the
    sky (below).

    Paired with data (see Observation.with_dataset), of the dataset class
    of its kind, the data give the options of its kind of
    sampling_from_data (its sampling: e.g. its grid; see
    _options_from_data), and the unit of its light (see unit): the data
    win, and those given, for the observation without data (e.g. one that
    renders galaxies on its own grid, and is fitted to data of any grid),
    are left out, with a line in the log. Without data, those of
    required_without_data are required, and the others take their values
    of defaults_without_data. The files of its options (e.g. regions
    without data) are named with the prefix of the dump, as those of the
    datasets.

    Its model is in the unit of its data, if it is paired with data (see
    data_unit): the light of the components whose emission has a unit is
    turned into it (see Component.plan), and that of the others is in it.
    Without data, the model is in its unit, if given, or else in the unit
    of the components it sees, which must then have one (or none).

    Its model can have a flux scale: its items of light (see light_items)
    are times the values of the parameter '{name}_flux_scale', one value
    ('one'; e.g. the calibration between the instruments of a joint fit),
    one for each spectrum of its data ('per_spectrum'), or one for each
    line of the image of slitless spectra along the dispersion
    ('per_row', the row amplitudes of geko). Its values can be fixed,
    tied, free or solved (see ParamSpace): e.g. {solve: {type: gauss,
    mean: 1, sigma: 0.5, min: 0}} (see SolveGauss). Scales per spectrum
    make a fit take the shape of each spectrum from the model (where its
    lines are, how wide; with beam smearing, also how the light of the
    model weighs its velocities), and its brightness mostly from the
    data, which suits a model whose light misses structure of the data
    (e.g. the clumps of a clumpy disk); solved, each is worked out at each
    evaluation with exact formulas, never putting the flux of the data
    into the model.

    Why that prior for scales per spectrum: the log-likelihood of samplers
    needs one (they integrate the solved values out). A flat prior on a
    scale makes the integral grow as the model gets fainter (half the
    light, twice the scales, twice the integral): samplers would prefer
    galaxies of no light. A flat prior on the flux of each spectrum (the
    scale times the sum of the model there) removes that, but favours
    model lines whose flux the data pin down loosely (broad lines) where
    the spectra are faint: in 4000 random spectra of 61 channels of
    10 km/s (noise 1), the log-evidence of a 60 km/s model line over a
    30 km/s one was +0.35 on noise alone and +0.22 on a faint line truly
    30 km/s wide (flux 3); with this prior (sigma 0.5), +0.03 and -0.09.
    Over thousands of faint spectra (e.g. of clumpy disks at z ~ 1.5) the
    flat prior would pull the dispersion up. This prior is proper (the
    evidence is a normal one), and keeps the brightness of the model
    meaningful: it is the mean brightness, about which the spectra
    scatter. Its cost is sigma, a statement of how much the spectra may
    differ from the model: a very large sigma is a flat prior on the
    scales again, with its bias.

    Parameters
    ----------
    source_grid : SourceGrid, optional
        How the grids of the planes of galaxies behind lenses that it sees
        are laid out (see SourceGrid); SourceGrid() by default (automatic).
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
    continuum : bool, optional
        Whether a spectral observation sees the velocities of the
        continuum light of the components that emit in bands and move
        (e.g. the kinematics of their stars), besides their lines; not by
        default. Not of images.
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
        The flux scale of its model (see above): 'one', or of a kind that
        has them (see flux_scale_kinds), 'per_spectrum' or 'per_row'; its name
        is then required. None by default. Mock data take scales per
        spectrum or per row of 1 only.
    offsets : Mapping, optional
        The offsets of its frame that are fitted, for data whose positions
        or velocities are off from those of the other observations (e.g.
        by the error of their astrometry, or of the zero point of their
        velocities): 'position' (true or false) adds the parameters
        '{name}_dx' and '{name}_dy' (arcsec, +x to the west, +y to the
        north), and 'velocity' adds '{name}_dv' (km/s); its name is then
        required. They move what it sees: the centres (xpos, ypos) and
        systemic velocities (vsys) of all the components of the galaxies
        it sees, and the lenses in front of them (see TracePlan). None by
        default; a galaxy needs an observation without them (see
        ObservationGroup).
    noise : Noise, optional
        The noise of its mock data (see Observation).
    oversampling : Sequence of int, optional
        How many times the model is oversampled along each axis of the
        data (an accuracy setting); 1 along each by default. The smooth components
        are sampled at the centre of each (sub)pixel, which differs from
        the mean of the pixel where the light or the velocity changes
        within it: at the centre of a disk a few pixels across, at edges
        inside the data, and in cubes, whose lines then miss the
        broadening by the velocity gradient within the pixel. On a mock
        disk of scale length 2.5 pixels at peak S/N 20, 1 biased the
        dispersion by +2 error bars and 2 by +0.5; on data of many pixels
        per beam (e.g. the cubes of LVHIS), it changed nothing.
    depth : Depth, optional
        The depth axis along the line of sight, if what it sees needs one
        (thick components, or opacity; see Scene.evaluation), and the
        options of radiative transfer along it; Depth() by default. What
        it sees decides how it is evaluated along the line of sight: on
        the plane of the sky, on the depth axis, or by radiative transfer
        (line opacity, or several galaxies that dim one another).
    dtype : str, optional
        The floating type of the model: 'float32' or 'float64'. By default
        float64 if the noise of an item of its data is correlated (see
        Data): the whitening of its residuals (see LikelihoodGauss)
        magnifies the rounding of a float32 model enough to stop least
        squares up to about 1 sigma short of the best fit; float32
        otherwise (and without data). Paired with other data (see
        with_dataset, e.g. mock data), it chooses again, by them.
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
        of each point of the spectral axis for a PSF that varies. Mock
        data of the model have the unit, and that beam: of a PSF that is
        not a Gaussian, the round Gaussian of its solid angle (see
        draw_mocks).

    Raises
    ------
    ValueError
        If a name is invalid, there are no galaxies or they repeat, a band
        is given to a kind that is not an image, it has a flux scale or
        offsets but no name, its flux scale is not one of its kind, the
        offsets are not position and velocity, true or false, the
        oversampling is not one integer of at least 1 for each axis, an
        option of its kind needed without data is missing, its kind
        cannot be seen through the instrument (see _check_options), the
        dtype or the fft_dtype is not one of the native modules (see
        Driver.dtypes), or the fft_dtype is less precise than the dtype,
        or the unit is not a unit, or is per beam without a beam (see
        light_unit).
    """

    spectral_axis: ClassVar[int | None]

    # Whether it sees a spectral axis (a spectral cube and what is made of
    # one), or a band (an image); see view
    spectral_view: ClassVar[bool]

    # What it sees of the stars on orbits whose weights are solved (see
    # Component.orbit_weights and _check_scenes): their light ('light', of
    # an image), their moments ('moments', see ObservationRegionMoments),
    # or nothing
    orbit_stars: ClassVar[Literal['light', 'moments'] | None]

    # The titles of its options, and of those of several kinds: of their
    # grids (the spatial axes, and the spectral axis of a kind with one)
    # and their smoothing
    config_titles = dict(
        size='Pixels', step='Pixel size', rpix='Reference pixel',
        rval='Reference position', rota='Rotation',
        spectral='Spectral axis', smooth_weights='Smooth weights',
        median_size='Median filter', source_grid='Source-plane grid',
        instrument='Instrument', components='Components', lines='Lines',
        continuum='Continuum', band='Band', pixel_weights='Pixel weights',
        flux_scale='Flux scale', offsets='Offsets',
        oversampling='Oversampling', depth='Depth', dtype='Precision',
        fft_dtype='FFT precision', unit='Unit')

    # The options of its kind that data give (see sampling_from_data):
    # those required without data, and the values of the others without
    # data, when not given
    required_without_data: ClassVar[tuple[str, ...]] = ('size',)
    defaults_without_data: ClassVar[Mapping[str, Any]] = dict(
        step=(1, 1), rval=(0, 0), rota=0)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        """
        Dump the observation to its configuration: its type, the options
        of its kind but those its data give, if it is paired with data
        (see options_from_data), and those of every observation on the
        sky. It has no data (a setup pairs it with its dataset).

        Parameters
        ----------
        prefix : str, optional
            The start of the names of the files of its parts (e.g. images;
            see Dataset.dump), and of its options (e.g. the table of a
            trace).
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
        options = self._sky_options
        name = dict(name=self._name) if self._name is not None else {}
        galaxies = {} if self._galaxies is None else dict(
            galaxies=list(self._galaxies))
        components = {} if self._components is None else dict(
            components=list(self._components))
        lines = {} if self._lines is None else dict(lines=list(self._lines))
        continuum = dict(continuum=True) if self._continuum else {}
        band = {} if self._band is None else dict(
            band=band_parser.dump(self._band))
        pixel_weights = {} if options.pixel_weights is None else dict(
            pixel_weights=options.pixel_weights.dump())
        flux_scale = {} if self._flux_scale is None else dict(
            flux_scale=self._flux_scale)
        offsets = {} if not self._offsets else dict(offsets=dict(
            self._offsets))
        unit = {} if self._light is None else dict(
            unit=gbkfit.units.format_unit(self._light.unit))
        noise = {} if self._noise is None else dict(
            noise=noise_parser.dump(self._noise, **kwargs))
        # (only a given dtype: one the data chose is chosen again by the
        # data it is loaded with, e.g. mock data of correlated noise)
        dtype = {} if self._given_dtype is None else dict(
            dtype=self._given_dtype)
        return dict(type=self.type()) | name \
            | self._dump_kind(prefix, dump_path, overwrite) | galaxies \
            | components | lines | continuum | band | pixel_weights \
            | flux_scale | offsets | unit | noise | dtype | dict(
                depth=options.depth.dump(),
                source_grid=options.source_grid.dump(),
                instrument=instrument_parser.dump(
                    options.instrument, **kwargs),
                oversampling=options.oversampling,
                fft_dtype=options.fft_dtype.name)

    def _dump_kind(
            self, prefix: str, dump_path: bool, overwrite: bool
    ) -> dict[str, Any]:
        """
        Return the configuration of the options of its kind (see dump):
        its grid (see _dump_grid) and its other options (see
        _dump_options), without those that its data give, if any (see
        options_from_data). Kinds with options in files that their data
        give (e.g. regions) override this: such a file is written only
        without data. prefix, dump_path and overwrite are those of dump.
        """
        return without_options_from_data(
            self, self._dump_grid()
            | self._dump_options(prefix, dump_path, overwrite),
            self._dataset)

    def _dump_grid(self) -> dict[str, Any]:
        """
        Return the configuration of its grid (see dump_grid); kinds whose
        options give their grid otherwise (e.g. a slit by its position
        angle) override this.
        """
        return dump_grid(self._grid)

    @abc.abstractmethod
    def _dump_options(
            self, prefix: str, dump_path: bool, overwrite: bool
    ) -> dict[str, Any]:
        """
        Return the configuration of the options of its kind but its grid
        (see _dump_kind), in their order in the configuration; prefix,
        dump_path and overwrite are those of dump.
        """

    @classmethod
    def options_from_data(cls, dataset: Dataset) -> tuple[str, ...]:
        """
        Return the options of an observation of its kind that a dataset
        gives (see Observation.options_from_data): those of its kind (its
        sampling, see sampling_from_data), and the unit of its light.
        """
        return (*cls.sampling_from_data(dataset), 'unit')

    @classmethod
    def sampling_from_data(cls, dataset: Dataset) -> tuple[str, ...]:
        """
        Return the options of its kind that a dataset gives: where and
        what the data were measured (e.g. the grid, the regions, the orders
        of moments).

        Here, the grid (and its spectral axis, if any); kinds of other
        data override this.

        Parameters
        ----------
        dataset : Dataset
            The data.

        Returns
        -------
        tuple of str
            The names of the options.
        """
        return GRID_OPTIONS + (
            ('spectral',) if cls.spectral_axis is not None else ())

    @classmethod
    def data_options(
            cls,
            dataset: Dataset,
            prefix: str | None = None,
            dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        """
        Return the options of its kind that a dataset gives (see
        sampling_from_data) as a configuration, as far as the data alone
        give them (see Observation.data_options).
        """
        cls.check_dataset_class(dataset)
        values = cls._options_from_data(dataset, {})
        return {
            key: config_value(values[key], prefix, dump_path, overwrite)
            for key in cls.sampling_from_data(dataset)
            if key in values and (prefix is not None or not in_files(
                values[key]))}

    def _options(self, options: dict[str, Any]) -> dict[str, Any]:
        """
        Return the options of its kind (see the class), as given (None
        where not given), completed, before it is made: those of
        required_without_data are required (paired with data, the data
        give them; see Observation.with_dataset), and those not given take
        their values of defaults_without_data.

        Parameters
        ----------
        options : dict
            The options of its kind, as given.

        Returns
        -------
        dict
            The options of its kind.

        Raises
        ------
        ValueError
            If one required without data is not given.
        """
        if missing := [
                key for key in self.required_without_data
                if options.get(key) is None]:
            desc = make_typed_desc(type(self), 'observation')
            raise ValueError(
                f"{desc} without data needs the options {missing}")
        return options | {
            key: value for key, value in self.defaults_without_data.items()
            if options.get(key) is None}

    @classmethod
    def _make_grid(
            cls,
            size: int | Sequence[int],
            step: float | Sequence[float],
            rpix: float | Sequence[float] | None,
            rval: float | Sequence[float],
            rota: float,
            spectral: SpectralAxis | None = None
    ) -> Grid:
        """
        Make the grid of its data: its axes on the sky (size, step, rpix,
        rval, rota; rpix is the centre if None), and of a kind with a
        spectral axis (see spectral_axis), the spectral axis after them
        (see make_cube_grid), whose velocities need their convention (see
        check_convention).
        """
        # (the axes on the sky are those before the spectral axis, or
        # without one, those of an image)
        naxes = 2 if cls.spectral_axis is None else cls.spectral_axis
        size = per_axis('size', size, naxes)
        if cls.spectral_axis is None:
            return make_grid(size, step, rpix, rval, rota)
        check_convention(spectral)
        return make_cube_grid(size, step, rpix, rval, rota, spectral)

    def __init__(
            self,
            grid: Grid,
            source_grid: SourceGrid | None,
            instrument: Instrument | None,
            galaxies: Sequence[str] | None,
            components: Sequence[str] | None,
            lines: Sequence[str] | None,
            continuum: bool,
            band: str | Band | None,
            pixel_weights: Weights | None,
            flux_scale: str | None,
            offsets: Mapping[str, bool] | None,
            noise: Noise | None,
            oversampling: Sequence[int],
            depth: Depth | None,
            dtype: DType | None,
            fft_dtype: DType,
            name: str | None,
            unit: str | None
    ):
        """
        Make the observation of a kind, on the grid of its data: a kind
        checks its options first (see check_options of gbkfit.config),
        completes those of its kind (see _options), and sets what it needs
        to give its data items (see keys) and its flux scales (see
        flux_scale_kinds).
        """
        self._grid = grid
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
        check_flux_scale(flux_scale, self, name)
        offsets = check_offsets(offsets, name)
        if band is not None and self.spectral_view:
            raise ValueError(
                "only images have a band; the lines of spectral cubes are "
                "their option lines")
        if continuum and not self.spectral_view:
            raise ValueError(
                "an image sees the light in its band; the continuum is of "
                "spectral observations")
        if instrument is None:
            instrument = Instrument()
        # (one value per axis, of at least 1, as its kind's annotation
        # gives it)
        oversampling = tuple(oversampling)
        self._components = components
        self._lines = lines
        self._continuum = continuum
        self._band = band
        self._flux_scale = flux_scale
        self._offsets = offsets
        super().__init__(
            self.keys(), galaxies, noise, name,
            self.model_can_have_missing_values())
        # (as given, for copies of other data and dumps: by default, the
        # data choose it, see _pair)
        self._given_dtype = dtype
        dtype = np.dtype('float32' if dtype is None else dtype)
        fft_dtype = np.dtype(fft_dtype)
        _check_fft_dtype(dtype, fft_dtype)
        # (a flux scale of one value scales the flux of the model where it
        # records it; one of several values, its items of light, see plan)
        scale = self.flux_scale()
        one = scale is not None and scale[1].axis is None
        self._sky_options = SkyOptions(
            depth=depth if depth is not None else Depth(),
            source_grid=source_grid if source_grid is not None
            else SourceGrid(),
            instrument=instrument, oversampling=oversampling, dtype=dtype,
            fft_dtype=fft_dtype, pixel_weights=pixel_weights,
            flux_scale_param=scale[0] if one else None)
        self._check_options(self._sky_options)
        # (the unit of the model without data, and its beam, of a unit per
        # beam: those of data of the model, e.g. mock data)
        self._light, self._light_beam = (None, None) if unit is None \
            else light_unit(unit, self, instrument)

    def _pair(self, dataset: Dataset) -> None:
        """
        Pair it with a dataset (see Observation._pair): the data must be
        of its data items and measured where it is modelled (see
        check_dataset); they choose the floating type of its model, if not
        given (see the class); and their beams are checked against its PSF
        (see check_beams).

        Raises
        ------
        ValueError
            If the data are not of it, or the fft_dtype is less precise
            than the floating type the data choose.
        """
        self.check_dataset(dataset)
        super()._pair(dataset)
        if self._given_dtype is None and has_correlated_noise(dataset):
            options = self._sky_options._replace(dtype=np.dtype(np.float64))
            _check_fft_dtype(options.dtype, options.fft_dtype)
            self._sky_options = options
        check_beams(dataset, self._sky_options.instrument, self)

    def _constructor_options(self) -> dict[str, Any]:
        options = self._sky_options
        unit = None if self._light is None \
            else gbkfit.units.format_unit(self._light.unit)
        return self._kind_options() | dict(
            source_grid=options.source_grid, instrument=options.instrument,
            galaxies=self._galaxies, components=self._components,
            lines=self._lines, continuum=self._continuum, band=self._band,
            pixel_weights=options.pixel_weights,
            flux_scale=self._flux_scale, offsets=self._offsets,
            noise=self._noise, oversampling=options.oversampling,
            depth=options.depth, dtype=self._given_dtype,
            fft_dtype=options.fft_dtype.name, name=self._name, unit=unit)

    @abc.abstractmethod
    def _kind_options(self) -> dict[str, Any]:
        """
        Return the options of its kind (those of its __init__ but those of
        every observation on the sky), all of them, as it has them (with
        those its data gave, if any): they make it again (see
        Observation.with_dataset), and its plan takes them (see
        same_plan).
        """

    def grid(self) -> Grid:
        """Return the grid of the data."""
        return self._grid

    def size(self) -> tuple[int, ...]:
        """Return the number of pixels of each axis of the grid."""
        return self._grid.size

    def step(self) -> tuple[float, ...]:
        """Return the step of each axis of the grid."""
        return self._grid.coords.step

    def rpix(self) -> tuple[float, ...]:
        """Return the reference pixel of each axis of the grid."""
        return self._grid.coords.rpix

    def rval(self) -> tuple[float, ...]:
        """Return the reference value of each axis of the grid."""
        return self._grid.coords.rval

    def rota(self) -> float:
        """Return the rotation of the grid on the sky."""
        return self._grid.coords.rota

    def sky_rval(self) -> tuple[float, float]:
        """
        Return the RA and Dec (degrees) of the origin of the frame of its
        model, from which the centres of what it sees are measured (see
        ObservationGroup): those of the reference pixel of its grid.
        """
        rval = self.rval()
        return float(rval[0]), float(rval[1])

    def spectral(self) -> SpectralAxis | None:
        """Return the spectral axis, if any."""
        if self.spectral_axis is None:
            return None
        return SpectralAxis.of(self._grid)

    @abc.abstractmethod
    def keys(self) -> tuple[str, ...]:
        """Return the names of the data items (e.g. 'spectra')."""

    @abc.abstractmethod
    def model_can_have_missing_values(self) -> bool:
        """
        Check whether its model can have missing values (NaN) where the
        data have some (e.g. the velocity of a spectrum below its
        flux_floor): the likelihood then needs a residual for them (see
        LikelihoodGauss).
        """

    def linear_in_light(self) -> bool:
        """
        Check whether its model is linear in the light of what it sees:
        the sum of the models of parts of the light, each times its
        amplitude (e.g. images and spectra, but not moments, which are
        ratios of sums of the light, nor a model that can have missing
        values, see model_can_have_missing_values), so that amplitudes of
        the light can be solved (see params.Solve); False here.
        """
        return False

    @abc.abstractmethod
    def make_dataset(self, items: Mapping[str, 'Data']) -> Dataset:
        """
        Make the dataset of data items measured as this observation (e.g.
        mock data), where its model is: the inverse of its options from
        data (see _options_from_data), where its data say the instrument
        looked (see footprint).

        Parameters
        ----------
        items : Mapping
            The data items, by name (see keys).

        Returns
        -------
        Dataset
            The dataset, of the dataset class of its kind.
        """

    def footprint(self) -> np.ndarray | None:
        """
        Return where the instrument looked, as its data say (see
        DatasetGrid); None without data on a grid, or without a footprint.
        """
        dataset = self._dataset
        return dataset.footprint() if isinstance(dataset, DatasetGrid) \
            else None

    def data_unit(self) -> DataUnit | None:
        """
        Return the unit of the light of its data, which its model takes
        (see _data_unit); None without data (its model is then in the unit
        of the components it sees), or without data of light.
        """
        return self._data_unit(self._dataset, self._light)

    @abc.abstractmethod
    def _data_unit(
            self,
            dataset: Dataset | None,
            light: LightUnit | None
    ) -> DataUnit | None:
        """
        Return the unit of the light of data of it, which its model takes.

        Parameters
        ----------
        dataset : Dataset, optional
            The data.
        light : LightUnit, optional
            The unit of the light of its model without data, if given.

        Returns
        -------
        DataUnit or None
            The unit of the data item of light (e.g. 'spectra' or
            'moment0'; see DataUnit), with its beam and the area of its
            pixels; without data, that of the given light; None without
            either, or without an item of light (e.g. moments of order 1
            and above only: velocities).
        """

    @abc.abstractmethod
    def light_items(self) -> tuple[str, ...]:
        """
        Return the names of its data items of light (e.g. 'spectra' or
        'moment0'): those that are times the flux of the model, which a
        flux scale multiplies (see flux_scale_kinds); none for data of
        velocities only.
        """

    def flux_scale_kinds(self) -> dict[str, FluxScale]:
        """
        Return the flux scales that its model can have (see the class), by
        kind: 'one', of its items of light, if any. Kinds of spectra add
        'per_spectrum' (or, of the lines of an image along an axis,
        'per_row').
        """
        items = self.light_items()
        return {} if not items else dict(one=FluxScale(items, None, 1))

    def _check_options(self, options: SkyOptions) -> None:
        """
        Check its options of every observation on the sky for its kind
        (e.g. the instrument it is seen through, the oversampling of its
        model and its pixel weights, if any), before any plan.

        It warns about the options of its kind that the instrument makes
        useless (see warn). Here, every option suits; kinds override this.

        Parameters
        ----------
        options : SkyOptions
            Its options.

        Raises
        ------
        ValueError
            If its kind cannot be seen so.
        """

    def check_dataset(self, dataset: Dataset) -> None:
        """
        Check that a dataset holds the data of this observation.

        The dataset must be of the dataset class of its kind, have its
        data items, and have been measured where it is modelled.

        Parameters
        ----------
        dataset : Dataset
            The data.

        Raises
        ------
        ValueError
            If the dataset does not hold the data of this observation.
        """
        self.check_dataset_class(dataset)
        if set(dataset.keys()) != set(self.keys()):
            desc = make_typed_desc(self.__class__, 'observation')
            raise ValueError(
                f"{desc} has the data items {sorted(self.keys())}, but the "
                f"data have {sorted(dataset.keys())}")
        self._check_dataset_coordinates(dataset)

    def _check_dataset_coordinates(self, dataset: Dataset) -> None:
        """
        Raise ValueError unless the data were measured on the grid of this
        observation. Kinds of other data override this.
        """
        if dataset.grid() != self._grid:
            desc = make_typed_desc(self.__class__, 'observation')
            raise ValueError(
                f"the data are on the grid {dataset.grid()}, but {desc} "
                f"is on the grid {self._grid}")

    def same_plan(self, other: Observation, seen: SeenGalaxies) -> bool:
        """
        Check whether its plan is also that of another observation (see
        Observation.same_plan): its plan takes the options of its kind,
        which data give (e.g. its grid; see _kind_options); with planes
        behind lenses, the pixels its data measure too, whose rays it
        traces (see _measured_pixels).
        """
        if not super().same_plan(other, seen) or type(other) is not type(
                self) or other._kind_options() != self._kind_options():
            return False
        return not seen.lensed() or np.array_equal(
            self._measured_pixels(), other._measured_pixels())

    def depth(self) -> Depth:
        """Return its depth axis, for what it sees that needs one."""
        return self._sky_options.depth

    def components(self) -> tuple[str, ...] | None:
        """Return the names of the components it sees, if given."""
        return self._components

    def lines(self) -> tuple[str, ...] | None:
        """Return the names of the emission lines it sees, if given."""
        return self._lines

    def continuum(self) -> bool:
        """Check whether it sees the continuum (see the class)."""
        return self._continuum

    def view(self) -> View:
        """
        Return what it sees of the light of the components (see View): its
        band, or its spectral axis with its lines and continuum.
        """
        return View(
            self.spectral_view, self._band, self._lines, self._continuum)

    def band(self) -> Band | None:
        """Return its band (of an image), if given."""
        return self._band

    def pixel_weights(self) -> Weights | None:
        """Return the weights of the pixels of its data, if any."""
        return self._sky_options.pixel_weights

    def pdescs(self) -> dict[str, ParamDesc]:
        """
        Return its parameters: that of its flux scale and those of the
        offsets of its frame, if any (see the class).
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
                self._offset_params(), descs)
            if param is not None}
        return flux_scale | offsets

    def _offset_params(self) -> tuple[str | None, str | None, str | None]:
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

    def sky_frame(self) -> SkyFrame:
        """
        Return the frame of its model on the sky (see SkyFrame and
        ObservationGroup): its origin is the reference pixel of its grid
        (see sky_rval), its offsets those it fits (see the class), and the
        frame of rest of its velocities that of its spectral axis (see
        SpectralAxis), if it has one that gives it.
        """
        spectral = self.spectral()
        return SkyFrame(
            self.sky_rval(), self._offset_params(),
            None if spectral is None else spectral.frame())

    def flux_scale(self) -> tuple[str, FluxScale] | None:
        """
        Return the parameter of its flux scale and how it scales its model
        (see the class), if it has one.
        """
        if self._flux_scale is None:
            return None
        return (f'{self._name}_flux_scale',
                self.flux_scale_kinds()[self._flux_scale])

    def source_grid(self) -> SourceGrid:
        """
        Return how the grids of the planes behind lenses that it sees are
        laid out.
        """
        return self._sky_options.source_grid

    def instrument(self) -> Instrument:
        """Return the instrument."""
        return self._sky_options.instrument

    def model_unit(
            self, key: str
    ) -> tuple[astropy.units.UnitBase | None, Beam | None]:
        """
        Return the unit of the values of a data item of its model, and
        their beam, of a unit per beam (see Observation.model_unit): those
        of its data, if any; without data, its items of light (see
        light_items) have its unit, if given (see the class), with the
        beam of the PSF of its instrument for a unit per beam (see
        light_unit), and the others none.
        """
        if self._dataset is not None or self._light is None \
                or key not in self.light_items():
            return super().model_unit(key)
        return self._light.unit, self._light_beam

    def output(
            self,
            data: np.ndarray,
            unit: astropy.units.UnitBase | None = None
    ) -> GridData | SpectraData | np.ndarray:
        """
        Return an array of the form of its data as an output (see
        _output), with the unit, if given and the output takes it
        (GridData, SpectraData).
        """
        output = self._output(data)
        if unit is None or not isinstance(
                output, (GridData, SpectraData)):
            return output
        return output._replace(unit=unit)

    def _output(
            self, data: np.ndarray
    ) -> GridData | SpectraData | np.ndarray:
        """
        Return an array of the form of the data as an output, without a
        unit.

        Here, on the grid of the observation, with its world coordinates;
        kinds of other data override this.

        Parameters
        ----------
        data : np.ndarray
            The array: the model, its mask or weights, or a residual.

        Returns
        -------
        GridData or SpectraData or np.ndarray
            The output.
        """
        return GridData(
            data, self._grid.coords, self._grid.spectral_axis)

    def oversampling(self) -> tuple[int, ...]:
        """Return the oversampling of the model along each axis."""
        return self._sky_options.oversampling

    def dtype(self) -> np.dtype:
        """Return the floating type of the model."""
        return self._sky_options.dtype

    def fft_dtype(self) -> np.dtype:
        """Return the floating type of the FFT convolutions of the model."""
        return self._sky_options.fft_dtype

    def light_amplitudes(self, seen: SeenGalaxies) -> dict[str, str | None]:
        linear = self.linear_in_light()
        amplitudes = {}
        for g, i, component in self._scenes(seen).components():
            galaxy, mapping = seen.galaxies[g], seen.mappings[g]
            reasons = galaxy.light_amplitudes()
            names = galaxy.component_param_names(i)
            for param in component.light_amplitudes():
                reason = reasons[names[param]]
                if reason is None and not linear:
                    reason = (
                        f"the observation of type {self.type()!r} is not "
                        f"linear in the light (e.g. moments, ratios of sums "
                        f"of the light)")
                amplitudes[mapping[names[param]]] = reason
        return amplitudes

    def orbit_components(
            self, seen: SeenGalaxies
    ) -> tuple[tuple[int, int], ...]:
        return self._scenes(seen).orbit_emitters()

    def _scenes(self, seen: SeenGalaxies) -> SeenScenes:
        """
        Return the scenes of the galaxies it sees (see make_scenes).
        """
        return make_scenes(
            seen, self._components, self.view(), self.data_unit())

    def check(self, seen: SeenGalaxies) -> None:
        scenes = self._scenes(seen)
        options = self._sky_options
        for scene in scenes.scenes:
            # (the light of an image is blurred by its PSF and again by
            # ours)
            scene.check_psf(options.instrument.psf())
            check_scene(scene)
        if options.pixel_weights is not None:
            weights_geometry(options.pixel_weights, seen)
        self._check_scenes(scenes)

    def _check_scenes(self, scenes: SeenScenes) -> None:
        """
        Check that it can see its scenes (see make_scenes) through its
        instrument, with its flux scale of one value, if any, before any
        plan. Here, that it sees the stars on orbits whose weights are
        solved, if any (see orbit_stars); the kinds of moments also check
        that they see one line (see _moments.check_one_line), and
        ObservationRegionMoments more.

        Parameters
        ----------
        scenes : SeenScenes
            What is seen.

        Raises
        ------
        ValueError
            If it cannot.
        """
        if scenes.orbit_emitters() and self.orbit_stars is None:
            # (the registered kinds of observations on the sky)
            from ..observation import observation_parser
            seeing = sorted(
                f"{type_} (their {cls.orbit_stars})" for type_, cls
                in observation_parser.registered_classes().items()
                if issubclass(cls, ObservationSky)
                and cls.orbit_stars is not None)
            raise ValueError(
                f"stars on orbits whose weights are solved to fit the data "
                f"(e.g. of Schwarzschild kinematics) are seen by "
                f"{', '.join(seeing)}; {self.type()} does not see them")

    def plan(
            self,
            driver: Driver,
            seen: SeenGalaxies,
            reference: Reference | None
    ) -> ObservationPlan:
        """
        Plan the evaluation of what it sees (see ObservationGroup.plan):
        the scenes of the galaxies it sees, along its line of sight (see
        make_scenes), as its kind of data (see _plan).

        Parameters
        ----------
        driver : Driver
            The driver it is evaluated on.
        seen : SeenGalaxies
            The galaxies it sees.
        reference : Reference, optional
            The values of the parameters at which the grids of the planes
            behind lenses that it sees are laid out (see
            ParamSpace.reference), with the masses of galaxies there;
            required with them.

        Returns
        -------
        ObservationPlan
            The plan.

        Raises
        ------
        ValueError
            If what it sees does not go with it (see make_scenes), or with
            its kind (see _plan), or it sees planes behind lenses and has
            no reference values.
        """
        plan = self._plan(driver, SkyDrawing(
            self._scenes(seen), reference, self._measured_pixels))
        # (its kind applies a flux scale of one value, see __init__; one
        # of several values multiplies the items of light of its plan)
        scale = self.flux_scale()
        if scale is None or scale[1].axis is None:
            return plan
        return FluxScaledPlan(plan, driver, *scale, self._sky_options.dtype)

    @abc.abstractmethod
    def _measured_pixels(self) -> np.ndarray:
        """
        Return the pixels of the grid of the cube of its model (x and y,
        before the oversampling; see ModelCube) whose light its data
        measure: those unmasked of data on its pixels, those of its
        regions or slits; all of them without data, or for data that see
        the whole field (e.g. visibilities). From them, the rays through
        the lenses in front of what it sees are traced (see TracePlan).

        Returns
        -------
        np.ndarray
            The pixels, a bool array of shape (ny, nx).
        """

    @abc.abstractmethod
    def _plan(self, driver: Driver, drawing: SkyDrawing) -> ObservationPlan:
        """
        Plan the evaluation of what it draws (see SkyDrawing) as its kind
        of data, with its options of every observation on the sky: its
        depth axis, for a scene that needs one, and the grids of the
        planes behind lenses (see plan_scenes of
        gbkfit.observation.depth), its instrument, the oversampling and
        the floating types of its model, its pixel weights and its flux
        scale of one value, if any (see SkyOptions).

        Parameters
        ----------
        driver : Driver
            The driver it is evaluated on.
        drawing : SkyDrawing
            What it draws: the scenes of the galaxies it sees (the
            components of each whose light is seen), and what lays out
            the planes behind lenses.

        Returns
        -------
        ObservationPlan
            The plan.

        Raises
        ------
        ValueError
            If the scenes cannot be planned (see CubePlan).
        """


def _check_fft_dtype(dtype: np.dtype, fft_dtype: np.dtype) -> None:
    """
    Raise ValueError if the floating type of the FFT convolutions of a
    model is less precise than that of the model.
    """
    if fft_dtype.itemsize < dtype.itemsize:
        raise ValueError(
            f"the fft_dtype ({fft_dtype.name}) is less precise than the "
            f"dtype ({dtype.name})")
