"""
The base of the observations on the sky (see ObservationSky): galaxies
seen through an instrument as one kind of data (images, moment maps,
spectral cubes and others), along a line of sight, through a foreground.
"""

import abc
import logging
from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING, Annotated, Any, ClassVar, Literal, NamedTuple, Self,
    TypeAlias)

import astropy.units
import numpy as np

import gbkfit.units
from gbkfit.config import (
    Choices, check_name, config_path, load_option_and_update_info,
    make_typed_desc)
from gbkfit.dataset import Dataset, dataset_parser
from gbkfit.driver import Driver
from gbkfit.grid import (
    Grid, GridData, SpectraData, SpectralAxis, check_convention,
    make_cube_grid, make_grid, per_axis)
from gbkfit.instrument import Instrument, instrument_parser
from gbkfit.io import Beam
from gbkfit.model import (
    Band, DataUnit, LightUnit, Scene, View, band_parser)
from gbkfit.model.components.lines import catalogue_in_use
from gbkfit.params import ParamDesc, ParamScalarDesc, ParamVectorDesc
from .._scene import check_component_names, make_scene
from ..depth import Depth, check_scene
from ..foreground import Foreground, foreground_parser
from ..observation import Observation, SeenGalaxies, SkyFrame
from ..plan import FluxScale, ObservationPlan
from ..weights import Weights, weights_geometry
from ._checks import (
    check_beams, check_flux_scale, check_offsets, has_correlated_noise,
    light_unit)
from ._flux_scale import FluxScaledPlan
from ._options import GRID_OPTIONS, dump_grid, without_options_from_data


_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gbkfit.dataset import Data


__all__ = [
    'ObservationSky'
]


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
    foreground : Foreground
        What the light meets before the telescope (e.g. a lens).
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
    foreground: Foreground
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
    cube), optionally with its data. They are seen along a line of sight,
    through a foreground (e.g. a gravitational lens) and the instrument,
    and an objective compares the model of the galaxies with the data
    under the likelihood of its setup (see Observation).

    Each kind of data is a subclass, which gives the form of the data
    (their items, see keys) and where they are measured (e.g. their grid),
    declares the index of the spectral axis of its data (spectral_axis,
    None if none) and the class of its datasets (dataset_class). Its
    options are those of its kind, then those of every observation on the
    sky (below).

    Its data (the option dataset), of the dataset class of its kind, give
    the options of its kind of options_from_data (e.g. its grid; see
    _options_from_data). The configuration can give those too, for the
    observation without data (e.g. one that renders galaxies on its own
    grid, and is fitted to data of any grid): with data, they are left
    out, with a line in the log. Without data, those of
    config_required_without (by 'dataset') are required, and the others
    take their values of defaults_without_data.
    The files of its options (e.g. regions without data) are named with
    the prefix of the dump, as those of the datasets.

    Its model is in the unit of its data (see data_unit): the light of
    the components whose emission has a unit is turned into it (see
    Component.plan), and that of the others is in it. Without data, the
    model is in the unit of the components it sees, which must then have
    one (or none).

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
        it sees, and the masses of its lens (deflection maps if shifted,
        see LensDeflectionMap). None by default; a galaxy needs an
        observation without them (see ObservationGroup).
    dataset : Dataset, optional
        The data, of the dataset class of its kind.
    oversampling : Sequence of int, optional
        How many times the model is oversampled along each axis of the
        data (an accuracy setting); 1 by default. The smooth components
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
        otherwise. A copy with other data (see with_dataset, e.g. mock
        data) then chooses again, by them.
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
        option of its kind needed without data is missing, the data are
        not of it (see check_dataset), the noise of an item is correlated
        where its kind of data cannot have it, its kind cannot be seen
        through the instrument (see _check_options), the dtype or the
        fft_dtype is not one of the native modules (see Driver.dtypes), or
        the fft_dtype is less precise than the dtype, it has pixel weights
        and a lens (not supported yet), or the unit is given with data, is
        not a unit, or is per beam without a beam (see light_unit).
    """

    spectral_axis: ClassVar[int | None]

    dataset_class: ClassVar[type[Dataset]]

    # Whether it sees a spectral axis (a spectral cube and what is made of
    # one), or a band (an image); see view
    spectral_view: ClassVar[bool]

    # What it sees of the stars on orbits whose weights are solved (see
    # Component.orbit_weights and _check_scene): their light ('light', of
    # an image), their moments ('moments', see ObservationRegionMoments),
    # or nothing
    orbit_stars: ClassVar[Literal['light', 'moments'] | None]

    # The options of its kind that its data give (see options_from_data):
    # those required without data (see Serializable), and the values of
    # the others without data, when not given
    config_required_without = dict(dataset=('size',))
    defaults_without_data: ClassVar[Mapping[str, Any]] = dict(
        step=(1, 1), rval=(0, 0), rota=0)

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        """
        Load an observation from its configuration.

        Its data are a dataset of the dataset class of its kind (see e.g.
        DatasetPixelSpectra.from_files, whose options outliers and
        mask_whole_spectra clean them), which gives it some of the options
        of its kind (see options_from_data).

        Parameters
        ----------
        info : dict
            The configuration.

        Returns
        -------
        ObservationSky
            The observation.

        Raises
        ------
        ConfigError
            If the configuration has an error.
        ValueError, TypeError
            If an object it makes refuses its options.
        """
        info = dict(info)
        load_option_and_update_info(dataset_parser, info, 'dataset')
        return super().load(info)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        """
        Dump the observation to its configuration: its type, the options
        of its kind but those its data give, if any (see
        options_from_data), and those of every observation on the sky.

        Parameters
        ----------
        prefix : str, optional
            The start of the names of the files of its data and of the
            data of its parts (e.g. images; see Dataset.dump), and of its
            options (e.g. the table of a trace).
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
        dataset = {} if self._dataset is None else dict(
            dataset=dataset_parser.dump(self._dataset, **kwargs))
        # (only a given dtype: one the data chose is chosen again by the
        # data it is loaded with, e.g. mock data of correlated noise)
        dtype = {} if self._given_dtype is None else dict(
            dtype=self._given_dtype)
        return dict(type=self.type()) | name \
            | self._dump_kind(prefix, dump_path, overwrite) | galaxies \
            | components | lines | continuum | band | pixel_weights \
            | flux_scale \
            | offsets | unit | dataset | dtype | dict(
                depth=options.depth.dump(),
                foreground=foreground_parser.dump(
                    options.foreground, **kwargs),
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
        Return the options of its kind that a dataset gives.

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

    def _options(
            self, dataset: Dataset | None, options: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Return the options of its kind (see the class), as given (None
        where not given), completed, before it is made.

        With data, those that the data give (see options_from_data) are
        theirs (see _options_from_data): those of them given are for the
        observation without data, and are left out (with a line in the
        log). Without data, those of config_required_without are
        required, and those not given take their values of
        defaults_without_data.

        Parameters
        ----------
        dataset : Dataset, optional
            The data, if any.
        options : dict
            The options of its kind, as given.

        Returns
        -------
        dict
            The options of its kind.

        Raises
        ------
        ValueError
            If one required without data (see config_required_without) is
            not given without data, or the dataset is not of the dataset
            class of its kind.
        """
        desc = make_typed_desc(type(self), 'observation')
        if dataset is None:
            if missing := [
                    key for key in self.config_required_without['dataset']
                    if options.get(key) is None]:
                raise ValueError(
                    f"{desc} without data needs the options {missing}")
            return options | {
                key: value for key, value in self.defaults_without_data.items()
                if options.get(key) is None}
        self._check_dataset_class(dataset)
        from_data = self.options_from_data(dataset)
        if given := [key for key in from_data if options.get(key) is not None]:
            _log.info(
                "%s: its data give the options %s; those given are used "
                "only without data", desc, given)
        options = {
            key: None if key in from_data else value
            for key, value in options.items()}
        return options | self._options_from_data(dataset, options)

    @classmethod
    @abc.abstractmethod
    def _options_from_data(
            cls, dataset: Dataset, options: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Return the options of its kind that a dataset gives (see
        options_from_data), and those of its options that the data
        complete (e.g. the spectral axis of moment maps, which covers the
        velocities of the data).

        Parameters
        ----------
        dataset : Dataset
            The data, of the dataset class of its kind.
        options : dict
            The options of its kind, as given (see _options).

        Returns
        -------
        dict
            The options.
        """

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
            foreground: Foreground | None,
            instrument: Instrument | None,
            galaxies: Sequence[str] | None,
            components: Sequence[str] | None,
            lines: Sequence[str] | None,
            continuum: bool,
            band: str | Band | None,
            pixel_weights: Weights | None,
            flux_scale: str | None,
            offsets: Mapping[str, bool] | None,
            dataset: Dataset | None,
            oversampling: Sequence[int] | None,
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
        ndim = len(self.size())
        oversampling = tuple(oversampling) if oversampling is not None \
            else (1,) * ndim
        if len(oversampling) != ndim or any(s < 1 for s in oversampling):
            raise ValueError(
                f"oversampling must have {ndim} values, each at least 1; it "
                f"is {oversampling}")
        if dataset is not None:
            self.check_dataset(dataset)
        self._components = components
        self._lines = lines
        self._continuum = continuum
        self._band = band
        self._flux_scale = flux_scale
        self._offsets = offsets
        super().__init__(
            self.keys(), galaxies, dataset, name,
            self.model_can_have_missing_values())
        # (as given, for copies of other data and dumps: by default, the
        # data choose it, see with_dataset)
        self._given_dtype = dtype
        if dtype is None:
            dtype = 'float64' if dataset is not None \
                and has_correlated_noise(dataset) else 'float32'
        dtype, fft_dtype = np.dtype(dtype), np.dtype(fft_dtype)
        if fft_dtype.itemsize < dtype.itemsize:
            raise ValueError(
                f"the fft_dtype ({fft_dtype.name}) is less precise than the "
                f"dtype ({dtype.name})")
        # (a flux scale of one value scales the flux of the model where it
        # records it; one of several values, its items of light, see plan)
        scale = self.flux_scale()
        one = scale is not None and scale[1].axis is None
        self._sky_options = SkyOptions(
            depth=depth if depth is not None else Depth(),
            foreground=foreground,
            instrument=instrument, oversampling=oversampling, dtype=dtype,
            fft_dtype=fft_dtype, pixel_weights=pixel_weights,
            flux_scale_param=scale[0] if one else None)
        self._check_options(self._sky_options)
        if pixel_weights is not None and foreground.lens() is not None:
            raise ValueError("a lens does not support pixel_weights yet")
        # (the unit of the model without data, and its beam, of a unit per
        # beam: those of data of the model, e.g. mock data)
        self._light, self._light_beam = (None, None) if unit is None \
            else light_unit(unit, dataset, self, instrument)
        if dataset is not None:
            check_beams(dataset, instrument, self)

    def with_dataset(self, dataset: Dataset) -> Self:
        """
        Return a copy of it with other data (e.g. mock data; see
        draw_mocks): its kind keeps its options (e.g. its grid; see
        _kind_options), which the data must have (see check_dataset).

        Parameters
        ----------
        dataset : Dataset
            The data, of the observation.

        Returns
        -------
        ObservationSky
            The copy.

        Raises
        ------
        ValueError
            If the data are not of the observation (see check_dataset).
        """
        # (the same observation with other data, e.g. mock data drawn on
        # its grid: the data must be of its options, and the copy takes
        # those that they give from them)
        self.check_dataset(dataset)
        from_data = self.options_from_data(dataset)
        options = self._sky_options
        return type(self)(
            **{key: value for key, value in self._kind_options().items()
               if key not in from_data},
            foreground=options.foreground, instrument=options.instrument,
            galaxies=self._galaxies, components=self._components,
            lines=self._lines, continuum=self._continuum, band=self._band,
            pixel_weights=options.pixel_weights, flux_scale=self._flux_scale,
            offsets=self._offsets, dataset=dataset,
            oversampling=options.oversampling, depth=options.depth,
            dtype=self._given_dtype, fft_dtype=options.fft_dtype.name,
            name=self._name)

    @abc.abstractmethod
    def _kind_options(self) -> dict[str, Any]:
        """
        Return the options of its kind (those of its __init__ but those of
        every observation on the sky), all of them, which make it again
        (see with_dataset): with data, those its data gave too.
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
        data (see _options_from_data).

        Parameters
        ----------
        items : Mapping
            The data items, by name (see keys).

        Returns
        -------
        Dataset
            The dataset, of the dataset class of its kind.
        """

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
        self._check_dataset_class(dataset)
        if set(dataset.keys()) != set(self.keys()):
            desc = make_typed_desc(self.__class__, 'observation')
            raise ValueError(
                f"{desc} has the data items {sorted(self.keys())}, but the "
                f"data have {sorted(dataset.keys())}")
        self._check_dataset_coordinates(dataset)

    @classmethod
    def _check_dataset_class(cls, dataset: Dataset) -> None:
        """Raise ValueError unless the dataset is of the dataset class."""
        if not isinstance(dataset, cls.dataset_class):
            desc = make_typed_desc(cls, 'observation')
            dataset_desc = make_typed_desc(dataset.__class__, 'dataset')
            raise ValueError(f"{desc} cannot be compared with {dataset_desc}")

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
        offsets of its frame, if any (see the class), and those of its
        lens, '{name}_lens_{parameter}' (see Foreground.pdescs).
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
        return flux_scale | offsets | self._sky_options.foreground.pdescs()

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
        (see sky_rval), its offsets those it fits (see the class), the
        frame of rest of its velocities that of its spectral axis (see
        SpectralAxis), if it has one that gives it, and what it adds the
        masses of its lens (see Foreground.centre_params and
        Foreground.shift_params).
        """
        spectral = self.spectral()
        foreground = self._sky_options.foreground
        return SkyFrame(
            self.sky_rval(), self._offset_params(),
            None if spectral is None else spectral.frame(),
            foreground.centre_params(), foreground.shift_params())

    def flux_scale(self) -> tuple[str, FluxScale] | None:
        """
        Return the parameter of its flux scale and how it scales its model
        (see the class), if it has one.
        """
        if self._flux_scale is None:
            return None
        return (f'{self._name}_flux_scale',
                self.flux_scale_kinds()[self._flux_scale])

    def foreground(self) -> Foreground:
        """Return the foreground."""
        return self._sky_options.foreground

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
        for galaxy, mapping, emitters in zip(
                seen.galaxies, seen.mappings, self._scene(seen).emitters()):
            reasons = galaxy.light_amplitudes()
            for i in emitters:
                names = galaxy.component_param_names(i)
                for param in galaxy.components()[i].light_amplitudes():
                    reason = reasons[names[param]]
                    if reason is None and not linear:
                        reason = (
                            f"the observation of type {self.type()!r} is "
                            f"not linear in the light (e.g. moments, ratios "
                            f"of sums of the light)")
                    amplitudes[mapping[names[param]]] = reason
        return amplitudes

    def orbit_components(
            self, seen: SeenGalaxies
    ) -> tuple[tuple[int, int], ...]:
        return self._scene(seen).orbit_emitters()

    def _scene(self, seen: SeenGalaxies) -> Scene:
        """
        Return the scene of the galaxies it sees (see make_scene).
        """
        return make_scene(
            seen, self._components, self.view(), self.data_unit())

    def check(self, seen: SeenGalaxies) -> None:
        scene = self._scene(seen)
        options = self._sky_options
        # (the light of an image is blurred by its PSF and again by ours)
        scene.check_psf(options.instrument.psf())
        if options.pixel_weights is not None:
            weights_geometry(options.pixel_weights, scene)
        check_scene(scene)
        self._check_scene(scene)

    def _check_scene(self, scene: Scene) -> None:
        """
        Check that it can see a scene (see make_scene) through its
        foreground and its instrument, with its flux scale of one value,
        if any, before any plan. Here, that it sees the stars on orbits
        whose weights are solved, if any (see orbit_stars); the kinds of
        moments also check that they see one line (see
        _moments.check_one_line), and ObservationRegionMoments more.

        Parameters
        ----------
        scene : Scene
            What is seen.

        Raises
        ------
        ValueError
            If it cannot.
        """
        if scene.orbit_emitters() and self.orbit_stars is None:
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

    def plan(self, driver: Driver, seen: SeenGalaxies) -> ObservationPlan:
        """
        Plan the evaluation of what it sees (see ObservationGroup.plan):
        the scene of the galaxies it sees, along its line of sight (see
        make_scene), as its kind of data (see _plan).

        Parameters
        ----------
        driver : Driver
            The driver it is evaluated on.
        seen : SeenGalaxies
            The galaxies it sees.

        Returns
        -------
        ObservationPlan
            The plan.

        Raises
        ------
        ValueError
            If what it sees does not go with it (see make_scene), or with
            its kind (see _plan).
        """
        plan = self._plan(driver, self._scene(seen))
        # (its kind applies a flux scale of one value, see __init__; one
        # of several values multiplies the items of light of its plan)
        scale = self.flux_scale()
        if scale is None or scale[1].axis is None:
            return plan
        return FluxScaledPlan(plan, driver, *scale, self._sky_options.dtype)

    @abc.abstractmethod
    def _plan(self, driver: Driver, scene: Scene) -> ObservationPlan:
        """
        Plan the evaluation of a scene (what it sees) as its kind of data,
        with its options of every observation on the sky: its depth axis,
        for a scene that needs one (see plan_scene of
        gbkfit.observation.depth), its foreground, its instrument, the
        oversampling and the floating types of its model, its pixel
        weights and its flux scale of one value, if any (see SkyOptions).

        Parameters
        ----------
        driver : Driver
            The driver it is evaluated on.
        scene : Scene
            What is seen: the galaxies, and the components of each whose
            light is seen.

        Returns
        -------
        ObservationPlan
            The plan.

        Raises
        ------
        ValueError
            If the pixel weights are not in the plane of a disk of the
            scene (see CubePlan).
        """
