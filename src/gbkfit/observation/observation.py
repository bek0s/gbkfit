"""
The base of the observations: how the model of galaxies gives data of
one kind, alone or paired with a dataset (see Observation), the galaxies
an observation sees (see SeenGalaxies), and the frame of an observation
on the sky (see SkyFrame).
"""

import abc
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, Self

import astropy.units
import numpy as np
from astropy.table import Table

from gbkfit.config import (
    InputError, TypedParser, TypedSerializable, check_name, constructor_of,
    make_typed_desc, parse_options_for_callable)
from gbkfit.cosmology import Cosmology
from gbkfit.dataset import Data, Dataset
from gbkfit.driver import Driver
from gbkfit.grid import GridData, SpectraData
from gbkfit.io import Beam
from gbkfit.lensing import Tracer
from gbkfit.model import DataUnit, DepthUnit, Galaxy, Reference
from gbkfit.params import ParamDesc
from .noise import Noise
from .plan import FluxScale, ObservationPlan

if TYPE_CHECKING:
    from ._frames import FrameShift


__all__ = [
    'Observation',
    'SeenGalaxies',
    'SeenPlane',
    'SkyFrame',
    'observation_parser'
]


_log = logging.getLogger(__name__)


class SeenPlane(NamedTuple):
    """
    A plane of the galaxies an observation sees (see SeenGalaxies), drawn
    as one scene (see Scene): its galaxies, by their positions among those
    the observation sees, their order along the line of sight, front to
    back (in units of their positions among those of the plane, see
    DepthUnit; None if not given), and the plane of the tracer of the
    group whose rays draw them (see Tracer), or None for the image plane.

    Without lenses, an observation sees one plane, of all the galaxies it
    sees, in the depth order of the group. With lenses, the planes of the
    tracer up to the first with lenses are one, the image plane, whose
    light no lens bends: their galaxies by redshift, those of each in the
    depth order. Each plane of the tracer behind it is a plane of its
    own, whose light the lenses in front bend (see Tracer.lensed). The
    planes without galaxies seen are left out.
    """
    galaxies: tuple[int, ...]
    order: tuple[DepthUnit, ...] | None
    traced: int | None


class SeenGalaxies(NamedTuple):
    """
    The galaxies an observation sees (see ObservationGroup and its
    option seen): the galaxies, the components of each that it sees, by
    their indices (None for all whose light it sees), the mapping of the
    parameters of each to those of the observation group, their labels
    (the id 'galaxy' of their extra outputs), their planes, near to far (see SeenPlane), the RA and Dec
    (degrees) of the origin of the frame of their parameters (that of the
    first observation on the sky; None without one), their cosmology (see
    ObservationGroup), the paths of the light of the planes of the group
    through its lenses, if it has any (see Tracer), the mapping of the
    parameters of each lens of the tracer to those of the group, and how
    the observation moves what it sees into its frame (see FrameShift):
    the rays of its pixels meet the lenses moved as the galaxies.
    """
    galaxies: tuple[Galaxy, ...]
    components: tuple[tuple[int, ...] | None, ...]
    mappings: tuple[dict[str, str], ...]
    labels: tuple[str, ...]
    planes: tuple[SeenPlane, ...]
    sky_origin: tuple[float, float] | None
    cosmology: Cosmology
    tracer: Tracer | None
    lens_mappings: tuple[dict[str, str], ...]
    shift: 'FrameShift'

    def lensed(self) -> bool:
        """Check whether the light of a plane it sees is bent by a lens."""
        return any(plane.traced is not None for plane in self.planes)


class SkyFrame(NamedTuple):
    """
    The frame of the model of an observation on the sky (see
    Observation.sky_frame), in which it sees the galaxies (see
    ObservationGroup).

    Attributes
    ----------
    origin : tuple of float
        The RA and Dec (degrees) of its origin, from which the centres of
        what the observation sees are measured (e.g. the reference pixel
        of the grid of its data, see ObservationSky.sky_rval).
    offsets : tuple of (str or None)
        The names of the parameters of the offsets of the frame, of x, y
        and the velocity (see ObservationSky): None for those it does not
        fit.
    velocity_frame : str or None
        The frame of rest of its velocities (FITS SPECSYS, see FRAMES), if
        known.
    """
    origin: tuple[float, float]
    offsets: tuple[str | None, str | None, str | None]
    velocity_frame: str | None


class Observation(TypedSerializable, abc.ABC):
    """
    How the model of galaxies gives data of one kind (the data model): what
    the plan of an observation group evaluates (see ObservationGroup.plan).
    Its kind says where the data are measured and how the model gives them
    (e.g. on the sky, through an instrument; see ObservationSky). Its
    configuration has no data: an observation alone renders its model (e.g.
    to draw mock data of it, see draw_mocks).

    Paired with a dataset of its kind (see paired, with_dataset and load;
    in configurations, the section 'datasets' of a setup, by observation
    name), it is fitted to those data: an objective compares its model
    with them under the likelihood its setup gives it (see Likelihoods and
    Objective). The data win at the pairing: the options that they give
    (see options_from_data; e.g. the grid of the model) are theirs, and
    those given are used only without data (they are left out, with a
    line in the log); the data are checked against the others (e.g. the
    data items, and the correlation of their noise).

    Its data are named items of one shape (see keys). Its kind can have
    parameters of its own, named after it (see pdescs), e.g. a flux scale
    of its model (see flux_scale).

    Parameters
    ----------
    keys : Sequence of str
        The names of its data items (e.g. 'spectra').
    noise : Noise, optional
        The noise of its mock data (see draw_mocks), added to what its
        instrument records (e.g. a cube, of which moments are measured).
        By default, Gaussian noise of the errors of its data, if it is
        paired with data, whose errors are of what it records (e.g.
        spectra, an image).
    name : str, optional
        Its name, which is the id of its extra outputs instead of its
        position (see ObservationGroup.observation_label), starts the
        names of its parameters, and keys the options of setups given by
        observation (e.g. its dataset and its likelihood, see
        Likelihoods).
    model_can_have_missing_values : bool, optional
        Whether its model can have missing values (NaN) where the data
        have some (e.g. the velocity of a spectrum below its flux_floor):
        its likelihood then needs a residual for them (see
        LikelihoodGauss). False by default.

    Raises
    ------
    ValueError
        If the name is invalid.
    """

    # The class of the datasets of its kind, which it is paired with
    dataset_class: ClassVar[type[Dataset]]

    config_titles = dict(noise='Noise', name='Name')

    @classmethod
    def load(
            cls, info: dict[str, Any], dataset: Dataset | None = None
    ) -> Self:
        """
        Load an observation from its configuration, paired with a dataset
        if given (see the class): the options that the data give are then
        theirs, and those its configuration needs without data (e.g. the
        grid of its model) can be left out.

        Parameters
        ----------
        info : dict
            The configuration.
        dataset : Dataset, optional
            The data it is fitted to (in configurations, those of the
            section 'datasets' of its setup).

        Returns
        -------
        Observation
            The observation.

        Raises
        ------
        ConfigError
            If the configuration has an error.
        ValueError, TypeError
            If an object it makes refuses its options, or the data are not
            of it (see paired).
        """
        if dataset is None:
            return super().load(info)
        func, _ = constructor_of(cls, info)
        return cls.paired(dataset, **parse_options_for_callable(info, func))

    def __init__(
            self,
            keys: Sequence[str],
            noise: Noise | None = None,
            name: str | None = None,
            model_can_have_missing_values: bool = False
    ):
        keys = tuple(keys)
        check_name(name)
        self._keys = keys
        # (its data, once it is paired with them: see _pair)
        self._dataset: Dataset | None = None
        self._noise = noise
        self._name = name
        self._model_can_have_missing_values = model_can_have_missing_values

    @classmethod
    @abc.abstractmethod
    def options_from_data(cls, dataset: Dataset) -> tuple[str, ...]:
        """
        Return the options of an observation of its kind that a dataset
        gives (e.g. the grid of its model): those of the data, when it is
        paired with them (see the class).

        Parameters
        ----------
        dataset : Dataset
            The data.

        Returns
        -------
        tuple of str
            The names of the options.
        """

    @classmethod
    @abc.abstractmethod
    def _options_from_data(
            cls, dataset: Dataset, options: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Return the values of the options of an observation of its kind
        that a dataset gives (see options_from_data), and of those of its
        options that the data complete (e.g. the spectral axis of moment
        maps, which covers the velocities of the data), as arguments of
        its constructor.

        Parameters
        ----------
        dataset : Dataset
            The data, of the dataset class of its kind.
        options : dict
            Its other options, as given (those of its constructor).

        Returns
        -------
        dict
            The options.
        """

    @classmethod
    @abc.abstractmethod
    def data_options(
            cls,
            dataset: Dataset,
            prefix: str | None = None,
            dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        """
        Return the options of an observation of its kind that a dataset
        gives (see options_from_data) as a configuration, as far as the
        data alone give them (not those they complete): e.g. to copy the
        sampling of data into an observation that renders without them, as
        those data were measured.

        Parameters
        ----------
        dataset : Dataset
            The data.
        prefix : str, optional
            The start of the names of the files of the options that are in
            files (e.g. regions in bins; see dump), which are written; by
            default, those options are left out (nothing is written).
        dump_path : bool, optional
            Whether the configuration has the paths of the files, or only
            their names.
        overwrite : bool, optional
            Whether to overwrite existing files.

        Returns
        -------
        dict
            The options.

        Raises
        ------
        ValueError
            If the data are not of the dataset class of its kind.
        """

    @classmethod
    def check_dataset_class(cls, dataset: Dataset) -> None:
        """
        Check that a dataset is of the dataset class of its kind (see
        dataset_class), which an observation of its kind can be paired with
        (see paired).

        Parameters
        ----------
        dataset : Dataset
            The data.

        Raises
        ------
        InputError
            If it is not.
        """
        if not isinstance(dataset, cls.dataset_class):
            raise InputError(
                f"data of {dataset.type_title().lower()} · needs "
                f"{cls.dataset_class.type_title().lower()}")

    @classmethod
    def paired(cls, dataset: Dataset, /, **options: Any) -> Self:
        """
        Make an observation of its kind of the given options paired with a
        dataset (see the class; in configurations, see load): the options
        that the data give are theirs, and those given are left out, with
        a line in the log; those needed without data (e.g. the grid of its
        model) can be left out. E.g. one observation fitted to the data of
        many galaxies, each with its own grid, is made of its options for
        each dataset.

        Parameters
        ----------
        dataset : Dataset
            The data, of the dataset class of its kind.
        **options
            The options of its constructor.

        Returns
        -------
        Observation
            The observation.

        Raises
        ------
        InputError
            If the data are not of the dataset class of its kind.
        ValueError, TypeError
            If the observation refuses its options, or the data are not of
            it (see _pair).

        Examples
        --------
        >>> observation = ObservationPixelSpectra.paired(
        ...     DatasetPixelSpectra.from_files('cube.fits'),
        ...     instrument=Instrument(psf=PSFGauss(1.5)))
        """
        cls.check_dataset_class(dataset)
        from_data = cls.options_from_data(dataset)
        if given := [key for key in from_data if options.get(key) is not None]:
            _log.info(
                "%s: its data give the options %s; those given are used "
                "only without data", make_typed_desc(cls, 'observation'),
                given)
        own = {
            key: value for key, value in options.items()
            if key not in from_data}
        observation = cls(**own | cls._options_from_data(dataset, own))
        observation._pair(dataset)
        return observation

    def _pair(self, dataset: Dataset) -> None:
        """
        Pair it with a dataset, which gave it the options that data give
        (see paired): check the data against its other options, and take
        them. Here, the noise of each data item must be correlated as its
        kind of data can have it (see Dataset.noise_correlation); kinds
        check more (e.g. ObservationSky, and the beams of the data against
        its PSF).

        Raises
        ------
        ValueError
            If the data are not of it.
        """
        for key in self._keys:
            dataset.noise_correlation(key)
        self._dataset = dataset

    @abc.abstractmethod
    def _constructor_options(self) -> dict[str, Any]:
        """
        Return the options of its constructor that make it again, as it
        has them (with those its data gave, if any; see with_dataset).
        """

    def name(self) -> str | None:
        """Return its name, if any."""
        return self._name

    def dataset(self) -> Dataset | None:
        """Return the data it is paired with, if any (see with_dataset)."""
        return self._dataset

    def noise(self) -> Noise | None:
        """
        Return the noise of its mock data, if given (see draw_mocks: by
        default, that of the errors of its data).
        """
        return self._noise

    def pdescs(self) -> dict[str, ParamDesc]:
        """
        Return its parameters, named after it: none here; kinds add their
        own (e.g. its flux scale, see ObservationSky).
        """
        return {}

    def sky_frame(self) -> SkyFrame | None:
        """
        Return the frame of its model on the sky, in which it sees the
        galaxies (see SkyFrame and ObservationGroup); here, None: its data
        are not on the sky.
        """
        return None

    def takes_mass(self) -> bool:
        """
        Check whether its model takes the mass of the galaxies it sees
        beyond what their components take of it (e.g. profiles of their
        circular velocity, see ObservationProfile); here, it does not.
        """
        return False

    def keys(self) -> tuple[str, ...]:
        """Return the names of its data items (e.g. 'spectra')."""
        return self._keys

    def model_can_have_missing_values(self) -> bool:
        """
        Check whether its model can have missing values (NaN) where the
        data have some: the likelihood then needs a residual for them (see
        LikelihoodGauss).
        """
        return self._model_can_have_missing_values

    def flux_scale(self) -> tuple[str, FluxScale] | None:
        """
        Return the parameter of the flux scale of its model and how it
        scales the model (see ObservationSky), if it has one; none here.
        """
        return None

    @abc.abstractmethod
    def make_dataset(self, items: Mapping[str, Data]) -> Dataset:
        """
        Make the dataset of data items measured as this observation (e.g.
        mock data), where its model is.

        Parameters
        ----------
        items : Mapping
            The data items, by name (see keys).

        Returns
        -------
        Dataset
            The dataset.
        """

    @abc.abstractmethod
    def data_unit(self) -> DataUnit | None:
        """
        Return the unit of the light of its data, which its model takes;
        None without data, or without data of light.
        """

    def model_unit(
            self, key: str
    ) -> tuple[astropy.units.UnitBase | None, Beam | None]:
        """
        Return the unit of the values of a data item of its model, and
        their beam, of a unit per beam: those that data of the model
        have (e.g. mock data; see draw_mocks).

        Here, those of its data, if any (None for those they lack), and
        none without data; kinds whose model has a unit without data add
        it (see ObservationSky).

        Parameters
        ----------
        key : str
            The name of the data item (see keys).

        Returns
        -------
        tuple
            The unit, or None, and the beam, or None.
        """
        if self._dataset is None:
            return None, None
        item = self._dataset[key]
        return item.unit(), item.beam()

    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        """Return the floating type of its model."""

    def with_dataset(self, dataset: Dataset) -> Self:
        """
        Return a copy of it paired with a dataset (see the class): the
        options that the data give are theirs (e.g. the grid of its model),
        and the others are those it has. With data of its own options (e.g.
        mock data drawn on its grid, see draw_mocks, or those of a
        bootstrap), the copy is the same observation with other data. Of
        its other options, those that took values from its data (e.g. the
        channels of the cube of moment maps, which cover the velocities of
        the data) keep them; its options paired with other data (see
        paired and load) take theirs.

        Parameters
        ----------
        dataset : Dataset
            The data, of the dataset class of its kind.

        Returns
        -------
        Observation
            The copy.

        Raises
        ------
        InputError
            If the data are not of the dataset class of its kind.
        ValueError
            If the data are not of it (see _pair).
        """
        from_data = self.options_from_data(dataset)
        return self.paired(dataset, **{
            key: value for key, value in self._constructor_options().items()
            if key not in from_data})

    def same_plan(self, other: 'Observation', seen: SeenGalaxies) -> bool:
        """
        Check whether its plan is also that of another observation, the
        same one with other data (see with_dataset), which can then share
        it (see GroupPlan.with_datasets): whether all that its plan of the
        galaxies it sees takes from its data is the same in the other.

        Here, the unit of their light (see data_unit) and the floating
        type of their models, which the data can choose (see dtype);
        kinds whose plans take more from their data compare that too (see
        ObservationProfile, and ObservationSky).

        Parameters
        ----------
        other : Observation
            The other observation.
        seen : SeenGalaxies
            The galaxies they see.

        Returns
        -------
        bool
            True if the plan is also that of the other.
        """
        return _same_unit(self.data_unit(), other.data_unit()) \
            and self.dtype() == other.dtype()

    def light_amplitudes(self, seen: SeenGalaxies) -> dict[str, str | None]:
        """
        Return the amplitudes of the light that its model takes (see
        Component.light_amplitudes), by their names in its observation
        group: those of the components whose light it sees, of the
        galaxies it sees, each with None if its model is linear in it, or
        why not (also when it is not linear in the light: see
        ObservationSky.linear_in_light); none here, where it sees no light.
        """
        return {}

    def orbit_components(
            self, seen: SeenGalaxies
    ) -> tuple[tuple[int, int], ...]:
        """
        Return the components of the galaxies it sees whose stars it sees
        on orbits whose weights are solved (see Component.orbit_weights;
        with data, a recording of the orbits models it, see
        GroupPlan.orbit_series), each as the index of its galaxy (among
        those it sees) and its own index; none here, where it sees no
        light.
        """
        return ()

    @abc.abstractmethod
    def check(self, seen: SeenGalaxies) -> None:
        """
        Check that it can see the galaxies it sees, without a driver (see
        ObservationGroup, which calls this for each observation): every
        rule of its configuration about what it sees is raised here, before
        any plan, which then fails only for resources.

        Parameters
        ----------
        seen : SeenGalaxies
            The galaxies it sees.

        Raises
        ------
        ValueError
            If it cannot see them.
        """

    @abc.abstractmethod
    def plan(
            self,
            driver: Driver,
            seen: SeenGalaxies,
            reference: Reference | None
    ) -> ObservationPlan:
        """
        Plan the evaluation of its model (see ObservationGroup.plan).

        Parameters
        ----------
        driver : Driver
            The driver it is evaluated on.
        seen : SeenGalaxies
            The galaxies it sees.
        reference : Reference, optional
            The values of the parameters of the group at which what is
            laid out once for all evaluations is laid out (see
            ParamSpace.reference; e.g. the grids of the planes whose light
            a lens bends), with the masses on the sky there of the
            galaxies whose masses lenses are; required when it sees such
            planes.

        Returns
        -------
        ObservationPlan
            The plan.

        Raises
        ------
        ValueError
            If it needs the reference values and has none.
        """

    @abc.abstractmethod
    def output(
            self,
            data: np.ndarray,
            unit: astropy.units.UnitBase | None = None
    ) -> GridData | SpectraData | Table | np.ndarray:
        """
        Return an array of the form of its data as an output (e.g. on the
        grid of its data, with their world coordinates).

        Parameters
        ----------
        data : np.ndarray
            The array: the model, its mask or weights, or a residual.
        unit : astropy.units.UnitBase, optional
            The unit of its values, if known (e.g. that of the data, and of
            the model, which is in it), which the output takes.

        Returns
        -------
        GridData or SpectraData or Table or np.ndarray
            The output.
        """

    def images(
            self,
            driver: Driver,
            values: Mapping[str, np.ndarray],
            weights: Mapping[str, np.ndarray] | None,
            units: Mapping[str, astropy.units.UnitBase | None] | None
    ) -> dict[str, GridData]:
        """
        Return images of values of its data items, as outputs (e.g. the
        dirty images of visibilities; see ObservationVisibilities.images),
        made on a driver; here, none.

        Parameters
        ----------
        driver : Driver
            The driver they are made on.
        values : Mapping
            The values of each data item (e.g. those of the data, of the
            model, or of the data minus the model), NaN where they have
            none.
        weights : Mapping, optional
            The weights of the values of each data item (e.g. 1 / error^2);
            all 1 if None.
        units : Mapping, optional
            The unit of the values of each data item, if known.

        Returns
        -------
        dict
            The images, by name.
        """
        return {}

    def psf_images(
            self,
            driver: Driver,
            weights: Mapping[str, np.ndarray] | None
    ) -> dict[str, GridData]:
        """
        Return images of the PSF of its data (e.g. the dirty beam of
        visibilities; see ObservationVisibilities.psf_images), which its
        images are seen through, made on a driver; here, none.

        Parameters
        ----------
        driver : Driver
            The driver they are made on.
        weights : Mapping, optional
            The weights of the values of each data item (see images).

        Returns
        -------
        dict
            The images, by name.
        """
        return {}


def _same_unit(a: DataUnit | None, b: DataUnit | None) -> bool:
    """
    Return whether two units of the light of data are the same for a plan
    (see DataUnit, whose beams can be arrays, one for each spectral
    point). Data without a unit of their light are as no data: the light
    of the components is taken as it is (and their components have no
    unit, which the check of an observation makes sure of; see
    check_data_unit), e.g. mock data of an observation without data.
    """
    a, b = (None if x is None or x.unit is None else x for x in (a, b))
    if a is None or b is None:
        return a is b
    return all(np.array_equal(x, y) if isinstance(x, np.ndarray)
               else x == y for x, y in zip(a, b))


observation_parser = TypedParser(Observation, family='observation')
