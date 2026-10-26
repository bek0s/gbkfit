from collections.abc import Mapping, Sequence

import astropy.units
import numpy as np

from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Band',
    'Bands',
    'Line',
    'LineCatalogue',
    'LinePlacement',
    'Lines',
    'band_parser',
    'line_parser'
]


# The lines of the catalogue (see LineCatalogue): the rest wavelength in
# vacuum of the optical lines (those of the SDSS line list) and of the
# near-infrared hydrogen lines, and the rest frequency of the radio and
# (sub)millimetre lines. Optical lines are named after their wavelength in
# air, as usual.
_CATALOGUE_LINES = {
    'oii3726': '3727.092 Angstrom',
    'oii3729': '3729.875 Angstrom',
    'hbeta': '4862.683 Angstrom',
    'oiii4959': '4960.295 Angstrom',
    'oiii5007': '5008.240 Angstrom',
    'nii6548': '6549.86 Angstrom',
    'halpha': '6564.61 Angstrom',
    'nii6583': '6585.27 Angstrom',
    'sii6716': '6718.29 Angstrom',
    'sii6731': '6732.67 Angstrom',
    'paalpha': '18756.1 Angstrom',
    'brgamma': '21661.2 Angstrom',
    'hi21': '1420.405751768 MHz',
    'co10': '115.2712018 GHz',
    'co21': '230.538 GHz',
    'co32': '345.7959899 GHz',
    'cii158': '1900.5369 GHz'}

# The bands of the catalogue (see LineCatalogue): the effective wavelength
# of the filters of SDSS, 2MASS and of the IRAC channels 1 and 2 of
# Spitzer
_CATALOGUE_BANDS = {
    'sdss_u': '3551 Angstrom',
    'sdss_g': '4686 Angstrom',
    'sdss_r': '6166 Angstrom',
    'sdss_i': '7480 Angstrom',
    'sdss_z': '8932 Angstrom',
    'twomass_j': '1.235 um',
    'twomass_h': '1.662 um',
    'twomass_ks': '2.159 um',
    'irac1': '3.550 um',
    'irac2': '4.493 um'}


class Line(parseutils.Serializable):
    """
    An emission line of a spectral component.

    Parameters
    ----------
    name : str
        Its name, which names its flux ratio (see Lines).
    rest : str or Quantity
        Its rest wavelength (in vacuum) or frequency (see
        gridutils.make_rest).

    Raises
    ------
    ConfigError
        If the name or the rest is not valid.
    """

    def dump(self) -> dict[str, str]:
        return dict(name=self._name, rest=str(self._rest))

    def __init__(self, name: str, rest: str | astropy.units.Quantity):
        parseutils.check_name(name)
        self._name = name
        self._rest = gridutils.make_rest(rest)

    def name(self) -> str:
        """Return its name."""
        return self._name

    def rest(self) -> astropy.units.Quantity:
        """Return its rest wavelength or frequency."""
        return self._rest


line_parser = parseutils.BasicParser(Line)


class Band(parseutils.Serializable):
    """
    A band of images: a filter, of an effective wavelength.

    Parameters
    ----------
    name : str
        Its name, which names the flux ratio of the light of a component
        in it (see Bands).
    wavelength : str or Quantity
        Its effective wavelength (e.g. '6166 Angstrom').

    Raises
    ------
    ConfigError
        If the name or the wavelength is not valid.
    """

    def dump(self) -> dict[str, str]:
        return dict(name=self._name, wavelength=str(self._wavelength))

    def __init__(self, name: str, wavelength: str | astropy.units.Quantity):
        parseutils.check_name(name)
        error = ConfigError(
            f"the wavelength of a band must be a positive length with units "
            f"(e.g. '6166 Angstrom'); it is {wavelength!r}")
        try:
            wavelength = astropy.units.Quantity(wavelength)
        except (TypeError, ValueError) as e:
            raise error from e
        if not (wavelength.unit.is_equivalent(astropy.units.m)
                and wavelength.isscalar and wavelength.value > 0):
            raise error
        self._name = name
        self._wavelength = wavelength.to(astropy.units.m)

    def name(self) -> str:
        """Return its name."""
        return self._name

    def wavelength(self) -> astropy.units.Quantity:
        """Return its effective wavelength (m)."""
        return self._wavelength


band_parser = parseutils.BasicParser(Band)


class LineCatalogue:
    """
    The emission lines and the bands that configurations name: those of
    GBKFIT, and those of the user, which add to or replace them. The lines
    of GBKFIT are the brightest optical lines (oii3726, oii3729, hbeta,
    oiii4959, oiii5007, nii6548, halpha, nii6583, sii6716, sii6731), the
    near-infrared hydrogen lines paalpha and brgamma, and the radio and
    (sub)millimetre lines hi21, co10, co21, co32 and cii158; their rest
    wavelengths are in vacuum. Its bands are those of SDSS (sdss_u, sdss_g,
    sdss_r, sdss_i, sdss_z), of 2MASS (twomass_j, twomass_h, twomass_ks)
    and the IRAC channels 1 and 2 of Spitzer (irac1, irac2).

    Parameters
    ----------
    lines : Mapping of str to str or Quantity, optional
        The user's lines: the rest wavelength (in vacuum) or frequency of
        each, by name (see Line).
    bands : Mapping of str to str or Quantity, optional
        The user's bands: the effective wavelength of each, by name (see
        Band).

    Raises
    ------
    ConfigError
        If a name or a rest is not valid.

    Examples
    --------
    >>> catalogue = LineCatalogue({'mgii2796': '2796.35 Angstrom'})
    >>> catalogue.line('halpha').rest()
    <Quantity 6.56461e-07 m>
    """

    def __init__(
            self,
            lines: Mapping[str, str | astropy.units.Quantity] | None = None,
            bands: Mapping[str, str | astropy.units.Quantity] | None = None
    ):
        self._lines = {
            name: Line(name, rest)
            for name, rest in (_CATALOGUE_LINES | dict(lines or {})).items()}
        self._bands = {
            name: Band(name, wavelength)
            for name, wavelength in (
                _CATALOGUE_BANDS | dict(bands or {})).items()}

    def names(self) -> tuple[str, ...]:
        """Return the names of its lines."""
        return tuple(self._lines)

    def band_names(self) -> tuple[str, ...]:
        """Return the names of its bands."""
        return tuple(self._bands)

    def band(self, name: str) -> Band:
        """
        Return its band of the given name.

        Parameters
        ----------
        name : str
            The name.

        Returns
        -------
        Band
            The band.

        Raises
        ------
        ConfigError
            If it has no band of the name.
        """
        if name not in self._bands:
            raise ConfigError(
                f"there is no band {name!r} in the catalogue; its bands: "
                f"{list(self._bands)}")
        return self._bands[name]

    def line(self, name: str) -> Line:
        """
        Return its line of the given name.

        Parameters
        ----------
        name : str
            The name.

        Returns
        -------
        Line
            The line.

        Raises
        ------
        ConfigError
            If it has no line of the name.
        """
        if name not in self._lines:
            raise ConfigError(
                f"there is no line {name!r} in the catalogue; its lines: "
                f"{list(self._lines)}")
        return self._lines[name]


class Lines:
    """
    The emission lines of a spectral component. Without lines, it has one
    line at the velocity of the spectral axis, whatever the line is. With
    lines, each is at its place on the spectral axis, whose velocities
    refer to a rest (see gridutils.Coords), at the redshift of the galaxy
    of the component, and the flux of each line after the first, relative
    to the first, is a parameter ('{name}_ratio').

    Parameters
    ----------
    lines : Sequence of Line, optional
        The lines; one line at the velocity of the spectral axis if None.

    Raises
    ------
    ConfigError
        If there are no lines, or their names repeat.
    """

    def __init__(self, lines: Sequence[Line] | None):
        if lines is not None:
            lines = tuple(lines)
            if not lines:
                raise ConfigError("lines needs at least one line")
            names = [line.name() for line in lines]
            if repeated := sorted({n for n in names if names.count(n) > 1}):
                raise ConfigError(
                    f"the lines must have different names; repeated: "
                    f"{repeated}")
        self._lines = lines
        self._ratios = {} if lines is None else {
            f'{line.name()}_ratio': ParamScalarDesc(
                desc=f"the flux of line {line.name()} relative to line "
                     f"{lines[0].name()}")
            for line in lines[1:]}

    def lines(self) -> tuple[Line, ...] | None:
        """Return the lines, if given."""
        return self._lines

    def names(self) -> tuple[str, ...]:
        """Return the names of the lines (none without lines)."""
        return () if self._lines is None else tuple(
            line.name() for line in self._lines)

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        """Return the parameters of the lines: their flux ratios."""
        return self._ratios

    def select(self, names: Sequence[str] | None) -> tuple[int, ...]:
        """
        Return the indices of the lines of the given names (all if None). The
        one line of a component without lines is always selected. Raise
        ConfigError if none of the lines is selected.
        """
        if self._lines is None:
            return (0,)
        if names is None:
            return tuple(range(len(self._lines)))
        selected = tuple(
            i for i, name in enumerate(self.names()) if name in names)
        if not selected:
            raise ConfigError(
                f"a component of the lines {list(self.names())} has none of "
                f"the selected lines {list(names)}; leave it out with the "
                f"components of the observation")
        return selected

    def ratio_name(self, index: int) -> str | None:
        """
        Return the name of the flux ratio of the line of the given index:
        none for the first line, whose flux is that of the component.
        """
        return None if index == 0 else f'{self.names()[index]}_ratio'

    def factors(
            self, spectral: gridutils.Grid, redshift: float
    ) -> np.ndarray:
        """
        Return the 1 + z of each line on a spectral axis, of a component of
        a galaxy of the given redshift at the systemic velocity 0 (see
        LinePlacement, which places the lines at a systemic velocity): the
        wavelength of the line over the rest wavelength of the axis, or
        the rest frequency of the axis over the frequency of the line. A
        line of rest wavelength w is at w (1 + z), one of rest frequency f
        at f / (1 + z). The rest of the axis can be that of a line in the
        frame of the galaxy (a factor 1 + z) or as observed (1): its
        velocities are then near c z or near 0, and those of the galaxy
        are the same. One line without a name is at the rest of the axis
        (1), whatever the redshift.

        The rest wavelengths of the lines are in vacuum. If those of the
        axis are in air (see gridutils.Coords), w (1 + z) is that of each
        line in air: divided by the refractive index of air at its
        wavelength (in vacuum), with the galaxy at the velocity at which
        its first line is at the centre of the axis. (The index changes by
        less than 1e-7 across 2000 km/s.)

        Raises
        ------
        ConfigError
            If the lines have names, and the spectral axis has no rest.
        """
        if self._lines is None:
            return np.ones(1)
        rest = spectral.coords.rest
        if rest is None:
            raise ConfigError(
                "the lines of a component need the rest wavelength or "
                "frequency that the velocities of the spectral axis refer "
                "to: RESTWAV or RESTFRQ in the header of the data, or the "
                "rest option of the observable (spec_rest for moments and "
                "spectra in regions)")
        # (wavelengths in vacuum, but that of a rest wavelength in air)
        emitted = np.array([
            line.rest().to_value(astropy.units.m, astropy.units.spectral())
            for line in self._lines])
        factors = emitted * (1 + redshift) \
            / rest.to_value(astropy.units.m, astropy.units.spectral())
        if spectral.coords.medium == 'air':
            convention = gridutils.VelocityConvention.of(spectral.coords)
            centre = spectral.zero()[0] \
                + (spectral.size[0] - 1) / 2 * spectral.coords.step[0]
            # (1 + v / c of the galaxy at the velocity v at which its first
            # line is at the centre of the axis)
            shift = convention.factor(centre) / factors[0]
            factors = factors / np.array([
                gridutils.air_refractive_index(w * (1 + redshift) * shift)
                for w in emitted])
        return factors

    def frequencies(
            self, spectral: gridutils.Grid, redshift: float
    ) -> np.ndarray:
        """
        Return the frequency (Hz, in vacuum) at which a galaxy of the given
        redshift emits each line, in its rest frame: the rest of each, or
        for one line without a name, which is at the velocity of the
        spectral axis (see factors), the rest of the axis times 1 +
        redshift.

        Raises
        ------
        ConfigError
            If the line has no name, and the spectral axis has no rest, or
            the galaxy has a redshift.
        """
        if self._lines is not None:
            return np.array([
                line.rest().to_value(astropy.units.Hz, astropy.units.spectral())
                for line in self._lines])
        # (with a redshift, the rest of the axis can be that of the line as
        # emitted or as observed, and nothing tells which: the frequency
        # would be off by 1 + z, and the light dimmed by up to (1 + z)^5;
        # refused rather than guessed)
        if redshift != 0:
            raise ConfigError(
                "a line without a name cannot have a unit in a galaxy with "
                "a redshift: the rest of the spectral axis may be that of "
                "the line as emitted or as observed, which the unit needs; "
                "name the line (e.g. lines: [hi21])")
        rest = spectral.coords.rest
        if rest is None:
            raise ConfigError(
                "a line without a name is at the rest of the spectral axis, "
                "which its frequency needs: RESTWAV or RESTFRQ in the header "
                "of the data, or the rest option of the observable")
        frequency = rest.to_value(astropy.units.Hz, astropy.units.spectral())
        if spectral.coords.medium == 'air':
            frequency /= gridutils.air_refractive_index(
                rest.to_value(astropy.units.m))
        return np.array([frequency * (1 + redshift)])


class LinePlacement:
    """
    Where the lines of a component are on a spectral axis, with the
    component at a systemic velocity (see __call__): the offset and scale
    of each, such that a line of velocity v in the frame of the galaxy
    (the systemic velocity plus the motions within the galaxy, e.g. its
    rotation) is at offset + scale * v on the axis.

    The systemic velocity is the radial velocity of the component from the
    redshift of the galaxy, relativistic: a line is redshifted by D0
    doppler(vsys), D0 its 1 + z at the systemic velocity 0 (see
    Lines.factors), whatever the axis. Its centre is the velocity of the
    axis of that factor D: c (D - 1) on an axis of optical velocities (a
    rest wavelength), c ln D on one of log velocities (a rest wavelength,
    the axis uniform in ln w), c (1 - 1 / D) on one of radio velocities (a
    rest frequency), and c (D^2 - 1) / (D^2 + 1) on an axis without a rest,
    whose velocities are relativistic, as those of the galaxy (VELO); the
    convention of the axis says which (see gridutils.VelocityConvention).
    So one systemic velocity places a galaxy alike on axes of every
    convention (e.g. its cubes of HI and of Halpha; read in the convention
    of each axis, as an optical velocity on one and a radio one on the
    other, it would place them c (D_radio - D_optical), about vsys^2 / c,
    apart: 83 km/s at 5000 km/s).

    The systemic velocity stretches the motions about it: one km/s of the
    galaxy at the centre u spans 1 + u / c km/s of an axis of optical
    velocities, 1 - u / c km/s of one of radio velocities, and 1 - (u /
    c)^2 km/s of one of relativistic velocities (e.g. on an axis of HI at
    rest, with the galaxy at the redshift 0 and vsys = 10000 km/s, the
    rotation spans 3.3% fewer km/s of the axis than its own). The scale is
    that, at each line, and the offset keeps the line at u, so that the
    rotation and the dispersion are those in the rest frame of the galaxy.

    The motions about the systemic velocity are linear on the axis. The
    Doppler effect has terms of their second order too (v^2 / 2c: 0.15
    km/s at 300 km/s, the same on both sides of the galaxy, so that they
    do not change its rotation), which are left out.
    """

    def __init__(self, factors: np.ndarray, spectral: gridutils.Grid):
        self._convention = gridutils.VelocityConvention.of(spectral.coords)
        # The 1 + z of each line at the systemic velocity 0
        self._factors = np.array(factors, dtype=float)

    def __len__(self) -> int:
        return len(self._factors)

    def centres(self, vsys: float) -> np.ndarray:
        """
        Return the velocity of the axis at the centre of each line (km/s),
        with the component at the given systemic velocity (km/s).
        """
        return self._convention.velocity(
            self._factors * gridutils.doppler(vsys))

    def __call__(self, vsys: float) -> np.ndarray:
        """
        Return the offset and scale of each line (an array of shape
        (nlines, 2)), with the component at the given systemic velocity
        (km/s).
        """
        centres = self.centres(vsys)
        scales = self._convention.stretch(centres)
        return np.column_stack((centres - scales * vsys, scales))


class Bands:
    """
    The bands of a component that emits in bands. Without bands, its light
    is the same in every band. With bands, it emits in those only: its
    light in the first band is its density, and that in each other band,
    relative to the first, is a parameter ('{name}_ratio').

    Parameters
    ----------
    bands : Sequence of Band, optional
        The bands; every band if None.

    Raises
    ------
    ConfigError
        If there are no bands, or their names repeat.
    """

    def __init__(self, bands: Sequence[Band] | None):
        if bands is not None:
            bands = tuple(bands)
            if not bands:
                raise ConfigError("bands needs at least one band")
            names = [band.name() for band in bands]
            if repeated := sorted({n for n in names if names.count(n) > 1}):
                raise ConfigError(
                    f"the bands must have different names; repeated: "
                    f"{repeated}")
        self._bands = bands
        self._ratios = {} if bands is None else {
            f'{band.name()}_ratio': ParamScalarDesc(
                desc=f"the flux in band {band.name()} relative to that in "
                     f"band {bands[0].name()}")
            for band in bands[1:]}

    def bands(self) -> tuple[Band, ...] | None:
        """Return the bands, if given."""
        return self._bands

    def names(self) -> tuple[str, ...]:
        """Return the names of the bands (none without bands)."""
        return () if self._bands is None else tuple(
            band.name() for band in self._bands)

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        """Return the parameters of the bands: their flux ratios."""
        return self._ratios

    def emits_in(self, band: Band | None) -> bool:
        """
        Check whether the component emits in a band (that of an image;
        None if not known).

        Raises
        ------
        ConfigError
            If the band is not known, and the component has bands.
        """
        if self._bands is None:
            return True
        if band is None:
            raise ConfigError(
                f"a component emits in the bands {list(self.names())}; the "
                f"image that sees it needs a band")
        return band.name() in self.names()

    def ratio_name(self, band: Band | None) -> str | None:
        """
        Return the name of the flux ratio of the light in a band in which
        the component emits (see emits_in): none in its first band, or
        without bands.
        """
        names = self.names()
        if not names or band.name() == names[0]:
            return None
        return f'{band.name()}_ratio'
